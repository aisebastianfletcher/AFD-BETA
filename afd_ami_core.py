"""
AFDInfinityAMI - AFD-driven assistant core.

Features:
- Rule-based neutralizer for LLM-free AFD mode.
- High-variance, seedable AFD renderer (temperature + seed).
- Memory persistence (CSV) and explainability JSONL.
- Deterministic AFD reflection and optional human-style post-hoc reflection (hidden in UI by default).
- Safe OpenAI/HF fallbacks when renderer_mode == 'llm'.
"""
import os
import json
import collections
import re
import numpy as np
import pandas as pd
import streamlit as st
import openai
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

try:
    from openai.error import AuthenticationError as OpenAIAuthError
except Exception:
    OpenAIAuthError = Exception


class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None, temperature: float = 0.0, memory_window: int = 10):
        # runtime params
        self.temperature = float(temperature or 0.0)
        self.memory_window = int(memory_window or 10)

        # OpenAI key normalization (do not print keys)
        raw_key = openai_api_key or None
        try:
            if raw_key is None and hasattr(st, "secrets"):
                raw_key = st.secrets.get("OPENAI_API_KEY") or raw_key
        except Exception:
            raw_key = raw_key
        if raw_key is None:
            raw_key = os.getenv("OPENAI_API_KEY")

        api_key = None
        if raw_key:
            api_key = str(raw_key).strip()
            if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
                api_key = api_key[1:-1].strip()

        self.use_openai = bool(api_key) or bool(use_openai)
        if api_key:
            openai.api_key = api_key

        self.reflection_log = []

        # Lightweight OpenAI test recorded only in reflection_log
        if self.use_openai and api_key:
            try:
                self._test_openai_key()
                self.reflection_log.append("OpenAI key test succeeded.")
            except OpenAIAuthError as e:
                self.use_openai = False
                self.reflection_log.append(f"OpenAI auth failed during init: {e}")
            except Exception as e:
                self.use_openai = False
                self.reflection_log.append(f"OpenAI test call failed during init: {e}")

        # persistence
        self.memory_file = "data/response_log.csv"
        self.explain_file = "data/explain_log.jsonl"
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        if not os.path.exists(self.memory_file):
            try:
                pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence"]).to_csv(
                    self.memory_file, index=False, encoding="utf-8-sig"
                )
            except Exception as e:
                self.reflection_log.append(f"Error creating memory file: {e}")

        # coefficients
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5

        # prompts
        self.neutralizer_system = (
            "You are a precise neutral translator. Convert the user's input into a short, "
            "neutral, factual description suitable for algorithmic processing. "
            "Do NOT add opinions or extra context. Keep it concise."
        )
        self.renderer_system = (
            "You are a factual renderer. Produce a clear, neutral, concise explanation "
            "based ONLY on the provided neutral input and the AFD directives below."
        )

        # lazy HF resources
        self._hf_llm = None
        self.sentiment_analyzer = None
        self.latest_explainability = None

    def _test_openai_key(self):
        # minimal call; exceptions bubbled to caller
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping for auth test. Reply with 'ok'."}],
            max_tokens=1,
            temperature=0.0,
        )
        return True

    #
    # HF caching (lazy)
    #
    @st.cache_resource
    def _cache_llm(_self):
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = model.config.eos_token_id
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    @st.cache_resource
    def _cache_sentiment_analyzer(_self):
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def _ensure_hf_llm(self):
        if self._hf_llm is None:
            try:
                self._hf_llm = self._cache_llm()
            except Exception as e:
                self.reflection_log.append(f"Could not init HF llm: {e}")
                self._hf_llm = None
        return self._hf_llm

    def _ensure_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = self._cache_sentiment_analyzer()
            except Exception as e:
                self.reflection_log.append(f"Could not init sentiment analyzer: {e}")
                self.sentiment_analyzer = None
        return self.sentiment_analyzer

    #
    # Rule-based neutralizer (no LLM) used for 'afd' renderer mode
    #
    def _rule_neutralize(self, user_input: str) -> str:
        s = (user_input or "").strip()
        if not s:
            return ""
        core = s.rstrip(" ?!.")
        low = core.lower()
        m = re.match(r"^(what|who|when|where|why|how)\s+(.+)", low)
        if m:
            qword = m.group(1)
            rest = core[len(qword):].strip()
            if qword == "what":
                return f"definition/explanation request: {rest}"
            if qword == "who" and "you" in rest:
                return "agent identity inquiry"
            return f"query about: {rest}"
        m2 = re.match(r"^(explain|describe|summarize|compare|list)\s+(.+)", low)
        if m2:
            verb = m2.group(1)
            target = m2.group(2).rstrip(" ?!.")
            return f"{verb} {target}"
        trimmed = core.strip()
        if len(trimmed) > 150:
            trimmed = trimmed[:147].rstrip() + "…"
        return f"neutral summary: {trimmed}"

    #
    # LLM neutralizers / renderers (used only when renderer_mode == 'llm')
    #
    def _neutralize_with_openai(self, user_input):
        try:
            messages = [
                {"role": "system", "content": self.neutralizer_system},
                {"role": "user", "content": user_input},
            ]
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.0, max_tokens=120)
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return resp.choices[0].get("text", "").strip()
        except OpenAIAuthError as e:
            self.reflection_log.append(f"OpenAI neutralize auth error: {e}; falling back to HF.")
            self.use_openai = False
            self._ensure_hf_llm()
            return self._neutralize_with_hf(user_input)
        except Exception as e:
            self.reflection_log.append(f"OpenAI neutralize error: {e}")
            return ""

    def _neutralize_with_hf(self, user_input):
        llm = self._ensure_hf_llm()
        if llm is None:
            self.reflection_log.append("HF neutralizer unavailable.")
            return ""
        prompt = f"{self.neutralizer_system}\n\nUser: {user_input}\n\nNeutral:"
        try:
            out = llm(prompt, max_new_tokens=80, do_sample=False, return_full_text=False, num_return_sequences=1)
            text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF neutralize error: {e}")
            return ""

    def _render_with_openai(self, neutral_input, afd_directives, max_tokens=250):
        content = f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."
        try:
            messages = [
                {"role": "system", "content": self.renderer_system},
                {"role": "user", "content": content},
            ]
            temp = max(0.0, min(float(self.temperature), 2.0))
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=temp, max_tokens=max_tokens)
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return resp.choices[0].get("text", "").strip()
        except OpenAIAuthError as e:
            self.reflection_log.append(f"OpenAI render auth error: {e}; falling back to HF.")
            self.use_openai = False
            self._ensure_hf_llm()
            return self._render_with_hf(neutral_input, afd_directives)
        except Exception as e:
            self.reflection_log.append(f"OpenAI render error: {e}")
            return "Unable to render response using OpenAI."

    def _render_with_hf(self, neutral_input, afd_directives):
        llm = self._ensure_hf_llm()
        if llm is None:
            return "Local model not available to render."
        prompt = f"{self.renderer_system}\n\nNeutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nAnswer:"
        try:
            temp = float(self.temperature)
            gen_args = {
                "prompt": prompt,
                "max_new_tokens": 220,
                "do_sample": True if temp > 0 else False,
                "return_full_text": False,
                "num_return_sequences": 1,
            }
            if temp > 0:
                gen_args.update({"temperature": float(min(temp, 2.0)), "top_k": 40, "top_p": 0.9})
            out = llm(**gen_args)
            text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            if text.startswith(prompt):
                text = text[len(prompt) :]
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF render error: {e}")
            return "Unable to render response locally."

    #
    # AFD core math
    #
    def predict_next_state(self, state, action):
        return state + np.random.normal(0, 0.1, state.shape)

    def compute_harmony(self, state, interp_s):
        return np.linalg.norm(interp_s - state) / (np.linalg.norm(state) + 1e-10)

    def compute_info_gradient(self, state, interp_s):
        return np.abs(interp_s - state).sum() / (np.linalg.norm(state) + 1e-10)

    def compute_oscillation(self, state, interp_s):
        return np.std(interp_s - state)

    def compute_potential(self, s_prime):
        return np.linalg.norm(s_prime) / 10.0

    def coherence_score(self, action, state):
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)
        h = self.compute_harmony(state, interp_s)
        i = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        score = self.alpha * h + self.beta * i - self.gamma * o + self.delta * phi
        return float(score), {"harmony": float(h), "info_gradient": float(i), "oscillation": float(o), "potential": float(phi)}

    def adjust_coefficients(self, coherence, metrics):
        log = f"Coherence: {coherence:.4f}, Metrics: {metrics}"
        if coherence < 0.5:
            self.alpha += 0.05
            self.reflection_log.append(f"Increased alpha to {self.alpha:.2f} for better harmony. {log}")
        elif coherence > 0.9:
            self.gamma += 0.05
            self.reflection_log.append(f"Increased gamma to {self.gamma:.2f} to reduce oscillation. {log}")
        else:
            self.reflection_log.append(f"No adjustment needed. {log}")

    #
    # Persistence
    #
    def save_memory(self, prompt, neutral_prompt, response, coherence):
        try:
            df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
            new_row = pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp.utcnow().isoformat()],
                    "prompt": [prompt],
                    "neutral_prompt": [neutral_prompt],
                    "response": [response],
                    "coherence": [coherence],
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.memory_file, index=False, encoding="utf-8-sig")
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not save to CSV ({e}).")

    def append_explainability(self, explainability: dict):
        try:
            os.makedirs(os.path.dirname(self.explain_file) or ".", exist_ok=True)
            with open(self.explain_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(explainability, ensure_ascii=False) + "\n")
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not append explainability JSONL ({e}).")

    def load_memory(self):
        try:
            return pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception as e:
            self.reflection_log.append(f"Error loading memory file: {e}")
            return pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence"])

    def get_last_explainability(self):
        return getattr(self, "latest_explainability", None)

    def get_latest_reflection(self):
        return self.reflection_log[-1] if self.reflection_log else "No reflections yet."

    #
    # Memory summarize
    #
    def _summarize_memory(self, n=None):
        n = int(n) if n is not None else int(self.memory_window)
        df = self.load_memory()
        if df.empty:
            return {"count": 0, "avg_coherence": None, "last_neutral_prompts": [], "top_tokens": []}
        tail = df.tail(n)
        avg_coh = None
        try:
            avg_coh = float(tail["coherence"].astype(float).mean())
        except Exception:
            avg_coh = None
        last_neutrals = tail["neutral_prompt"].fillna("").astype(str).tolist()
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "of", "to", "is", "are", "for", "with", "that", "this"}
        token_counter = collections.Counter()
        for s in last_neutrals:
            tokens = re.findall(r"[A-Za-z]{3,}", s.lower())
            for t in tokens:
                if t not in stopwords:
                    token_counter[t] += 1
        top_tokens = [tok for tok, _ in token_counter.most_common(8)]
        return {"count": len(tail), "avg_coherence": avg_coh, "last_neutral_prompts": last_neutrals, "top_tokens": top_tokens}

    #
    # High-variance, seedable AFD renderer (no LLM)
    #
    def _afd_renderer(self, neutral_prompt, afd_directives, metrics, coherence, memory_summary=None, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        temp = float(max(0.0, min(self.temperature, 2.5)))
        noise_scale = max(0.0, temp * 0.07)

        metrics_noisy = {}
        for k, v in metrics.items():
            try:
                metrics_noisy[k] = float(v) + float(rng.normal(0.0, noise_scale))
            except Exception:
                metrics_noisy[k] = float(v)

        c = float(coherence) + float(rng.normal(0.0, noise_scale))
        harmony = metrics_noisy.get("harmony", 0.0)
        info_grad = metrics_noisy.get("info_gradient", 0.0)
        oscill = metrics_noisy.get("oscillation", 0.0)
        potential = metrics_noisy.get("potential", 0.0)

        # fragment pools for variety
        openings = [
            "In brief:",
            "To summarize briefly:",
            "A short take:",
            "Concise summary:",
            "Here's a succinct view:",
            "Quickly put:",
            "Shortly:",
        ]
        middles = [
            f"{neutral_prompt}.",
            f"{neutral_prompt} This is the essential restatement.",
            f"{neutral_prompt} (neutralized summary).",
            f"{neutral_prompt} — core summary.",
            f"{neutral_prompt}; distilled.",
        ]
        closers = [
            "This evaluation supports a direct explanation.",
            "The metrics indicate a cautious framing is appropriate.",
            "I frame the answer to balance clarity and caution.",
            "I will emphasize main points while avoiding speculation.",
            "This suggests a focused, careful summary.",
        ]

        templates = []
        for op in openings:
            for mid in middles:
                for cl in closers:
                    templates.append(f"{op} {mid} {cl}")
        extra_templates = [
            f"{neutral_prompt}. From the AFD perspective, coherence={c:.3f} and this guides the tone.",
            f"{neutral_prompt} — AFD coherence {c:.3f}; the assistant adapts tone and detail.",
            f"Neutral summary: {neutral_prompt}. AFD coherence={c:.3f}, shaping the response.",
        ]
        templates.extend(extra_templates)

        paraphrases = {
            "present": ["provide", "offer", "deliver"],
            "concise": ["brief", "succinct", "compact"],
            "cautious": ["careful", "measured", "guarded"],
            "uncertainty": ["lack of strong consensus", "unclear signals", "uncertain evidence"],
        }

        if temp <= 0.0:
            template_idx = 0
        else:
            base = np.ones(len(templates))
            mix = min(1.0, temp / 1.5)
            weights = (1.0 - mix) * base + mix * (np.arange(len(templates)) * 0 + 1.0)
            weights = weights / weights.sum()
            template_idx = int(rng.choice(len(templates), p=weights))

        template = templates[template_idx]

        if temp > 0.45:
            for key, opts in paraphrases.items():
                if key in template and rng.random() < min(0.9, temp):
                    choice = rng.choice(opts)
                    template = template.replace(key, choice, 1)

        # rationale parts
        rationale_parts = []
        if c >= 0.8:
            rationale_parts.append("High coherence → deliver a direct, confident explanation.")
        elif c <= 0.35:
            rationale_parts.append("Low coherence → emphasize uncertainty and avoid strong claims.")
        else:
            rationale_parts.append("Moderate coherence → provide a balanced, cautious explanation.")

        if harmony > 0.6:
            rationale_parts.append("High harmony → highlight consistent points.")
        elif harmony < 0.2:
            rationale_parts.append("Low harmony → clarify ambiguities.")

        if info_grad > 0.5:
            rationale_parts.append("High information gradient → include supporting detail.")
        else:
            rationale_parts.append("Low information gradient → favor concision.")

        if oscill > 0.2:
            rationale_parts.append("High oscillation → avoid speculation.")

        if memory_summary and memory_summary.get("avg_coherence") is not None:
            avg = memory_summary.get("avg_coherence")
            rationale_parts.append(f"Recent memory avg coherence={avg:.3f}, used to calibrate tone.")

        if temp > 0.8:
            rng.shuffle(rationale_parts)

        extra_sentence = ""
        if temp >= 1.3:
            style_options = [
                "I'll be explicit about caveats and assumptions.",
                "I'll foreground the most robust points first, then add nuance.",
                "I'll give a short answer up front followed by a concise elaboration.",
            ]
            extra_sentence = str(rng.choice(style_options))

        # compose answer sentences
        answer_sentences = []
        if c >= 0.8:
            answer_sentences.append(f"A concise synthesis: {neutral_prompt}.")
            if info_grad > 0.4:
                answer_sentences.append("Key supporting points are included where helpful.")
        elif c <= 0.35:
            answer_sentences.append(f"A cautious synthesis: {neutral_prompt}.")
            answer_sentences.append("Information is not strongly aligned; I avoid definitive claims.")
        else:
            answer_sentences.append(f"A balanced synthesis: {neutral_prompt}.")
            if info_grad > 0.4:
                answer_sentences.append("I include a couple of clarifying details to support the summary.")

        if memory_summary and memory_summary.get("top_tokens") and temp > 0.3:
            tokens = memory_summary.get("top_tokens")[:4]
            if tokens:
                answer_sentences.append("Related recent topics: " + ", ".join(tokens) + ".")

        if temp > 1.0:
            rng.shuffle(answer_sentences)

        lines = []
        lines.append(template)
        lines.append("")
        lines.append("AFD rationale (derived):")
        for p in rationale_parts:
            lines.append(f"- {p}")
        if extra_sentence:
            lines.append("")
            lines.append(extra_sentence)
        lines.append("")
        lines.append("Answer:")
        for s in answer_sentences:
            lines.append(f"- {s}")

        return "\n".join(lines)

    #
    # Human-style reflection (post-hoc)
    #
    def generate_human_style_reflection(self, neutral_prompt, metrics, coherence, memory_summary=None):
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0))
        ig = float(metrics.get("info_gradient", 0.0))
        o = float(metrics.get("oscillation", 0.0))

        identity = (
            "I am an algorithmic assistant. I don't have consciousness or feelings, "
            "but I can reflect on my assessment and describe how it shapes my answer."
        )

        if c >= 0.80:
            confidence = "I feel confident about the information I will present."
        elif c <= 0.35:
            confidence = "I sense uncertainty and will be cautious in my statements."
        else:
            confidence = "I have moderate confidence and will balance clarity with caution."

        harmony_note = (
            "Internal signals show agreement, so a coherent picture is available."
            if h > 0.6
            else "There are mixed internal signals; I will highlight ambiguities."
        )
        detail_note = "I'll include supporting detail." if ig > 0.5 else "I'll keep the explanation concise."
        stability_note = "Internal state is reasonably stable; I won't speculate." if o <= 0.2 else "Internal state shows some instability; I'll avoid speculation."

        neutral_echo = neutral_prompt if len(neutral_prompt) <= 200 else neutral_prompt[:197] + "…"

        paragraphs = []
        paragraphs.append(identity)
        paragraphs.append("")
        paragraphs.append(f"- Neutral input: {neutral_echo}")
        paragraphs.append(f"- Assessment: coherence={c:.3f}. {confidence}")
        paragraphs.append(f"- Observations: {harmony_note} {detail_note} {stability_note}".strip())

        if memory_summary and memory_summary.get("count", 0) > 0:
            mem_count = memory_summary.get("count")
            avg_coh = memory_summary.get("avg_coherence")
            top = ", ".join(memory_summary.get("top_tokens", [])[:5])
            paragraphs.append("")
            mem_line = f"I recall {mem_count} recent interactions; average coherence {avg_coh:.3f}."
            if top:
                mem_line += f" Recent topics include: {top}."
            paragraphs.append(mem_line)

        paragraphs.append("")
        paragraphs.append("This summary explains how my internal AFD evaluation shapes the tone and caution level of the forthcoming answer.")
        return "\n".join(paragraphs)

    #
    # Structured AFD reflection
    #
    def generate_afd_reflection(self, prompt, neutral_prompt, state, action, s_prime, interp_s, metrics, coherence, renderer_used):
        lines = []
        lines.append("AFD Reflection:")
        lines.append(f"- original_prompt: {prompt}")
        lines.append(f"- neutral_prompt: {neutral_prompt}")
        lines.append(f"- state (proxy): {np.array(state).tolist()}")
        lines.append(f"- action proxy: {np.array(action).tolist()}")
        lines.append(f"- predicted s_prime: {np.array(s_prime).tolist()}")
        lines.append(f"- interp_s: {np.array(interp_s).tolist()}")
        lines.append(
            f"- afd_metrics: harmony={metrics.get('harmony'):.6f}, info_gradient={metrics.get('info_gradient'):.6f}, "
            f"oscillation={metrics.get('oscillation'):.6f}, potential={metrics.get('potential'):.6f}"
        )
        lines.append(f"- coherence: {coherence:.6f}")
        if coherence < 0.4:
            lines.append("- Rationale: Low coherence -> prioritized neutrality and highlighted uncertainty.")
        elif coherence > 0.8:
            lines.append("- Rationale: High coherence -> confident, concise explanation.")
        else:
            lines.append("- Rationale: Moderate coherence -> balanced, cautious explanation.")
        lines.append(f"- renderer_used: {renderer_used}")
        return "\n".join(lines)

    def _build_renderer_prompt(self, neutral_input, afd_directives):
        return f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."

    #
    # Main respond
    #
    def respond(self, prompt, renderer_mode="afd", include_memory=True, seed=None):
        # neutralize (LLM-free for afd)
        if renderer_mode == "afd":
            neutral_prompt = self._rule_neutralize(prompt)
        else:
            if self.use_openai:
                neutral_prompt = self._neutralize_with_openai(prompt)
            else:
                neutral_prompt = self._neutralize_with_hf(prompt)
        if not neutral_prompt:
            neutral_prompt = prompt.strip()

        # memory summary
        memory_summary = None
        if include_memory:
            try:
                memory_summary = self._summarize_memory(self.memory_window)
            except Exception as e:
                self.reflection_log.append(f"Memory summary error: {e}")
                memory_summary = None

        # sentiment/state proxy
        try:
            if self.use_openai and renderer_mode != "llm":
                sentiment_score = 0.5
                sentiment_label = "NEUTRAL"
            else:
                sent_an = self._ensure_sentiment_analyzer()
                if sent_an:
                    sent = sent_an(neutral_prompt)[0]
                    sentiment_score = float(sent.get("score", 0.5))
                    sentiment_label = sent.get("label", "NEUTRAL")
                else:
                    sentiment_score = 0.5
                    sentiment_label = "NEUTRAL"
        except Exception as e:
            self.reflection_log.append(f"Sentiment error: {e}")
            sentiment_score = 0.5
            sentiment_label = "NEUTRAL"

        state = np.array([sentiment_score] * 5)
        action = np.array([1 if str(sentiment_label).upper().startswith("POS") else -1] * 5)

        # AFD math
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)
        harmony = self.compute_harmony(state, interp_s)
        info_grad = self.compute_info_gradient(state, interp_s)
        oscill = self.compute_oscillation(state, interp_s)
        potential = self.compute_potential(s_prime)
        coherence = float(self.alpha * harmony + self.beta * info_grad - self.gamma * oscill + self.delta * potential)
        metrics = {"harmony": float(harmony), "info_gradient": float(info_grad), "oscillation": float(oscill), "potential": float(potential)}

        # small memory-driven coefficient nudges
        try:
            if memory_summary and memory_summary.get("avg_coherence") is not None:
                avg_hist = memory_summary.get("avg_coherence")
                if avg_hist < 0.45:
                    self.alpha += 0.02
                    self.reflection_log.append(f"Adjusted alpha due to memory avg_coherence={avg_hist:.3f}")
                elif avg_hist > 0.8:
                    self.gamma += 0.02
                    self.reflection_log.append(f"Adjusted gamma due to memory avg_coherence={avg_hist:.3f}")
        except Exception:
            pass

        self.adjust_coefficients(coherence, metrics)

        afd_directives = (
            f"AFD metrics:\n"
            f"- coherence: {coherence:.6f}\n"
            f"- harmony: {metrics['harmony']:.6f}\n"
            f"- info_gradient: {metrics['info_gradient']:.6f}\n"
            f"- oscillation: {metrics['oscillation']:.6f}\n"
            f"- potential: {metrics['potential']:.6f}\n"
            f"Coefficients: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}\n"
            "The renderer must use ONLY the neutral input and the numeric AFD metrics above to construct the response."
        )
        renderer_prompt = self._build_renderer_prompt(neutral_prompt, afd_directives)

        # render
        if renderer_mode == "afd":
            final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence, memory_summary=memory_summary, seed=seed)
            renderer_used = "AFD (deterministic)"
        else:
            if self.use_openai:
                final_text = self._render_with_openai(neutral_prompt, afd_directives)
                renderer_used = "OpenAI"
            else:
                final_text = self._render_with_hf(neutral_prompt, afd_directives)
                renderer_used = "HF"

        # persist
        try:
            self.save_memory(prompt, neutral_prompt, final_text, coherence)
        except Exception:
            pass

        explainability = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "original_prompt": prompt,
            "neutral_prompt": neutral_prompt,
            "neutralizer_system": self.neutralizer_system,
            "renderer_system": self.renderer_system,
            "renderer_used": renderer_used,
            "renderer_prompt": renderer_prompt,
            "sentiment_label": sentiment_label,
            "sentiment_score": float(sentiment_score),
            "state": state.tolist(),
            "action": action.tolist(),
            "s_prime": s_prime.tolist(),
            "interp_s": interp_s.tolist(),
            "afd_metrics": metrics,
            "coherence": float(coherence),
            "coefficients": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta},
            "final_text": final_text,
            "memory_summary": memory_summary,
        }

        human_reflection = self.generate_human_style_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        explainability["human_reflection"] = human_reflection

        self.latest_explainability = explainability
        try:
            self.append_explainability(explainability)
        except Exception:
            pass

        afd_reflection = self.generate_afd_reflection(
            prompt, neutral_prompt, state, action, s_prime, interp_s, metrics, coherence, renderer_used
        )
        try:
            self.reflection_log.append(afd_reflection)
        except Exception:
            self.reflection_log = [afd_reflection]

        return final_text, coherence, afd_reflection
