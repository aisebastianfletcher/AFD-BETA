"""
AFDInfinityAMI - AFD-driven assistant core with memory-aware reflections and temperature variability.

Features:
- Deterministic AFD renderer (default) with controlled variability via temperature.
- Human-style, post-hoc reflection derived from AFD internals and recent memory (NOT chain-of-thought).
- Memory persistence: CSV memory log + JSONL explainability log.
- Memory summarization used to influence reflections and small coefficient adjustments.
- Lazy HF init and safe OpenAI handling.
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

# Best-effort import of OpenAI auth error type
try:
    from openai.error import AuthenticationError as OpenAIAuthError
except Exception:
    OpenAIAuthError = Exception


class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None, temperature: float = 0.0, memory_window: int = 10):
        # Store temperature and memory window
        self.temperature = float(temperature or 0.0)
        self.memory_window = int(memory_window or 10)

        # Read OpenAI key from param, streamlit secrets, or env
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

        # Files and persistence
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

        # AFD coefficients
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5

        # system prompts
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
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping for auth test. Reply with 'ok'."}],
            max_tokens=1,
            temperature=0.0,
        )
        return True

    #
    # Lazy HF initializers and caches
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
    # Neutralizers and renderers (OpenAI & HF fallbacks)
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
            self.reflection_log.append(f"OpenAI neutralize auth error: {e}. Falling back to HF.")
            self.use_openai = False
            self._ensure_hf_llm()
            return self._neutralize_with_hf(user_input)
        except Exception as e:
            self.reflection_log.append(f"OpenAI neutralize error: {e}")
            return ""

    def _neutralize_with_hf(self, user_input):
        llm = self._ensure_hf_llm()
        if llm is None:
            self.reflection_log.append("HF neutralizer not available (llm init failed).")
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
            self.reflection_log.append(f"OpenAI render auth error: {e}. Falling back to HF.")
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
    # Memory and explainability persistence
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
    # Memory summarization helper (simple, fast)
    #
    def _summarize_memory(self, n=None):
        """
        Return a small summary dictionary of the last n memory rows:
        - avg_coherence, count, last_prompts (neutral), top_tokens
        """
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

        # naive token counting, filter short tokens and common stopwords
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "of", "to", "is", "are", "for", "with", "that", "this"}
        token_counter = collections.Counter()
        for s in last_neutrals:
            # simple tokenization
            tokens = re.findall(r"[A-Za-z]{3,}", s.lower())
            for t in tokens:
                if t not in stopwords:
                    token_counter[t] += 1
        top_tokens = [tok for tok, _ in token_counter.most_common(8)]

        return {"count": len(tail), "avg_coherence": avg_coh, "last_neutral_prompts": last_neutrals, "top_tokens": top_tokens}

    #
    # Deterministic AFD renderer (uses memory_summary to reference past)
    #
    def _afd_renderer(self, neutral_prompt, afd_directives, metrics, coherence, memory_summary=None):
        rng = np.random.default_rng()
        temp = max(0.0, float(self.temperature))
        noise_scale = max(0.0, temp * 0.04)

        metrics_noisy = {}
        for k, v in metrics.items():
            try:
                metrics_noisy[k] = float(v) + float(rng.normal(0.0, noise_scale))
            except Exception:
                metrics_noisy[k] = float(v)

        c = float(coherence) + float(rng.normal(0.0, noise_scale))
        h = metrics_noisy.get("harmony", 0.0)
        ig = metrics_noisy.get("info_gradient", 0.0)
        o = metrics_noisy.get("oscillation", 0.0)
        phi = metrics_noisy.get("potential", 0.0)

        templates = [
            f"In short: {neutral_prompt}. The AFD evaluation supports a direct, concise answer.",
            f"I would summarize: {neutral_prompt}. The AFD metrics support a straightforward explanation.",
            f"Summarizing: {neutral_prompt}. The numeric AFD evaluation suggests the following concise take.",
        ]

        if temp <= 0.0:
            template_idx = 0
        else:
            base = np.array([0.6, 0.25, 0.15])
            mix = min(1.0, temp / 1.0)
            weights = (1.0 - mix) * base + mix * np.array([1.0 / 3.0] * 3)
            weights = weights.clip(min=1e-6)
            weights = weights / weights.sum()
            template_idx = int(rng.choice(3, p=weights))

        answer_line = templates[template_idx]

        rationale_parts = []
        if c >= 0.8:
            rationale_parts.append("High coherence → present a concise, confident explanation.")
        elif c <= 0.35:
            rationale_parts.append("Low coherence → emphasize uncertainty and avoid strong claims.")
        else:
            rationale_parts.append("Moderate coherence → balanced and cautious explanation.")

        if h > 0.6:
            rationale_parts.append("High harmony → highlight consistent points.")
        elif h < 0.2:
            rationale_parts.append("Low harmony → clarify ambiguities.")

        if ig > 0.5:
            rationale_parts.append("High information gradient → include supporting details.")
        else:
            rationale_parts.append("Low information gradient → keep it concise.")

        if o > 0.2:
            rationale_parts.append("High oscillation → avoid speculation.")

        if memory_summary and memory_summary.get("avg_coherence") is not None:
            avg = memory_summary.get("avg_coherence")
            rationale_parts.append(f"Past interactions (last {memory_summary.get('count')}) average coherence={avg:.3f}, which the agent uses to calibrate tone.")

        if temp > 0.6:
            extra_styles = [
                "I will phrase this succinctly.",
                "I will focus on clarity and brevity.",
                "I will aim to highlight the most relevant points first.",
            ]
            extra_choice = int(rng.integers(0, len(extra_styles)))
            rationale_parts.append(extra_styles[extra_choice])

        lines = []
        lines.append(f"Neutral input: {neutral_prompt}")
        lines.append("")
        lines.append("AFD Summary (numeric):")
        lines.append(f"- coherence: {c:.6f}")
        lines.append(f"- harmony: {h:.6f}")
        lines.append(f"- info_gradient: {ig:.6f}")
        lines.append(f"- oscillation: {o:.6f}")
        lines.append(f"- potential: {phi:.6f}")
        lines.append("")
        lines.append("Deterministic explanation derived from AFD metrics:")
        for p in rationale_parts:
            lines.append(f"- {p}")
        lines.append("")
        lines.append("Answer:")
        lines.append(answer_line)
        return "\n".join(lines)

    def generate_human_style_reflection(self, neutral_prompt, metrics, coherence, memory_summary=None):
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0))
        ig = float(metrics.get("info_gradient", 0.0))
        o = float(metrics.get("oscillation", 0.0))
        phi = float(metrics.get("potential", 0.0))

        identity = (
            "I am an algorithmic assistant. I don't have consciousness or feelings in the human sense, "
            "but I can reflect on my internal assessment and describe how it shapes the answer I give."
        )

        if c >= 0.80:
            confidence = "Right now I feel quite confident about the information I will present."
        elif c <= 0.35:
            confidence = "I notice uncertainty in my assessment, so I will be cautious and avoid strong claims."
        else:
            confidence = "I have a moderate level of confidence; I'll aim for a balanced, measured response."

        harmony_note = (
            "My internal signals show strong agreement, so there is a clear, coherent picture to report."
            if h > 0.6
            else "There are mixed signals internally, so I will highlight ambiguities where relevant."
        )

        detail_note = "I'll include supporting detail to clarify points." if ig > 0.5 else "I'll keep the explanation concise and focused."
        stability_note = (
            "The internal state is reasonably stable, so I won't speculate beyond the data."
            if o <= 0.2
            else "The internal state shows some instability; I'll avoid speculative statements."
        )
        potential_note = "There is notable potential for structured guidance." if phi > 0.5 else ""

        neutral_echo = neutral_prompt if len(neutral_prompt) <= 200 else neutral_prompt[:197] + "…"

        paragraphs = []
        paragraphs.append(identity)
        paragraphs.append("")
        paragraphs.append(f"- Neutral input (summary): {neutral_echo}")
        paragraphs.append(f"- Assessment: coherence={c:.3f}. {confidence}")
        paragraphs.append(f"- Observations: {harmony_note} {detail_note} {stability_note} {potential_note}".strip())

        if memory_summary and memory_summary.get("count", 0) > 0:
            mem_count = memory_summary.get("count")
            avg_coh = memory_summary.get("avg_coherence")
            top = ", ".join(memory_summary.get("top_tokens", [])[:5])
            paragraphs.append("")
            mem_line = f"I also recall {mem_count} recent interactions; their average coherence was {avg_coh:.3f}."
            if top:
                mem_line += f" Common topics from recent neutral prompts include: {top}."
            paragraphs.append(mem_line)

        paragraphs.append("")
        paragraphs.append(
            "As a result of this assessment, here's how I will approach the answer: "
            "I will prioritize clarity and factual content drawn from the neutralized prompt. "
            "If uncertainty is present, I will explicitly signal caution and avoid definitive claims."
        )
        paragraphs.append("")
        paragraphs.append("This reflection is a concise, human-style summary of my internal AFD evaluation and how it influences the forthcoming answer.")
        return "\n".join(paragraphs)

    #
    # Deterministic AFD reflection (structured, authoritative)
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
            lines.append("- Rationale: Low coherence -> prioritized neutrality and highlighted uncertainty in the response.")
        elif coherence > 0.8:
            lines.append("- Rationale: High coherence -> produced a confident, concise explanation emphasizing agreement.")
        else:
            lines.append("- Rationale: Moderate coherence -> balanced explanation with cautious statements.")
        lines.append(f"- renderer_used: {renderer_used}")
        return "\n".join(lines)

    def _build_renderer_prompt(self, neutral_input, afd_directives):
        return f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."

    def respond(self, prompt, renderer_mode="afd", include_memory=True):
        # neutralize
        neutral_prompt = ""
        if self.use_openai:
            neutral_prompt = self._neutralize_with_openai(prompt)
        else:
            neutral_prompt = self._neutralize_with_hf(prompt)
        if not neutral_prompt:
            neutral_prompt = prompt.strip()

        # optional memory summary
        memory_summary = None
        if include_memory:
            try:
                memory_summary = self._summarize_memory(self.memory_window)
            except Exception as e:
                self.reflection_log.append(f"Memory summary error: {e}")
                memory_summary = None

        # sentiment/state proxy
        try:
            if self.use_openai:
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
            self.reflection_log.append(f"Sentiment error in respond: {e}")
            sentiment_score = 0.5
            sentiment_label = "NEUTRAL"

        state = np.array([sentiment_score] * 5)
        action = np.array([1 if str(sentiment_label).upper().startswith("POS") else -1] * 5)

        # compute AFD internals
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)
        h = self.compute_harmony(state, interp_s)
        i = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        coherence = float(self.alpha * h + self.beta * i - self.gamma * o + self.delta * phi)
        metrics = {"harmony": float(h), "info_gradient": float(i), "oscillation": float(o), "potential": float(phi)}

        # lightweight historical feedback: nudge coefficients by historical avg coherence (if available)
        try:
            if memory_summary and memory_summary.get("avg_coherence") is not None:
                avg_hist = memory_summary.get("avg_coherence")
                # small adjustment: if past average low, increase alpha to favor harmony
                if avg_hist < 0.45:
                    self.alpha += 0.02
                    self.reflection_log.append(f"Adjusted alpha slightly based on memory avg_coherence={avg_hist:.3f}")
                elif avg_hist > 0.8:
                    self.gamma += 0.02
                    self.reflection_log.append(f"Adjusted gamma slightly based on memory avg_coherence={avg_hist:.3f}")
        except Exception as e:
            self.reflection_log.append(f"Memory-based coefficient adjustment error: {e}")

        # adjust coefficients according to current coherence
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

        # choose renderer
        if renderer_mode == "afd":
            final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence, memory_summary=memory_summary)
            renderer_used = "AFD (deterministic)"
        else:
            if self.use_openai:
                final_text = self._render_with_openai(neutral_prompt, afd_directives)
                renderer_used = "OpenAI"
            else:
                final_text = self._render_with_hf(neutral_prompt, afd_directives)
                renderer_used = "HF"

        # persist memory & explainability
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
