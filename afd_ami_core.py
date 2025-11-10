"""
AFDInfinityAMI - AFD-driven assistant core.

This file defines the AFDInfinityAMI class. Ensure this file replaces the existing afd_ami_core.py
so no top-level stray function definitions remain outside the class.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import openai
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Try to import explicit OpenAI auth exception (fall back if unavailable)
try:
    from openai.error import AuthenticationError as OpenAIAuthError
except Exception:
    OpenAIAuthError = Exception


class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None):
        # Read key from explicit param, Streamlit secrets, or environment
        raw_key = openai_api_key or None
        try:
            if raw_key is None:
                raw_key = st.secrets.get("OPENAI_API_KEY") or raw_key
        except Exception:
            raw_key = raw_key

        if raw_key is None:
            raw_key = os.getenv("OPENAI_API_KEY")

        # Normalize the key (strip whitespace and accidental quotes)
        api_key = None
        if raw_key:
            api_key = str(raw_key).strip()
            if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
                api_key = api_key[1:-1].strip()

        # Set initial preference; will verify auth below
        self.use_openai = bool(api_key) or use_openai
        if api_key:
            openai.api_key = api_key

        # Reflection log collects init/runtime notes (safe to display)
        self.reflection_log = []

        # If an API key is present, do a lightweight auth test and disable OpenAI on auth failure
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

        # Memory and coefficients
        self.memory_file = "data/response_log.csv"
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5

        # System prompts for neutralizer and renderer
        self.neutralizer_system = (
            "You are a precise neutral translator. Convert the user's input into a short, "
            "neutral, factual description suitable for algorithmic processing. "
            "Do NOT add opinions, recommendations, or extra context. Keep it concise."
        )
        self.renderer_system = (
            "You are a factual renderer. Produce a clear, neutral, and concise explanation "
            "based ONLY on the provided neutral input and the AFD directives below. "
            "Do NOT add any external opinions, extra facts, or speculation. Use full sentences."
        )

        # Lazy-loaded HF resources (do not initialize heavy models here)
        self._hf_llm = None
        self.sentiment_analyzer = None

        # Latest explainability snapshot (in-memory)
        self.latest_explainability = None

        # Ensure memory file exists (safe, non-blocking)
        if not os.path.exists(self.memory_file):
            try:
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                pd.DataFrame(columns=["prompt", "neutral_prompt", "response", "coherence"]).to_csv(
                    self.memory_file, index=False, encoding="utf-8-sig"
                )
            except Exception as e:
                self.reflection_log.append(f"Error creating memory file: {e}")

    def _test_openai_key(self):
        """
        Lightweight OpenAI key test: perform a minimal ChatCompletion call.
        Raises OpenAIAuthError on invalid key.
        """
        # Small, cheap call to validate auth (minimal tokens)
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping for auth test. Reply with 'ok'."}],
            max_tokens=1,
            temperature=0.0,
        )
        return True

    #
    # Lazy initializers to avoid heavy downloads at import time
    #
    def _ensure_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = self._cache_sentiment_analyzer()
            except Exception as e:
                self.reflection_log.append(f"Could not init sentiment analyzer: {e}")
                self.sentiment_analyzer = None
        return self.sentiment_analyzer

    def _ensure_hf_llm(self):
        if self._hf_llm is None:
            try:
                self._hf_llm = self._cache_llm()
            except Exception as e:
                self.reflection_log.append(f"Could not init HF llm: {e}")
                self._hf_llm = None
        return self._hf_llm

    #
    # Caching utilities for HF pipelines (local)
    #
    @st.cache_resource
    def _cache_llm(_self):
        # Use a smaller causal model by default to reduce footprint; change if you prefer
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = model.config.eos_token_id
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    @st.cache_resource
    def _cache_sentiment_analyzer(_self):
        # Standard SST-2 sentiment analyzer; used only to produce a numeric state proxy.
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    #
    # LLM wrappers: neutralizer and renderer, with auth-fallback
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
            # Auth error: disable OpenAI and fall back to HF
            self.reflection_log.append(f"OpenAI neutralize auth error: {e}. Falling back to HF.")
            self.use_openai = False
            self._ensure_hf_llm()
            return self._neutralize_with_hf(user_input)
        except Exception as e:
            self.reflection_log.append(f"OpenAI neutralize error: {e}")
            return ""

    def _render_with_openai(self, neutral_input, afd_directives, max_tokens=250):
        content = f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."
        try:
            messages = [
                {"role": "system", "content": self.renderer_system},
                {"role": "user", "content": content},
            ]
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.2, max_tokens=max_tokens)
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

    def _render_with_hf(self, neutral_input, afd_directives):
        llm = self._ensure_hf_llm()
        if llm is None:
            return "Local model not available to render."
        prompt = f"{self.renderer_system}\n\nNeutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nAnswer:"
        try:
            out = llm(
                prompt,
                max_new_tokens=220,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=0.6,
                return_full_text=False,
                num_return_sequences=1,
            )
            text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            if text.startswith(prompt):
                text = text[len(prompt) :]
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF render error: {e}")
            return "Unable to render response locally."

    #
    # AFD framework (core math)
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
    # Memory I/O
    #
    def save_memory(self, prompt, neutral_prompt, response, coherence):
        try:
            df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
            new_row = pd.DataFrame(
                {"prompt": [prompt], "neutral_prompt": [neutral_prompt], "response": [response], "coherence": [coherence]}
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.memory_file, index=False, encoding="utf-8-sig")
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not save to CSV ({e}).")

    def load_memory(self):
        try:
            return pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception as e:
            self.reflection_log.append(f"Error loading memory file: {e}")
            return pd.DataFrame(columns=["prompt", "neutral_prompt", "response", "coherence"])

    def get_latest_reflection(self):
        return self.reflection_log[-1] if self.reflection_log else "No reflections yet."

    #
    # Explainability helpers, deterministic AFD renderer and reflections
    #
    def _build_renderer_prompt(self, neutral_input, afd_directives):
        return f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."

    def get_last_explainability(self):
        """Return the most recent explainability snapshot (dict) if available."""
        return getattr(self, "latest_explainability", None)

    def _afd_renderer(self, neutral_prompt, afd_directives, metrics, coherence):
        """
        Deterministic AFD renderer: produce an answer synthesized from the neutral prompt
        and numeric AFD metrics. No LLM involved. This ensures the reply is produced
        by the AFD agent logic, not by a freeform LLM.
        """
        # Basic structured summary
        lines = []
        lines.append(f"Neutral input: {neutral_prompt}")
        lines.append("")
        lines.append("AFD Summary (numeric):")
        lines.append(f"- coherence: {coherence:.6f}")
        lines.append(f"- harmony: {metrics.get('harmony'):.6f}")
        lines.append(f"- info_gradient: {metrics.get('info_gradient'):.6f}")
        lines.append(f"- oscillation: {metrics.get('oscillation'):.6f}")
        lines.append(f"- potential: {metrics.get('potential'):.6f}")
        lines.append("")

        # Deterministic, rule-based explanation mapping numeric metrics -> plain-language
        rationale_parts = []
        c = coherence
        h = metrics.get("harmony", 0.0)
        ig = metrics.get("info_gradient", 0.0)
        o = metrics.get("oscillation", 0.0)
        phi = metrics.get("potential", 0.0)

        # Confidence/stance mapping from coherence
        if c >= 0.8:
            rationale_parts.append("The AFD state indicates high coherence, so the assistant presents a concise, confident explanation.")
        elif c <= 0.35:
            rationale_parts.append("The AFD state indicates low coherence, so the assistant emphasizes uncertainty and avoids strong claims.")
        else:
            rationale_parts.append("The AFD state indicates moderate coherence, so the assistant gives a balanced, cautious explanation.")

        # Harmony mapping
        if h < 0.2:
            rationale_parts.append("Low harmony suggests divergent information; the response focuses on clarifying ambiguities.")
        elif h > 0.6:
            rationale_parts.append("High harmony suggests information is mutually consistent; the response highlights agreement and key points.")

        # Info gradient mapping
        if ig > 0.5:
            rationale_parts.append("High information gradient indicates a need to surface more supporting details.")
        else:
            rationale_parts.append("Low information gradient suggests concision is appropriate.")

        # Oscillation mapping
        if o > 0.2:
            rationale_parts.append("High oscillation suggests instability; the assistant avoids speculative statements.")
        # Potential mapping (optional)
        if phi > 0.5:
            rationale_parts.append("Potential is relatively high, so the assistant attempts to present structured guidance.")

        # Build final deterministic explanation text
        lines.append("Deterministic explanation derived from AFD metrics:")
        for p in rationale_parts:
            lines.append(f"- {p}")

        # Compose an explicit answer sentence derived from the neutral prompt and AFD cues
        answer_line = ""
        if c >= 0.8:
            answer_line = f"In short: {neutral_prompt}. The AFD evaluation supports a direct, concise answer."
        elif c <= 0.35:
            answer_line = f"In short: {neutral_prompt}. The AFD evaluation indicates uncertainty; I therefore present a cautious summary rather than definitive claims."
        else:
            answer_line = f"In short: {neutral_prompt}. The AFD evaluation suggests a balanced response with measured confidence."

        lines.append("")
        lines.append("Answer:")
        lines.append(answer_line)

        return "\n".join(lines)

    def generate_afd_reflection(self, prompt, neutral_prompt, state, action, s_prime, interp_s, metrics, coherence, renderer_used):
        """
        Deterministic AFD reflection describing which numeric factors produced the answer.
        This is the AFD agent's reflection (not chain-of-thought) and will be returned as the reflection.
        """
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
        # Deterministic rationale text based on coherence thresholds
        if coherence < 0.4:
            lines.append("- Rationale: Low coherence -> prioritized neutrality and highlighted uncertainty in the response.")
        elif coherence > 0.8:
            lines.append("- Rationale: High coherence -> produced a confident, concise explanation emphasizing agreement.")
        else:
            lines.append("- Rationale: Moderate coherence -> balanced explanation with cautious statements.")
        lines.append(f"- renderer_used: {renderer_used}")
        return "\n".join(lines)

    def respond(self, prompt, renderer_mode="afd"):
        """
        High-level respond that supports renderer_mode:
         - 'afd' (default): deterministic AFD renderer (no LLM) — final_text is produced by AFD logic.
         - 'llm': use the existing LLM renderers (_render_with_openai/_render_with_hf).

        The returned reflection is the deterministic AFD reflection (generate_afd_reflection).
        """
        # 1) Neutralize / translate the prompt
        neutral_prompt = ""
        if self.use_openai:
            neutral_prompt = self._neutralize_with_openai(prompt)
        else:
            neutral_prompt = self._neutralize_with_hf(prompt)

        if not neutral_prompt:
            neutral_prompt = prompt.strip()

        # 2) Compute sentiment/state from neutral prompt to drive AFD math.
        try:
            if self.use_openai:
                # Avoid initializing HF sentiment analyzer when using OpenAI.
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

        # Construct numeric state and action for AFD computations
        state = np.array([sentiment_score] * 5)
        action = np.array([1 if str(sentiment_label).upper().startswith("POS") else -1] * 5)

        # 3) Compute AFD internals (s_prime, interp_s, metrics)
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)

        h = self.compute_harmony(state, interp_s)
        i = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        coherence = float(self.alpha * h + self.beta * i - self.gamma * o + self.delta * phi)
        metrics = {"harmony": float(h), "info_gradient": float(i), "oscillation": float(o), "potential": float(phi)}

        # Adjust coefficients according to the AFD framework
        self.adjust_coefficients(coherence, metrics)

        # 4) Craft AFD directives and renderer prompt (exact strings stored)
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

        # 5) Choose renderer mode: deterministic AFD renderer (default) or LLM
        if renderer_mode == "afd":
            final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence)
            renderer_used = "AFD (deterministic)"
        else:
            if self.use_openai:
                final_text = self._render_with_openai(neutral_prompt, afd_directives)
                renderer_used = "OpenAI"
            else:
                final_text = self._render_with_hf(neutral_prompt, afd_directives)
                renderer_used = "HF"

        # 6) Save to memory and record explainability snapshot
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
        }
        self.latest_explainability = explainability

        # 7) Build deterministic AFD reflection and append to reflection_log
        afd_reflection = self.generate_afd_reflection(
            prompt=prompt,
            neutral_prompt=neutral_prompt,
            state=state,
            action=action,
            s_prime=s_prime,
            interp_s=interp_s,
            metrics=metrics,
            coherence=coherence,
            renderer_used=renderer_used,
        )
        try:
            self.reflection_log.append(afd_reflection)
        except Exception:
            self.reflection_log = [afd_reflection]

        return final_text, coherence, afd_reflection
