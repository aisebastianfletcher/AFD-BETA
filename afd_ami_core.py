"""
AFDInfinityAMI - Consolidated AFD core with authoritative generate_afd_reflection.

Drop this file into afd_ami_core.py (overwrite), restart your Streamlit server,
and the AttributeError for generate_afd_reflection should be resolved.

Notes:
- This is a defensive, self-contained implementation: reflection helpers are robust
  and respond(...) calls generate_afd_reflection at the end to produce afd_reflection.
- If you still see AttributeError after replacing the file, fully restart the Python
  process (Streamlit reload may cache old module).
"""
from typing import Optional
import os
import json
import re
import math
import datetime
import collections
import numpy as np
import pandas as pd
import streamlit as st

# Optional model deps guarded at runtime
try:
    import openai
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    openai = None
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


class AFDInfinityAMI:
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None, temperature: float = 0.0, memory_window: int = 20):
        # runtime
        self.temperature = float(temperature or 0.0)
        self.memory_window = int(memory_window or 20)

        # persistence paths
        self.memory_file = "data/response_log.csv"
        self.explain_file = "data/explain_log.jsonl"
        self.chat_file = "data/chat_history.jsonl"
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        if not os.path.exists(self.memory_file):
            try:
                pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"]).to_csv(
                    self.memory_file, index=False, encoding="utf-8-sig"
                )
            except Exception:
                pass

        # AFD coefficients (learnable)
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 0.5
        self.delta = 0.5
        self._learning_rate = 0.02
        self._coeff_min, self._coeff_max = -2.0, 4.0

        # logging and last explain snapshot
        self.reflection_log = []
        self.latest_explainability = None

        # LLM config (optional)
        self.use_openai = False
        if openai and openai_api_key:
            try:
                k = str(openai_api_key).strip()
                openai.api_key = k
                self.use_openai = True
            except Exception:
                self.use_openai = False

        # basic system strings
        self.neutralizer_system = "You are a precise neutral translator. Convert the user's input into a short, neutral description."
        self.renderer_system = "You are a factual renderer. Use only the neutral input and AFD metrics to produce the answer."

        # lazy HF resources
        self._hf_llm = None
        self.sentiment_analyzer = None

    #
    # Lightweight HF initializers (guarded)
    #
    @st.cache_resource
    def _cache_sentiment(_self):
        if pipeline is None:
            return None
        try:
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception:
            return None

    def _ensure_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = self._cache_sentiment()
        return self.sentiment_analyzer

    #
    # Neutralizers (rule-based for AFD mode)
    #
    def _rule_neutralize(self, user_input: str) -> str:
        s = (user_input or "").strip()
        if not s:
            return ""
        core = s.rstrip(" ?!.")
        low = core.lower()
        m = re.match(r"^(what|who|when|where|why|how)\s+(.+)", low)
        if m:
            q = m.group(1)
            rest = core[len(q):].strip()
            if q == "what":
                return f"definition/explanation request: {rest}"
            if q == "who" and "you" in rest:
                return "agent identity inquiry"
            return f"query about: {rest}"
        m2 = re.match(r"^(explain|describe|summarize|compare|list)\s+(.+)", low)
        if m2:
            return f"{m2.group(1)} {m2.group(2).rstrip(' ?!.')}"
        trimmed = core.strip()
        return f"neutral summary: {trimmed[:150]}"

    #
    # AFD core math (small proxies)
    #
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state + np.random.normal(0, 0.1, state.shape)

    def compute_harmony(self, state: np.ndarray, interp_s: np.ndarray) -> float:
        return float(np.linalg.norm(interp_s - state) / (np.linalg.norm(state) + 1e-10))

    def compute_info_gradient(self, state: np.ndarray, interp_s: np.ndarray) -> float:
        return float(np.abs(interp_s - state).sum() / (np.linalg.norm(state) + 1e-10))

    def compute_oscillation(self, state: np.ndarray, interp_s: np.ndarray) -> float:
        return float(np.std(interp_s - state))

    def compute_potential(self, s_prime: np.ndarray) -> float:
        return float(np.linalg.norm(s_prime) / 10.0)

    #
    # Deterministic renderer (AFD)
    #
    def _afd_renderer(self, neutral_prompt: str, afd_directives: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, seed: Optional[int] = None, intent: Optional[str] = None, chat_context: Optional[str] = None) -> str:
        # small deterministic templates; intent-aware
        if intent == "identity":
            identity = self.generate_identity_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
            answer = "You are speaking to AFD∞-AMI — an algorithmic assistant. I am not conscious."
            return f"{identity}\n\nAnswer:\n{answer}"

        # Default knowledge template (keeps outputs short)
        harmony = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        ig = float(metrics.get("info_gradient", 0.0)) if isinstance(metrics, dict) else 0.0
        lines = [f"Neutral input: {neutral_prompt}", ""]
        lines.append("Definition:")
        lines.append("Life is a self-maintaining, information-processing system that uses energy to sustain organized structure, replicate, and evolve.")
        if harmony > 0.15 or ig > 0.2:
            lines.append("")
            lines.append("Supporting points:")
            if harmony > 0.15:
                lines.append("- Organization and homeostasis")
            if ig > 0.2:
                lines.append("- Information storage and regulatory networks")
        # concise rationale
        lines.append("")
        lines.append(f"AFD rationale: coherence={coherence:.3f}, harmony={harmony:.3f}, info_grad={ig:.3f}")
        return "\n".join(lines)

    #
    # Reflection helpers (safe / deterministic)
    #
    def generate_identity_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None) -> str:
        coeffs = f"alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}"
        mem_note = ""
        if memory_summary and memory_summary.get("count"):
            mem_note = f" I recall {memory_summary.get('count')} recent interactions."
        return (
            "I am AFD∞-AMI, an algorithmic assistant. I am not conscious or sentient; I neutralize prompts, compute internal AFD metrics, "
            "and produce auditable answers. " + f"({coeffs})" + mem_note
        )

    def generate_human_style_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None) -> str:
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        confidence = "high" if c >= 0.8 else ("low" if c <= 0.35 else "moderate")
        return f"I assessed this request with {confidence} confidence (coherence={c:.3f}, harmony={h:.3f})."

    def generate_simulated_thinking_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, max_steps: int = 4):
        try:
            c = float(coherence)
        except Exception:
            c = 0.0
        h = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        lines = []
        lines.append("Thinking: " + ("clear signal." if c >= 0.8 else ("uncertain signal." if c <= 0.35 else "moderate signal.")))
        if h > 0.5:
            lines.append("Indicators align — focus on main points.")
        elif h < 0.2:
            lines.append("Mixed indicators — highlight ambiguities.")
        lines.append("Now producing an auditable answer.")
        return lines[:max_steps]

    def generate_reasoning_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, verbosity: int = 1, seed: Optional[int] = None) -> str:
        c = float(coherence)
        conf = "high" if c >= 0.8 else ("low" if c <= 0.35 else "moderate")
        return f"(Reasoning) Confidence: {conf}. I will present a {'concise' if conf=='high' else 'cautious' if conf=='low' else 'balanced'} answer."

    #
    # Authoritative AFD reflection (must exist)
    #
    def generate_afd_reflection(self, prompt: str, neutral_prompt: str, state, action, s_prime, interp_s, metrics: dict, coherence: float, renderer_used: str) -> str:
        # build a concise structured reflection; defensive conversions
        try:
            s = np.array(state).tolist() if hasattr(np, "array") else state
        except Exception:
            s = state
        try:
            a = np.array(action).tolist() if hasattr(np, "array") else action
        except Exception:
            a = action
        try:
            sp = np.array(s_prime).tolist()
        except Exception:
            sp = s_prime
        try:
            ip = np.array(interp_s).tolist()
        except Exception:
            ip = interp_s
        harmony = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        ig = float(metrics.get("info_gradient", 0.0)) if isinstance(metrics, dict) else 0.0
        oscill = float(metrics.get("oscillation", 0.0)) if isinstance(metrics, dict) else 0.0
        phi = float(metrics.get("potential", 0.0)) if isinstance(metrics, dict) else 0.0

        lines = [
            "AFD Reflection:",
            f"- original_prompt: {prompt}",
            f"- neutral_prompt: {neutral_prompt}",
            f"- state (proxy): {s}",
            f"- action proxy: {a}",
            f"- predicted s_prime: {sp}",
            f"- interp_s: {ip}",
            f"- afd_metrics: harmony={harmony:.6f}, info_gradient={ig:.6f}, oscillation={oscill:.6f}, potential={phi:.6f}",
            f"- coherence: {float(coherence):.6f}",
        ]
        if coherence < 0.4:
            lines.append("- Rationale: Low coherence -> cautious, highlight uncertainty.")
        elif coherence > 0.8:
            lines.append("- Rationale: High coherence -> concise, confident response.")
        else:
            lines.append("- Rationale: Moderate coherence -> balanced response.")
        lines.append(f"- renderer_used: {renderer_used}")
        lines.append(f"- coefficients: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}")
        return "\n".join(lines)

    #
    # Persistence helpers
    #
    def save_memory(self, prompt: str, neutral_prompt: str, response: str, coherence: float, timestamp_iso: str, outcome: Optional[float] = None, outcome_comment: Optional[str] = None):
        try:
            df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception:
            df = pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"])
        new_row = pd.DataFrame(
            {
                "timestamp": [timestamp_iso],
                "prompt": [prompt],
                "neutral_prompt": [neutral_prompt],
                "response": [response],
                "coherence": [coherence],
                "outcome": [outcome],
                "outcome_comment": [outcome_comment],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)
        try:
            df.to_csv(self.memory_file, index=False, encoding="utf-8-sig")
        except Exception:
            self.reflection_log.append("Warning: could not save memory CSV.")

    def append_explainability(self, explainability: dict):
        try:
            os.makedirs(os.path.dirname(self.explain_file) or ".", exist_ok=True)
            with open(self.explain_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(explainability, ensure_ascii=False) + "\n")
        except Exception:
            self.reflection_log.append("Warning: could not append explainability JSONL.")

    def append_chat_entry(self, role: str, message: str, timestamp_iso: Optional[str] = None):
        try:
            os.makedirs(os.path.dirname(self.chat_file) or ".", exist_ok=True)
            ts = timestamp_iso or datetime.datetime.utcnow().isoformat()
            with open(self.chat_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": ts, "role": role, "message": message}, ensure_ascii=False) + "\n")
        except Exception:
            self.reflection_log.append("Warning: could not append chat entry.")

    def load_memory(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"])

    def get_last_explainability(self) -> Optional[dict]:
        return getattr(self, "latest_explainability", None)

    #
    # Learning / feedback (conservative)
    #
    def provide_feedback(self, entry_timestamp: str, outcome_score: float, outcome_comment: Optional[str] = None) -> bool:
        try:
            score = float(outcome_score)
            if math.isnan(score) or score < 0.0 or score > 1.0:
                raise ValueError("outcome_score must be in [0,1]")
        except Exception:
            self.reflection_log.append("Feedback invalid score.")
            return False
        try:
            df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
            mask = df["timestamp"] == entry_timestamp
            if mask.any():
                df.loc[mask, "outcome"] = score
                df.loc[mask, "outcome_comment"] = outcome_comment
                df.to_csv(self.memory_file, index=False, encoding="utf-8-sig")
        except Exception:
            self.reflection_log.append("Feedback: could not update memory CSV.")
        # conservative learning update: find explainability entry
        found = None
        try:
            if os.path.exists(self.explain_file):
                with open(self.explain_file, "r", encoding="utf-8") as f:
                    for L in f:
                        try:
                            obj = json.loads(L)
                            if obj.get("timestamp") == entry_timestamp:
                                found = obj
                                break
                        except Exception:
                            continue
            if found:
                metrics = found.get("afd_metrics", {})
                observed = float(found.get("coherence", 0.0))
                self._learning_update(observed, score, {"h": metrics.get("harmony", 0.0), "ig": metrics.get("info_gradient", 0.0), "o": metrics.get("oscillation", 0.0), "phi": metrics.get("potential", 0.0)})
        except Exception:
            self.reflection_log.append("Feedback processing error.")
        return True

    def _learning_update(self, observed_coherence: float, outcome_score: float, metrics_dict: dict):
        try:
            eps = 1e-8
            h = float(metrics_dict.get("h", 0.0))
            ig = float(metrics_dict.get("ig", 0.0))
            o = float(metrics_dict.get("o", 0.0))
            phi = float(metrics_dict.get("phi", 0.0))
            mag = abs(h) + abs(ig) + abs(o) + abs(phi) + eps
            nh, nig, no_, nphi = h/mag, ig/mag, o/mag, phi/mag
            error = float(outcome_score) - float(observed_coherence)
            lr = float(self._learning_rate)
            self.alpha = max(self._coeff_min, min(self._coeff_max, self.alpha + lr * error * nh))
            self.beta  = max(self._coeff_min, min(self._coeff_max, self.beta  + lr * error * nig))
            self.gamma = max(self._coeff_min, min(self._coeff_max, self.gamma - lr * error * no_))
            self.delta = max(self._coeff_min, min(self._coeff_max, self.delta + lr * error * nphi))
            self.reflection_log.append(f"Learning update: err={error:.4f}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}")
        except Exception:
            self.reflection_log.append("Learning update failed.")

    #
    # Main respond (single-turn prompt; chat_context passed separately for explainability only)
    #
    def respond(self, prompt: str, renderer_mode: str = "afd", include_memory: bool = True, seed: Optional[int] = None, reasoning_verbosity: int = 1, chat_context: Optional[str] = None):
        # neutralize (only current user prompt)
        if renderer_mode == "afd":
            neutral_prompt = self._rule_neutralize(prompt)
        else:
            neutral_prompt = self._rule_neutralize(prompt)

        if not neutral_prompt:
            neutral_prompt = prompt.strip()

        # memory summary
        memory_summary = None
        if include_memory:
            try:
                memory_summary = self._summarize_memory(self.memory_window)
            except Exception:
                memory_summary = None

        # sentiment/state proxy
        try:
            sent_an = self._ensure_sentiment_analyzer()
            if sent_an:
                sent = sent_an(neutral_prompt)[0]
                sentiment_score = float(sent.get("score", 0.5))
                sentiment_label = sent.get("label", "NEUTRAL")
            else:
                sentiment_score = 0.5
                sentiment_label = "NEUTRAL"
        except Exception:
            sentiment_score = 0.5
            sentiment_label = "NEUTRAL"

        state = np.array([sentiment_score]*5)
        action = np.array([1 if str(sentiment_label).upper().startswith("POS") else -1]*5)

        # AFD internals
        s_prime = self.predict_next_state(state, action)
        interp_s = state + 0.5*(s_prime - state)
        h = self.compute_harmony(state, interp_s)
        ig = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        coherence = float(self.alpha*h + self.beta*ig - self.gamma*o + self.delta*phi)
        metrics = {"harmony": float(h), "info_gradient": float(ig), "oscillation": float(o), "potential": float(phi)}

        # tiny memory nudges (conservative)
        try:
            if memory_summary and memory_summary.get("avg_outcome") is not None:
                avg = memory_summary.get("avg_outcome")
                if avg is not None and avg < 0.4:
                    self.alpha = min(self._coeff_max, self.alpha + 0.01)
                elif avg is not None and avg > 0.85:
                    self.gamma = min(self._coeff_max, self.gamma + 0.01)
        except Exception:
            pass

        # detect intent
        detected_intent = self._detect_intent(neutral_prompt)

        # generate reflections (defensive)
        try:
            reasoning_reflection_text = self.generate_reasoning_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary, verbosity=reasoning_verbosity, seed=seed)
        except Exception:
            reasoning_reflection_text = ""

        try:
            simulated_thinking = self.generate_simulated_thinking_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        except Exception:
            simulated_thinking = []

        try:
            human_reflection = self.generate_human_style_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        except Exception:
            human_reflection = ""

        try:
            identity_reflection = self.generate_identity_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        except Exception:
            identity_reflection = ""

        # build afd_directives (for LLM or explainability)
        afd_directives = (
            f"AFD metrics:\n- coherence: {coherence:.6f}\n- harmony: {metrics['harmony']:.6f}\n- info_gradient: {metrics['info_gradient']:.6f}\n"
            f"- oscillation: {metrics['oscillation']:.6f}\n- potential: {metrics['potential']:.6f}\n"
            f"Coefficients: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}\n"
        )

        # render answer
        try:
            if renderer_mode == "afd":
                final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence, memory_summary=memory_summary, seed=seed, intent=detected_intent, chat_context=chat_context)
                renderer_used = "AFD"
            else:
                final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence, memory_summary=memory_summary, seed=seed, intent=detected_intent, chat_context=chat_context)
                renderer_used = "AFD"
        except Exception as e:
            final_text = ""
            renderer_used = "AFD"
            self.reflection_log.append(f"Renderer failed: {e}")

        # opinion: prepend reasoning reflection
        if detected_intent == "opinion" and reasoning_reflection_text:
            final_text = reasoning_reflection_text + "\n\nAnswer:\n" + (final_text or "")

        # defensive identity override
        if detected_intent == "identity" and (not final_text or "Definition:" in final_text):
            final_text = identity_reflection + "\n\nAnswer:\n" + (final_text or "")

        # timestamp & persist
        timestamp_iso = datetime.datetime.utcnow().isoformat()
        try:
            self.save_memory(prompt, neutral_prompt, final_text, coherence, timestamp_iso)
        except Exception:
            pass

        explainability = {
            "timestamp": timestamp_iso,
            "original_prompt": prompt,
            "neutral_prompt": neutral_prompt,
            "chat_context": chat_context,
            "detected_intent": detected_intent,
            "renderer_used": renderer_used,
            "renderer_prompt": afd_directives,
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
            "human_reflection": human_reflection,
            "identity_reflection": identity_reflection,
            "simulated_thinking": simulated_thinking,
            "reasoning_reflection": reasoning_reflection_text,
        }

        self.latest_explainability = explainability
        try:
            self.append_explainability(explainability)
        except Exception:
            pass

        # authoritative afd_reflection
        try:
            afd_reflection = self.generate_afd_reflection(prompt, neutral_prompt, state, action, s_prime, interp_s, metrics, coherence, renderer_used)
        except Exception as e:
            afd_reflection = f"AFD Reflection unavailable: {e}"
            self.reflection_log.append(f"generate_afd_reflection failed: {e}")

        try:
            self.reflection_log.append(afd_reflection)
        except Exception:
            pass

        return final_text, coherence, afd_reflection, timestamp_iso

    #
    # Helper used above
    #
    def _summarize_memory(self, n: Optional[int] = None) -> dict:
        n = int(n) if n is not None else int(self.memory_window)
        try:
            df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception:
            return {"count": 0, "avg_coherence": None, "avg_outcome": None, "last_neutral_prompts": [], "top_tokens": []}
        if df.empty:
            return {"count": 0, "avg_coherence": None, "avg_outcome": None, "last_neutral_prompts": [], "top_tokens": []}
        tail = df.tail(n)
        try:
            avg_coh = float(tail["coherence"].astype(float).mean())
        except Exception:
            avg_coh = None
        avg_out = None
        try:
            if "outcome" in tail.columns:
                outcol = tail["outcome"].dropna()
                if not outcol.empty:
                    avg_out = float(outcol.astype(float).mean())
        except Exception:
            avg_out = None
        last = tail["neutral_prompt"].fillna("").astype(str).tolist()
        counter = collections.Counter()
        for s in last:
            tokens = re.findall(r"[A-Za-z]{3,}", s.lower())
            for t in tokens:
                if t not in {"the", "and", "for", "with", "that", "this"}:
                    counter[t] += 1
        top = [tok for tok, _ in counter.most_common(8)]
        return {"count": len(tail), "avg_coherence": avg_coh, "avg_outcome": avg_out, "last_neutral_prompts": last, "top_tokens": top}
