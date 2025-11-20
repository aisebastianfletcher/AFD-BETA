# afd_ami_core.py
"""
AFDInfinityAMI - Full AFD agent core (deterministic AFD renderer, safe reflections,
intent detection, chat persistence, explainability, and conservative learning).
Designed to be defensive so helper exceptions won't crash respond().
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

# Optional libs (guarded)
try:
    import streamlit as st  # used for caching in some environments
except Exception:
    st = None

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
        # runtime params
        self.temperature = float(temperature or 0.0)
        self.memory_window = int(memory_window or 20)

        # persistence paths
        self.memory_file = "data/response_log.csv"
        self.explain_file = "data/explain_log.jsonl"
        self.chat_file = "data/chat_history.jsonl"
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        # ensure base memory CSV exists
        if not os.path.exists(self.memory_file):
            pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"]).to_csv(
                self.memory_file, index=False, encoding="utf-8-sig"
            )

        # AFD coefficients
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 0.5
        self.delta = 0.5
        self._learning_rate = 0.02
        self._coeff_min, self._coeff_max = -2.0, 4.0

        # logs and last explain snapshot
        self.reflection_log = []
        self.latest_explainability = None

        # OpenAI setup (optional)
        self.use_openai = False
        if openai and openai_api_key:
            try:
                key = str(openai_api_key).strip()
                if key:
                    openai.api_key = key
                    self.use_openai = True
            except Exception as e:
                self.reflection_log.append(f"OpenAI key init error: {e}")

        # system prompts (for LLM mode only)
        self.neutralizer_system = "You are a precise neutral translator. Convert the user's input into a short, neutral, factual description."
        self.renderer_system = "You are a factual renderer. Use only the neutral input and numeric AFD metrics to construct the answer; do NOT provide chain-of-thought."

        # lazy model slots
        self._hf_llm = None
        self.sentiment_analyzer = None

    # Optional cache helpers (Streamlit cache if available)
    if st is not None:
        @st.cache_resource
        def _cache_sentiment(_self):
            if pipeline is None:
                return None
            try:
                return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            except Exception:
                return None
    else:
        def _cache_sentiment(_self):
            return None

    def _ensure_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = self._cache_sentiment()
            except Exception:
                self.sentiment_analyzer = None
        return self.sentiment_analyzer

    #
    # Neutralizers
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
        if len(trimmed) > 150:
            trimmed = trimmed[:147].rstrip() + "…"
        return f"neutral summary: {trimmed}"

    #
    # Intent detection (deterministic)
    #
    def _detect_intent(self, neutral_prompt: str) -> str:
        nprompt = (neutral_prompt or "").lower().strip()
        # identity patterns
        if re.search(r"\b(who (am i|are you|is this|am i speaking to|am i talking to)|are you a|who are you)\b", nprompt):
            return "identity"
        if re.search(r"\b(agent identity|agent identity inquiry|identity inquiry)\b", nprompt):
            return "identity"
        # opinion/thoughts
        if re.search(r"\b(your thoughts|what are your thoughts|what do you think|your view|what's your view|do you think)\b", nprompt):
            return "opinion"
        # definitions
        if re.match(r"^(definition|definition/explanation request:|what is|what are|define)\b", nprompt):
            return "definition"
        # explain/describe
        if nprompt.startswith(("explain ", "describe ", "summarize ", "compare ", "tell me about ")):
            return "explain"
        return "unknown"

    #
    # AFD math helpers (proxies)
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
    # Deterministic AFD renderer (intent-aware)
    #
    def _afd_renderer(self, neutral_prompt: str, afd_directives: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, seed: Optional[int] = None, intent: Optional[str] = None, chat_context: Optional[str] = None) -> str:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        temp = float(max(0.0, min(self.temperature, 3.0)))
        noise_scale = max(0.0, temp * 0.06)

        # defensive metrics handling
        if not isinstance(metrics, dict):
            metrics = {"harmony": 0.0, "info_gradient": 0.0, "oscillation": 0.0, "potential": 0.0}
        harmony = float(metrics.get("harmony", 0.0))
        ig = float(metrics.get("info_gradient", 0.0))
        oscill = float(metrics.get("oscillation", 0.0))
        potential = float(metrics.get("potential", 0.0))

        # identity target
        if intent == "identity":
            id_ref = self.generate_identity_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
            answer = "You are speaking to AFD∞-AMI — an algorithmic assistant. I am not conscious."
            return f"{id_ref}\n\nAnswer:\n{answer}"

        # default knowledge template (concise)
        lines = [f"Neutral input: {neutral_prompt}", ""]
        lines.append("Definition:")
        lines.append("Life is a self-maintaining, information-processing system that uses energy to sustain organized structure, replicate, and evolve over time.")
        if harmony > 0.15 or ig > 0.2:
            lines.append("")
            lines.append("Supporting points:")
            if harmony > 0.15:
                lines.append("- Organization and homeostasis")
            if ig > 0.2:
                lines.append("- Information storage and regulatory networks")
        lines.append("")
        lines.append(f"AFD rationale: coherence={coherence:.3f}, harmony={harmony:.3f}, info_grad={ig:.3f}")
        # keep concise
        return "\n".join(lines)

    #
    # Reflection helpers (safe, deterministic, audit-ready)
    #
    def generate_identity_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None) -> str:
        coeffs = f"alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}"
        mem_note = ""
        try:
            if memory_summary and memory_summary.get("count", 0) > 0:
                mem_note = f" I recall {memory_summary.get('count')} recent interactions."
        except Exception:
            mem_note = ""
        return "I am AFD∞-AMI — an algorithmic assistant (not conscious). I neutralize prompts, compute AFD metrics, produce auditable answers, and accept explicit feedback." + f" ({coeffs})" + mem_note

    def generate_human_style_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None) -> str:
        try:
            c = float(coherence)
        except Exception:
            c = 0.0
        h = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        confidence = "high" if c >= 0.8 else ("low" if c <= 0.35 else "moderate")
        return f"I assessed this request with {confidence} confidence (coherence={c:.3f}, harmony={h:.3f})."

    def generate_simulated_thinking_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, max_steps: int = 5):
        try:
            c = float(coherence)
        except Exception:
            c = 0.0
        h = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        ig = float(metrics.get("info_gradient", 0.0)) if isinstance(metrics, dict) else 0.0
        lines = []
        lines.append("Thinking: " + ("clear signal." if c >= 0.8 else ("uncertain signal." if c <= 0.35 else "moderate signal.")))
        if h > 0.6:
            lines.append("Indicators align — focus on main points.")
        elif h < 0.2:
            lines.append("Mixed indicators — highlight ambiguities.")
        if ig > 0.5:
            lines.append("Supporting detail is available.")
        lines.append("Now producing an auditable answer.")
        return lines[:max_steps]

    def generate_reasoning_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, verbosity: int = 1, seed: Optional[int] = None) -> str:
        try:
            c = float(coherence)
        except Exception:
            c = 0.0
        conf = "high" if c >= 0.8 else ("low" if c <= 0.35 else "moderate")
        return f"(Reasoning) Confidence: {conf}. I will produce a {'concise' if conf=='high' else 'cautious' if conf=='low' else 'balanced'} answer."

    #
    # Authoritative structured AFD reflection
    #
    def generate_afd_reflection(self, prompt: str, neutral_prompt: str, state, action, s_prime, interp_s, metrics: dict, coherence: float, renderer_used: str) -> str:
        try:
            s_list = np.array(state).tolist() if hasattr(np, "array") else state
        except Exception:
            s_list = state
        try:
            a_list = np.array(action).tolist()
        except Exception:
            a_list = action
        try:
            sp_list = np.array(s_prime).tolist()
        except Exception:
            sp_list = s_prime
        try:
            ip_list = np.array(interp_s).tolist()
        except Exception:
            ip_list = interp_s
        harmony = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        ig = float(metrics.get("info_gradient", 0.0)) if isinstance(metrics, dict) else 0.0
        oscill = float(metrics.get("oscillation", 0.0)) if isinstance(metrics, dict) else 0.0
        phi = float(metrics.get("potential", 0.0)) if isinstance(metrics, dict) else 0.0

        lines = [
            "AFD Reflection:",
            f"- original_prompt: {prompt}",
            f"- neutral_prompt: {neutral_prompt}",
            f"- state (proxy): {s_list}",
            f"- action proxy: {a_list}",
            f"- predicted s_prime: {sp_list}",
            f"- interp_s: {ip_list}",
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
    # Persistence & explainability
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

    def load_chat_history(self, n: Optional[int] = None):
        try:
            if not os.path.exists(self.chat_file):
                return []
            with open(self.chat_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            entries = []
            for L in lines:
                try:
                    entries.append(json.loads(L))
                except Exception:
                    continue
            if n is None:
                return entries
            return entries[-n:]
        except Exception:
            self.reflection_log.append("Warning: could not load chat history.")
            return []

    def load_memory(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"])

    def get_last_explainability(self) -> Optional[dict]:
        return getattr(self, "latest_explainability", None)

    #
    # Memory summarization
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

    #
    # Feedback / learning
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
        # conservative learning update if explain found
        try:
            found = None
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
            nh, nig, no_, nphi = h / mag, ig / mag, o / mag, phi / mag
            error = float(outcome_score) - float(observed_coherence)
            lr = float(self._learning_rate)
            self.alpha = max(self._coeff_min, min(self._coeff_max, self.alpha + lr * error * nh))
            self.beta = max(self._coeff_min, min(self._coeff_max, self.beta + lr * error * nig))
            self.gamma = max(self._coeff_min, min(self._coeff_max, self.gamma - lr * error * no_))
            self.delta = max(self._coeff_min, min(self._coeff_max, self.delta + lr * error * nphi))
            self.reflection_log.append(f"Learning update: err={error:.4f}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}")
        except Exception:
            self.reflection_log.append("Learning update failed.")
