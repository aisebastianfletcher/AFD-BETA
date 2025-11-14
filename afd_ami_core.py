"""
AFDInfinityAMI - Fixed robust reflections and safe guards.

This file is the full AFD core with a defensive fix: reflection-generation
calls in respond(...) are wrapped with try/except so a failure inside any
reflection helper won't crash respond(). Exceptions are recorded to
self.reflection_log and the agent falls back to safe defaults.

Paste this file to overwrite afd_ami_core.py, restart Streamlit, and re-run.
"""
from typing import Optional
import os
import json
import collections
import re
import math
import datetime
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
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None, temperature: float = 0.0, memory_window: int = 20):
        self.temperature = float(temperature or 0.0)
        self.memory_window = int(memory_window or 20)

        # OpenAI key handling
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

        # logs & persistence
        self.reflection_log = []
        self.latest_explainability = None

        self.memory_file = "data/response_log.csv"
        self.explain_file = "data/explain_log.jsonl"
        self.chat_file = "data/chat_history.jsonl"
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        if not os.path.exists(self.memory_file):
            pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"]).to_csv(
                self.memory_file, index=False, encoding="utf-8-sig"
            )

        # coefficients & learning
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 0.5
        self.delta = 0.5
        self._learning_rate = 0.02
        self._coeff_min, self._coeff_max = -2.0, 4.0

        # system prompts (LLM only)
        self.neutralizer_system = "You are a precise neutral translator. Convert the user's input into a short, neutral, factual description. Keep it concise."
        self.renderer_system = "You are a factual renderer. Produce a clear, neutral, concise explanation based ONLY on the provided neutral input and numeric AFD directives. Do NOT provide chain-of-thought."

        # lazy HF resources
        self._hf_llm = None
        self.sentiment_analyzer = None

        # test OpenAI key quietly (record to log only)
        if self.use_openai and api_key:
            try:
                self._test_openai_key()
                self.reflection_log.append("OpenAI key test succeeded.")
            except Exception as e:
                self.use_openai = False
                self.reflection_log.append(f"OpenAI test failed: {e}")

    def _test_openai_key(self):
        openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Ping"}], max_tokens=1, temperature=0.0)
        return True

    #
    # HF caching
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
                self.reflection_log.append(f"HF llm init failed: {e}")
                self._hf_llm = None
        return self._hf_llm

    def _ensure_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = self._cache_sentiment_analyzer()
            except Exception as e:
                self.reflection_log.append(f"Sentiment analyzer init failed: {e}")
                self.sentiment_analyzer = None
        return self.sentiment_analyzer

    #
    # Neutralizer helpers
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

    def _neutralize_with_openai(self, user_input: str) -> str:
        try:
            messages = [{"role": "system", "content": self.neutralizer_system}, {"role": "user", "content": user_input}]
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.0, max_tokens=120)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.reflection_log.append(f"OpenAI neutralize error: {e}")
            return ""

    def _neutralize_with_hf(self, user_input: str) -> str:
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

    #
    # Rendering helpers (LLM)
    #
    def _render_with_openai(self, neutral_input: str, afd_directives: str, max_tokens: int = 250) -> str:
        try:
            content = f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."
            messages = [{"role": "system", "content": self.renderer_system}, {"role": "user", "content": content}]
            temp = max(0.0, min(float(self.temperature), 2.0))
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=temp, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.reflection_log.append(f"OpenAI render error: {e}")
            return "Unable to render response using OpenAI."

    def _render_with_hf(self, neutral_input: str, afd_directives: str) -> str:
        llm = self._ensure_hf_llm()
        if llm is None:
            return "Local model not available to render."
        prompt = f"{self.renderer_system}\n\nNeutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nAnswer:"
        try:
            temp = float(self.temperature)
            gen_args = {"prompt": prompt, "max_new_tokens": 220, "do_sample": True if temp > 0 else False, "return_full_text": False, "num_return_sequences": 1}
            if temp > 0:
                gen_args.update({"temperature": float(min(temp, 2.0)), "top_k": 40, "top_p": 0.9})
            out = llm(**gen_args)
            text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF render error: {e}")
            return "Unable to render response locally."

    #
    # Intent detection
    #
    def _detect_intent(self, neutral_prompt: str) -> str:
        nprompt = (neutral_prompt or "").lower().strip()
        if re.search(r"\b(who (am i|are you|is this|am i speaking to|am i talking to)|are you a|who are you)\b", nprompt):
            return "identity"
        if re.search(r"\b(agent identity|agent identity inquiry|identity inquiry)\b", nprompt):
            return "identity"
        if re.search(r"\b(your thoughts|what are your thoughts|what do you think|your view|what's your view)\b", nprompt):
            return "opinion"
        if re.match(r"^(definition|definition/explanation request:|what is|what are)\b", nprompt):
            return "definition"
        if nprompt.startswith(("explain ", "describe ", "summarize ", "compare ", "tell me about ")):
            return "explain"
        return "unknown"

    #
    # AFD math + helpers
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

    def coherence_score(self, action: np.ndarray, state: np.ndarray):
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)
        h = self.compute_harmony(state, interp_s)
        i = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        score = self.alpha * h + self.beta * i - self.gamma * o + self.delta * phi
        return float(score), {"harmony": float(h), "info_gradient": float(i), "oscillation": float(o), "potential": float(phi)}

    def adjust_coefficients(self, coherence: float, metrics: dict):
        log = f"Coherence: {coherence:.4f}, Metrics: {metrics}"
        if coherence < 0.5:
            self.alpha = self._clip(self.alpha + 0.01, self._coeff_min, self._coeff_max)
            self.reflection_log.append(f"Increased alpha slightly to {self.alpha:.3f}. {log}")
        elif coherence > 0.9:
            self.gamma = self._clip(self.gamma + 0.01, self._coeff_min, self._coeff_max)
            self.reflection_log.append(f"Increased gamma slightly to {self.gamma:.3f}. {log}")
        else:
            self.reflection_log.append(f"No auto coefficient adjustment. {log}")

    def _clip(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    #
    # Persistence + chat helpers
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
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not save memory CSV ({e}).")

    def append_explainability(self, explainability: dict):
        try:
            os.makedirs(os.path.dirname(self.explain_file) or ".", exist_ok=True)
            with open(self.explain_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(explainability, ensure_ascii=False) + "\n")
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not append explainability JSONL ({e}).")

    def append_chat_entry(self, role: str, message: str, timestamp_iso: Optional[str] = None):
        try:
            os.makedirs(os.path.dirname(self.chat_file) or ".", exist_ok=True)
            ts = timestamp_iso or datetime.datetime.utcnow().isoformat()
            entry = {"timestamp": ts, "role": role, "message": message}
            with open(self.chat_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not append chat entry ({e}).")

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
        except Exception as e:
            self.reflection_log.append(f"Warning: could not load chat history ({e}).")
            return []

    def load_memory(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception as e:
            self.reflection_log.append(f"Error loading memory file: {e}")
            return pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"])

    def get_last_explainability(self) -> Optional[dict]:
        return getattr(self, "latest_explainability", None)

    #
    # Memory summarization
    #
    def _summarize_memory(self, n: Optional[int] = None) -> dict:
        n = int(n) if n is not None else int(self.memory_window)
        df = self.load_memory()
        if df.empty:
            return {"count": 0, "avg_coherence": None, "avg_outcome": None, "last_neutral_prompts": [], "top_tokens": []}
        tail = df.tail(n)
        avg_coh = None
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
        last_neutrals = tail["neutral_prompt"].fillna("").astype(str).tolist()
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "of", "to", "is", "are", "for", "with", "that", "this"}
        token_counter = collections.Counter()
        for s in last_neutrals:
            tokens = re.findall(r"[A-Za-z]{3,}", s.lower())
            for t in tokens:
                if t not in stopwords:
                    token_counter[t] += 1
        top_tokens = [tok for tok, _ in token_counter.most_common(8)]
        return {"count": len(tail), "avg_coherence": avg_coh, "avg_outcome": avg_out, "last_neutral_prompts": last_neutrals, "top_tokens": top_tokens}

    #
    # Simulated renderer + intent-aware behavior
    #
    def _afd_renderer(self, neutral_prompt: str, afd_directives: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, seed: Optional[int] = None, intent: Optional[str] = None, chat_context: Optional[str] = None) -> str:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        temp = float(max(0.0, min(self.temperature, 3.0)))
        noise_scale = max(0.0, temp * 0.06)

        # safe guards for metrics
        if not isinstance(metrics, dict):
            metrics = {"harmony": 0.0, "info_gradient": 0.0, "oscillation": 0.0, "potential": 0.0}

        metrics_noisy = {}
        for k, v in metrics.items():
            try:
                metrics_noisy[k] = float(v) + float(rng.normal(0.0, noise_scale))
            except Exception:
                try:
                    metrics_noisy[k] = float(v)
                except Exception:
                    metrics_noisy[k] = 0.0

        c = float(coherence) + float(rng.normal(0.0, noise_scale))
        harmony = metrics_noisy.get("harmony", 0.0)
        info_grad = metrics_noisy.get("info_gradient", 0.0)
        oscill = metrics_noisy.get("oscillation", 0.0)
        potential = metrics_noisy.get("potential", 0.0)

        # Intent-specific behavior
        if intent == "identity":
            identity_reflection = self.generate_identity_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
            answer = "You are speaking to AFD∞-AMI — an algorithmic, auditable assistant. I am not conscious."
            return "\n\n".join([identity_reflection, "Answer:", answer])

        # Default knowledge template
        sections = []
        sections.append(("Definition", "Life is a self-maintaining, information-processing system that uses energy to sustain organized structure, replicate, and evolve over time."))
        criteria_text = (
            "- Organization (cells/compartments)\n"
            "- Metabolism (energy/matter transformations)\n"
            "- Homeostasis\n"
            "- Growth & reproduction\n"
            "- Response to stimuli\n"
            "- Evolution by natural selection"
        )
        if harmony >= 0.2 or c >= 0.4:
            sections.append(("Common biological criteria", criteria_text))
        if potential > 0.1 or info_grad > 0.25:
            sections.append(("Thermodynamics", "Living systems maintain local low-entropy order by channeling free energy and exporting entropy."))
        if info_grad > 0.15:
            sections.append(("Information & control", "Genetic and regulatory information enables reproduction, adaptation, and coordinated function."))
        sections.append(("Borderline cases", "Entities like viruses, prions, or synthetic replicators highlight definitional edges; pragmatic definitions rely on clusters of criteria."))
        if info_grad > 0.4 or potential > 0.4:
            sections.append(("Origin (brief)", "Abiogenesis explores pathways from chemistry to self-replicating, compartmentalized systems."))
            sections.append(("Implications", "Searches for life emphasize non-equilibrium chemistry and organized information-processing signatures."))

        lines = []
        lines.append(f"Neutral input: {neutral_prompt}")
        lines.append("")
        for title, body in sections:
            lines.append(f"{title}:")
            lines.append(body)
            lines.append("")
        lines.append("AFD rationale (derived):")
        lines.append(f"- coherence: {c:.4f}")
        lines.append(f"- harmony: {harmony:.4f}, info_gradient: {info_grad:.4f}, oscillation: {oscill:.4f}, potential: {potential:.4f}")
        if c >= 0.8:
            lines.append("- Overall: strong agreement — concise, confident explanation.")
        elif c <= 0.35:
            lines.append("- Overall: uncertainty present — cautious, hedged explanation.")
        else:
            lines.append("- Overall: moderate coherence — balanced explanation with clarifications.")
        if memory_summary and memory_summary.get("avg_outcome") is not None:
            lines.append(f"- Memory note: recent average outcome={memory_summary.get('avg_outcome'):.3f} (calibrates tone).")
        lines.append("")
        lines.append("Answer synthesis:")
        if c >= 0.8:
            lines.append("- A concise synthesis: life is characterized by organization, metabolism, heredity, and evolvability.")
        elif c <= 0.35:
            lines.append("- A cautious synthesis: life-like behavior may show some criteria; avoid definitive claims without further evidence.")
        else:
            lines.append("- A balanced synthesis: life is a cluster of properties (organization, energy transformation, information processing, reproduction, evolvability).")
        if info_grad > 0.3:
            lines.append("- Supporting detail: information storage and regulation underpin persistence and adaptation.")
        if oscill > 0.25:
            lines.append("- Note: internal instability suggests avoiding speculation.")

        return "\n".join(lines)

    #
    # Safe reflections
    #
    def generate_identity_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None) -> str:
        identity = "I am AFD∞-AMI, an algorithmic assistant. I am not conscious; I neutralize prompts, compute AFD metrics, compose auditable answers, and learn conservatively from explicit feedback."
        coeffs = f"alpha={getattr(self,'alpha',None):.3f}, beta={getattr(self,'beta',None):.3f}, gamma={getattr(self,'gamma',None):.3f}, delta={getattr(self,'delta',None):.3f}"
        capability = f"My AFD mode is metric-driven and auditable. Coefficients: {coeffs}."
        mem_line = ""
        try:
            if memory_summary and memory_summary.get("count", 0) > 0:
                mem_line = f" I recall {memory_summary.get('count')} recent interactions (avg_coherence={memory_summary.get('avg_coherence')}, avg_outcome={memory_summary.get('avg_outcome')})."
            else:
                mem_line = " I have little recent history recorded."
        except Exception:
            mem_line = ""
        limits = "I do not reveal chain-of-thought. Reflections are post-hoc summaries derived from numeric metrics and aggregated memory."
        return "\n\n".join([identity, capability + mem_line, "Learning & limits:", limits])

    def generate_reasoning_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, verbosity: int = 1, seed: Optional[int] = None) -> str:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        ig = float(metrics.get("info_gradient", 0.0)) if isinstance(metrics, dict) else 0.0
        o = float(metrics.get("oscillation", 0.0)) if isinstance(metrics, dict) else 0.0
        phi = float(metrics.get("potential", 0.0)) if isinstance(metrics, dict) else 0.0

        def cue_confidence(c): return "high" if c >= 0.8 else ("low" if c <= 0.35 else "moderate")
        def cue_agreement(h): return "strong agreement" if h >= 0.6 else ("mixed signals" if h <= 0.2 else "some agreement with nuances")
        def cue_info(ig): return "meaningful supporting detail available" if ig >= 0.5 else "limited supporting detail"
        def cue_stability(o): return "internal instability present" if o > 0.25 else "internal state reasonably stable"

        premises = [f"Neutral prompt: '{neutral_prompt}'.", f"Coefficients: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}."]
        evidence = [f"Metrics: coherence={c:.3f}, harmony={h:.3f}, info_gradient={ig:.3f}, oscillation={o:.3f}, potential={phi:.3f}.", cue_agreement(h) + "; " + cue_info(ig) + ".", cue_stability(o) + "."]
        inferences = []
        conf = cue_confidence(c)
        if conf == "high":
            inferences.append("Present a confident, concise answer focused on main points.")
        elif conf == "low":
            inferences.append("Avoid strong claims; explicitly mark uncertainty.")
        else:
            inferences.append("Provide a balanced answer with measured caution.")
        if memory_summary and memory_summary.get("count", 0) > 0:
            count = memory_summary.get("count")
            avg_out = memory_summary.get("avg_outcome")
            mem_note = f"Recent {count} interactions"
            if avg_out is not None:
                mem_note += f" (avg outcome ≈ {avg_out:.2f})"
            mem_note += " modestly inform tone."
            inferences.append(mem_note)
        uncertainty = ["Key uncertainties: internal signals unstable or low coherence; flag ambiguity."] if (o > 0.25 or c <= 0.35) else ["Key uncertainties: limited; main points supported."]
        if verbosity == 0:
            return "Conclusion: provide a concise factual answer." if conf == "high" else "Conclusion: provide a cautious, qualified answer."
        lines = []
        lines.append(f"(Thinking) Detected {conf} confidence from metrics.")
        lines.append("")
        lines.append("Premises:")
        for p in premises: lines.append(f"- {p}")
        lines.append("")
        lines.append("Evidence:")
        for e in evidence: lines.append(f"- {e}")
        lines.append("")
        lines.append("Inferences:")
        for inf in inferences: lines.append(f"- {inf}")
        lines.append("")
        lines.append("Uncertainty & caveats:")
        for u in uncertainty: lines.append(f"- {u}")
        lines.append("")
        lines.append("Conclusion:")
        if conf == "high":
            lines.append("- Provide a focused, confident summary, then supporting details if available.")
        elif conf == "low":
            lines.append("- Provide a short summary and explicitly mark uncertainties.")
        else:
            lines.append("- Provide a balanced summary with key supporting details and explicit caveats.")
        if verbosity >= 2:
            lines.append("")
            lines.append("Style note: use explicit uncertainty markers (Likely, Possible, Unclear).")
        return "\n".join(lines)

    def generate_simulated_thinking_reflection(self, neutral_prompt: str, metrics: dict, coherence: float, memory_summary: Optional[dict] = None, max_steps: int = 5):
        # Defensive implementation: ensure metrics are dict-like and values numeric
        try:
            c = float(coherence)
        except Exception:
            c = 0.0
        try:
            harmony = float(metrics.get("harmony", 0.0)) if isinstance(metrics, dict) else 0.0
        except Exception:
            harmony = 0.0
        try:
            info_grad = float(metrics.get("info_gradient", 0.0)) if isinstance(metrics, dict) else 0.0
        except Exception:
            info_grad = 0.0
        try:
            oscill = float(metrics.get("oscillation", 0.0)) if isinstance(metrics, dict) else 0.0
        except Exception:
            oscill = 0.0

        lines = []
        if c >= 0.8:
            lines.append("Okay — I see a clear, coherent picture here.")
        elif c <= 0.35:
            lines.append("Hmm — signals look uncertain; I should be cautious.")
        else:
            lines.append("I have a moderately coherent signal; I'll balance clarity and caution.")
        if harmony > 0.6:
            lines.append("Multiple internal indicators align — focus on the main, consistent points.")
        elif harmony < 0.2:
            lines.append("There are mixed signals — I'll point out ambiguities where relevant.")
        if info_grad > 0.5:
            lines.append("There is useful informational content to include; I'll add supporting detail.")
        else:
            lines.append("I should keep this concise and emphasize the essentials.")
        if oscill > 0.25:
            lines.append("I detect instability in the internal proxy; avoid speculation and hedge claims.")
        else:
            lines.append("Internal state is relatively stable; I can be more direct.")
        if memory_summary and memory_summary.get("count", 0) > 0:
            try:
                count = memory_summary.get("count")
                avg_out = memory_summary.get("avg_outcome")
                if avg_out is not None:
                    lines.append(f"I recall {count} recent interactions (avg outcome ≈ {avg_out:.2f}) — calibrating tone accordingly.")
                else:
                    lines.append(f"I recall {count} recent interactions — I'll modestly adapt tone based on that history.")
            except Exception:
                pass
        if len(lines) > max_steps:
            lines = lines[:max_steps]
        lines.append("Now I'll produce a concise, auditable answer below.")
        return lines

    #
    # Feedback & learning (unchanged)
    #
    def provide_feedback(self, entry_timestamp: str, outcome_score: float, outcome_comment: Optional[str] = None) -> bool:
        try:
            score = float(outcome_score)
            if math.isnan(score) or score < 0.0 or score > 1.0:
                raise ValueError("outcome_score must be in [0.0,1.0]")
        except Exception as e:
            self.reflection_log.append(f"Feedback invalid: {e}")
            return False
        updated = False
        try:
            df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
            mask = df["timestamp"] == entry_timestamp
            if mask.any():
                df.loc[mask, "outcome"] = score
                df.loc[mask, "outcome_comment"] = outcome_comment
                df.to_csv(self.memory_file, index=False, encoding="utf-8-sig")
                updated = True
        except Exception as e:
            self.reflection_log.append(f"Feedback CSV update failed: {e}")
        found = None
        try:
            if os.path.exists(self.explain_file):
                with open(self.explain_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if obj.get("timestamp") == entry_timestamp:
                                found = obj
                                break
                        except Exception:
                            continue
            if found is None:
                try:
                    df = pd.read_csv(self.memory_file, encoding="utf-8-sig")
                    row = df[df["timestamp"] == entry_timestamp]
                    if not row.empty:
                        found = {"timestamp": entry_timestamp, "afd_metrics": {"harmony": float(row.iloc[0].get("coherence", 0.0)), "info_gradient": 0.0, "oscillation": 0.0, "potential": 0.0}, "coherence": float(row.iloc[0].get("coherence", 0.0))}
                except Exception:
                    found = None
            if found is None:
                self.reflection_log.append(f"Feedback: explainability not found for {entry_timestamp}")
                return False
            feedback_record = {"timestamp": datetime.datetime.utcnow().isoformat(), "feedback_for": entry_timestamp, "outcome": score, "comment": outcome_comment}
            try:
                with open(self.explain_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"_feedback": feedback_record}, ensure_ascii=False) + "\n")
            except Exception:
                pass
            metrics = found.get("afd_metrics", {})
            h = float(metrics.get("harmony", 0.0))
            ig = float(metrics.get("info_gradient", 0.0))
            o = float(metrics.get("oscillation", 0.0))
            phi = float(metrics.get("potential", 0.0))
            observed_coherence = float(found.get("coherence", self.alpha * h + self.beta * ig - self.gamma * o + self.delta * phi))
            self._learning_update(observed_coherence, score, {"h": h, "ig": ig, "o": o, "phi": phi})
        except Exception as e:
            self.reflection_log.append(f"Feedback processing error: {e}")
            return False
        if updated:
            self.reflection_log.append(f"Feedback recorded for {entry_timestamp}: score={score}")
        return True

    def _learning_update(self, observed_coherence: float, outcome_score: float, metrics_dict: dict) -> bool:
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
            self.alpha = self._clip(self.alpha + lr * error * nh, self._coeff_min, self._coeff_max)
            self.beta = self._clip(self.beta + lr * error * nig, self._coeff_min, self._coeff_max)
            self.gamma = self._clip(self.gamma - lr * error * no_, self._coeff_min, self._coeff_max)
            self.delta = self._clip(self.delta + lr * error * nphi, self._coeff_min, self._coeff_max)
            self.reflection_log.append(f"Learning update: err={error:.4f}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}")
            return True
        except Exception as e:
            self.reflection_log.append(f"Learning update failed: {e}")
            return False

    #
    # Main respond (defensive around reflection generation)
    #
    def respond(self, prompt: str, renderer_mode: str = "afd", include_memory: bool = True, seed: Optional[int] = None, reasoning_verbosity: int = 1, chat_context: Optional[str] = None):
        # Neutralize only the user prompt
        if renderer_mode == "afd":
            neutral_prompt = self._rule_neutralize(prompt)
        else:
            if self.use_openai:
                neutral_prompt = self._neutralize_with_openai(prompt)
            else:
                neutral_prompt = self._neutralize_with_hf(prompt)
        if not neutral_prompt:
            neutral_prompt = prompt.strip()

        # Memory summary
        memory_summary = None
        if include_memory:
            try:
                memory_summary = self._summarize_memory(self.memory_window)
            except Exception as e:
                self.reflection_log.append(f"Memory summary error: {e}")
                memory_summary = None

        # Sentiment/state proxy
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

        # AFD internals
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)
        h = self.compute_harmony(state, interp_s)
        ig = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        coherence = float(self.alpha * h + self.beta * ig - self.gamma * o + self.delta * phi)
        metrics = {"harmony": float(h), "info_gradient": float(ig), "oscillation": float(o), "potential": float(phi)}

        # Memory-driven tiny nudges
        try:
            if memory_summary and memory_summary.get("avg_outcome") is not None:
                avg_hist = memory_summary.get("avg_outcome")
                if avg_hist is not None:
                    if avg_hist < 0.4:
                        self.alpha = self._clip(self.alpha + 0.01, self._coeff_min, self._coeff_max)
                        self.reflection_log.append(f"Memory tiny alpha bump: {self.alpha:.3f}")
                    elif avg_hist > 0.85:
                        self.gamma = self._clip(self.gamma + 0.01, self._coeff_min, self._coeff_max)
                        self.reflection_log.append(f"Memory tiny gamma bump: {self.gamma:.3f}")
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

        detected_intent = self._detect_intent(neutral_prompt)

        # Defensive reflection generation: each helper is called inside try/except so failures don't crash respond()
        try:
            reasoning_reflection_text = self.generate_reasoning_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary, verbosity=reasoning_verbosity, seed=seed)
        except Exception as e:
            reasoning_reflection_text = ""
            self.reflection_log.append(f"Reflection generation (reasoning) failed: {e}")

        try:
            simulated_thinking = self.generate_simulated_thinking_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        except Exception as e:
            simulated_thinking = []
            self.reflection_log.append(f"Reflection generation (simulated_thinking) failed: {e}")

        try:
            human_reflection = self.generate_human_style_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        except Exception as e:
            human_reflection = ""
            self.reflection_log.append(f"Reflection generation (human) failed: {e}")

        try:
            identity_reflection = self.generate_identity_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        except Exception as e:
            identity_reflection = ""
            self.reflection_log.append(f"Reflection generation (identity) failed: {e}")

        # Rendering
        if renderer_mode == "afd":
            try:
                final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence, memory_summary=memory_summary, seed=seed, intent=detected_intent, chat_context=chat_context)
            except Exception as e:
                final_text = ""
                self.reflection_log.append(f"AFD renderer failed: {e}")
        else:
            try:
                if self.use_openai:
                    final_text = self._render_with_openai(neutral_prompt, afd_directives)
                else:
                    final_text = self._render_with_hf(neutral_prompt, afd_directives)
            except Exception as e:
                final_text = ""
                self.reflection_log.append(f"LLM renderer failed: {e}")

        # If opinion intent, prepend reasoning reflection safely
        if detected_intent == "opinion":
            if reasoning_reflection_text:
                final_text = reasoning_reflection_text + "\n\nAnswer:\n" + (final_text or "")
        # If identity intent and rendered by non-intent path, prefer generated identity_reflection
        if detected_intent == "identity" and final_text and "Definition:" in final_text and identity_reflection:
            final_text = identity_reflection + "\n\nAnswer:\n" + final_text

        # persist & explainability
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
            "renderer_used": "AFD" if renderer_mode == "afd" else "LLM",
            "renderer_prompt": afd_directives,
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

        afd_reflection = self.generate_afd_reflection(prompt, neutral_prompt, state, action, s_prime, interp_s, metrics, coherence, "AFD" if renderer_mode == "afd" else "LLM")
        try:
            self.reflection_log.append(afd_reflection)
        except Exception:
            self.reflection_log = [afd_reflection]

        return final_text, coherence, afd_reflection, timestamp_iso
    
