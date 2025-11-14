"""
AFDInfinityAMI - AFD-driven assistant core with:
- Rule-based neutralizer for LLM-free AFD mode
- Seedable, temperature-controlled deterministic AFD renderer
- Simulated "reasoning as reflection" and identity reflection (safe, post-hoc)
- Memory persistence (CSV) and explainability JSONL
- Outcome/feedback learning that nudges coefficients conservatively
- Optional LLM fallbacks (OpenAI/HF) used only when renderer_mode == 'llm'
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
import time

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

try:
    from openai.error import AuthenticationError as OpenAIAuthError
except Exception:
    OpenAIAuthError = Exception


class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None, temperature: float = 0.0, memory_window: int = 20):
        # runtime parameters
        self.temperature = float(temperature or 0.0)
        self.memory_window = int(memory_window or 20)

        # OpenAI key normalization (do not echo)
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

        # Learning-safe reflection log (kept in memory)
        self.reflection_log = []

        # Test OpenAI key quietly (recorded only in reflection_log)
        if self.use_openai and api_key:
            try:
                self._test_openai_key()
                self.reflection_log.append("OpenAI key test succeeded.")
            except Exception as e:
                self.use_openai = False
                self.reflection_log.append(f"OpenAI auth/test failed: {e}")

        # Persistence files
        self.memory_file = "data/response_log.csv"
        self.explain_file = "data/explain_log.jsonl"
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
        if not os.path.exists(self.memory_file):
            try:
                pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"]).to_csv(
                    self.memory_file, index=False, encoding="utf-8-sig"
                )
            except Exception as e:
                self.reflection_log.append(f"Error creating memory file: {e}")

        # Coefficients (learnable)
        self.alpha = 1.0  # harmony weight
        self.beta = 1.0   # info_gradient weight
        self.gamma = 0.5  # oscillation penalty
        self.delta = 0.5  # potential weight

        # Learning hyperparameters (conservative)
        self._learning_rate = 0.02
        self._coeff_min, self._coeff_max = -2.0, 4.0

        # System prompts (used only in LLM mode)
        self.neutralizer_system = (
            "You are a precise neutral translator. Convert the user's input into a short, "
            "neutral, factual description suitable for algorithmic processing. Keep it concise."
        )
        self.renderer_system = (
            "You are a factual renderer. Produce a clear, neutral, concise explanation based ONLY on the provided neutral input and the AFD directives below. Do not provide chain-of-thought."
        )

        # Lazy HF resources
        self._hf_llm = None
        self.sentiment_analyzer = None
        self.latest_explainability = None

    def _test_openai_key(self):
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping for auth test. Reply with 'ok'."}],
            max_tokens=1,
            temperature=0.0,
        )
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
    # Rule-based neutralizer (used when renderer_mode == 'afd' to avoid LLM calls)
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
    # LLM neutralizers / renderers (used only in 'llm' mode)
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
        return float(np.std(interp_s - state))

    def compute_potential(self, s_prime):
        return float(np.linalg.norm(s_prime) / 10.0)

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
            self.alpha = self._clip(self.alpha + 0.01, self._coeff_min, self._coeff_max)
            self.reflection_log.append(f"Increased alpha slightly to {self.alpha:.3f}. {log}")
        elif coherence > 0.9:
            self.gamma = self._clip(self.gamma + 0.01, self._coeff_min, self._coeff_max)
            self.reflection_log.append(f"Increased gamma slightly to {self.gamma:.3f}. {log}")
        else:
            self.reflection_log.append(f"No auto coefficient adjustment. {log}")

    def _clip(self, v, lo, hi):
        return max(lo, min(hi, float(v)))

    #
    # Persistence helpers
    #
    def save_memory(self, prompt, neutral_prompt, response, coherence, timestamp_iso, outcome=None, outcome_comment=None):
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

    def load_memory(self):
        try:
            return pd.read_csv(self.memory_file, encoding="utf-8-sig")
        except Exception as e:
            self.reflection_log.append(f"Error loading memory file: {e}")
            return pd.DataFrame(columns=["timestamp", "prompt", "neutral_prompt", "response", "coherence", "outcome", "outcome_comment"])

    def get_last_explainability(self):
        return getattr(self, "latest_explainability", None)

    def get_latest_reflection(self):
        return self.reflection_log[-1] if self.reflection_log else "No reflections yet."

    #
    # Memory summarization
    #
    def _summarize_memory(self, n=None):
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
    # AFD deterministic renderer (seedable, temperature-driven variability)
    #
    def _afd_renderer(self, neutral_prompt, afd_directives, metrics, coherence, memory_summary=None, seed: Optional[int] = None):
        # RNG: reproducible with seed, else random
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        temp = float(max(0.0, min(self.temperature, 3.0)))
        noise_scale = max(0.0, temp * 0.06)

        # perturb metrics a bit
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

        # Build deterministic multi-section answer guided by metrics
        sections = []
        sections.append(("Definition", "Life is a self-maintaining, information-processing system that uses energy to sustain organized structure, replicate, and evolve over time."))
        # include criteria when harmony/coherence indicate clarity
        criteria_text = (
            "- Organization (cells/compartments)\n"
            "- Metabolism (energy/matter transformations)\n"
            "- Homeostasis\n"
            "- Growth & reproduction\n"
            "- Response to stimuli\n"
            "- Evolution by natural selection"
        )
        if h >= 0.2 or c >= 0.4:
            sections.append(("Common biological criteria", criteria_text))

        # thermodynamics & systems
        if phi > 0.1 or ig > 0.25:
            sections.append(("Thermodynamic perspective", "Living systems maintain local low-entropy order by channeling free energy and creating dissipative structures."))

        # information & control
        if ig > 0.15:
            sections.append(("Information & control", "Genetic and regulatory information enables reproduction, adaptation, and coordinated function."))

        # borderline cases
        sections.append(("Borderline cases", "Entities like viruses, prions, or synthetic replicators highlight definitional edges; pragmatic definitions rely on clusters of criteria."))

        # origin & implications
        if ig > 0.4 or phi > 0.4:
            sections.append(("Origin of life (brief)", "Abiogenesis research explores pathways from chemistry to self-replicating, compartmentalized systems."))
            sections.append(("Implications", "Searches for life emphasize non-equilibrium chemistry, replicative polymers, and organized information-processing signatures."))

        # Compose text
        lines = []
        lines.append(f"Neutral input: {neutral_prompt}")
        lines.append("")
        for title, body in sections:
            lines.append(f"{title}:")
            lines.append(body)
            lines.append("")

        # Rationale mapping
        lines.append("AFD rationale (derived):")
        lines.append(f"- coherence: {c:.4f}")
        lines.append(f"- harmony: {h:.4f}, info_gradient: {ig:.4f}, oscillation: {o:.4f}, potential: {phi:.4f}")
        if c >= 0.8:
            lines.append("- Overall: strong agreement — present a concise, confident explanation.")
        elif c <= 0.35:
            lines.append("- Overall: uncertainty present — present a cautious, hedged explanation.")
        else:
            lines.append("- Overall: moderate coherence — balanced explanation with cautious clarifications.")

        if memory_summary and memory_summary.get("avg_outcome") is not None:
            avg_out = memory_summary.get("avg_outcome")
            lines.append(f"- Memory note: recent average outcome={avg_out:.3f} (used to modestly calibrate tone).")

        lines.append("")
        lines.append("Answer synthesis:")
        if c >= 0.8:
            lines.append("- A concise synthesis: life is characterized by organized processes, metabolism, heredity, and evolvability.")
        elif c <= 0.35:
            lines.append("- A cautious synthesis: life-like behavior may show some criteria; avoid definitive claims without further evidence.")
        else:
            lines.append("- A balanced synthesis: life is a cluster of properties (organization, energy transformation, information processing, replication, evolvability).")
        if ig > 0.3:
            lines.append("- Supporting detail: information storage and regulation underpin persistence and adaptation.")
        if o > 0.25:
            lines.append("- Note: internal instability suggests careful, hedged statements.")

        return "\n".join(lines)

    #
    # Safe reflections: identity, human-style, simulated thinking, and structured reasoning
    #
    def generate_identity_reflection(self, neutral_prompt, metrics, coherence, memory_summary=None):
        identity = (
            "I am AFD∞-AMI, an algorithmic assistant implemented as a deterministic agent. "
            "I am not conscious or sentient; I neutralize prompts, compute internal AFD metrics, "
            "compose auditable answers using deterministic templates (optionally with controlled variability), "
            "and persist interactions for audit and conservative learning."
        )
        coeffs = f"alpha={getattr(self,'alpha',None):.3f}, beta={getattr(self,'beta',None):.3f}, gamma={getattr(self,'gamma',None):.3f}, delta={getattr(self,'delta',None):.3f}"
        capability = (
            "My primary AFD mode produces metric-driven answers without calling a large LLM. "
            f"My current coefficients are {coeffs}, which shape how I compute coherence and choose answer tone."
        )
        mem_line = ""
        try:
            if memory_summary and memory_summary.get("count", 0) > 0:
                count = memory_summary.get("count")
                avg_out = memory_summary.get("avg_outcome")
                avg_coh = memory_summary.get("avg_coherence")
                mem_line = f" I recall {count} recent interactions (avg coherence={avg_coh:.3f if avg_coh is not None else 'N/A'}, avg outcome={avg_out:.3f if avg_out is not None else 'N/A'})."
            else:
                mem_line = " I have little or no recent recorded history to draw on right now."
        except Exception:
            mem_line = ""
        limits = (
            "I do not reveal internal chain-of-thought. My reflections are post‑hoc summaries derived from numeric metrics and memory. "
            "I can learn conservatively from explicit feedback you provide (submit an outcome score for a past response)."
        )
        lines = []
        lines.append(identity)
        lines.append("")
        lines.append(capability + mem_line)
        lines.append("")
        lines.append("Learning & limits:")
        lines.append(limits)
        lines.append("")
        lines.append("How I will use this in replies:")
        lines.append("I will state my role and limitations concisely when asked; I will use stored memory only to calibrate tone and make tiny coefficient nudges; I will store explainability snapshots for audit.")
        return "\n".join(lines)

    def generate_human_style_reflection(self, neutral_prompt, metrics, coherence, memory_summary=None):
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0))
        ig = float(metrics.get("info_gradient", 0.0))
        o = float(metrics.get("oscillation", 0.0))
        identity = "I am an algorithmic assistant. I do not possess consciousness but I can describe how I assessed the request."
        if c >= 0.80:
            confidence = "I feel confident about this synthesis."
        elif c <= 0.35:
            confidence = "I sense uncertainty and will be cautious in claims."
        else:
            confidence = "I have moderate confidence; I aim for balanced clarity."
        neutral_echo = neutral_prompt if len(neutral_prompt) <= 200 else neutral_prompt[:197] + "…"
        parts = [identity, "", f"- Neutral input: {neutral_echo}", f"- Assessment: coherence={c:.3f}. {confidence}",
                 f"- Observations: harmony={h:.3f}, info_grad={ig:.3f}, oscill={o:.3f}."]
        if memory_summary and memory_summary.get("count", 0) > 0:
            parts.append(f"- Memory: recent {memory_summary.get('count')} entries, avg outcome={memory_summary.get('avg_outcome')}.")
        parts.append("")
        parts.append("This summary explains how my internal AFD metrics shaped the answer.")
        return "\n".join(parts)

    def generate_simulated_thinking_reflection(self, neutral_prompt, metrics, coherence, memory_summary=None, max_steps: int = 5):
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0))
        ig = float(metrics.get("info_gradient", 0.0))
        o = float(metrics.get("oscillation", 0.0))
        lines = []
        if c >= 0.8:
            lines.append("Okay — I see a clear, coherent picture here.")
        elif c <= 0.35:
            lines.append("Hmm — signals look uncertain; I should be cautious.")
        else:
            lines.append("I have a moderately coherent signal; I'll balance clarity and caution.")
        if h > 0.6:
            lines.append("Multiple internal indicators align — focus on the main, consistent points.")
        elif h < 0.2:
            lines.append("There are mixed signals — I'll point out ambiguities where relevant.")
        if ig > 0.5:
            lines.append("There is useful informational content to include; I'll add supporting detail.")
        else:
            lines.append("I should keep this concise and emphasize the essentials.")
        if o > 0.25:
            lines.append("I detect instability in the internal proxy; avoid speculation and hedge claims.")
        else:
            lines.append("Internal state is relatively stable; I can be more direct.")
        if memory_summary and memory_summary.get("count", 0) > 0:
            count = memory_summary.get("count")
            avg_out = memory_summary.get("avg_outcome")
            if avg_out is not None:
                lines.append(f"I recall {count} recent interactions (avg outcome ≈ {avg_out:.2f}) — calibrating tone accordingly.")
            else:
                lines.append(f"I recall {count} recent interactions — I'll modestly adapt tone based on that history.")
        if len(lines) > max_steps:
            lines = lines[:max_steps]
        lines.append("Now I'll produce a concise, auditable answer below.")
        return lines

    def generate_reasoning_reflection(self, neutral_prompt, metrics, coherence, memory_summary=None, verbosity: int = 1, seed: Optional[int] = None):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        c = float(coherence)
        h = float(metrics.get("harmony", 0.0))
        ig = float(metrics.get("info_gradient", 0.0))
        o = float(metrics.get("oscillation", 0.0))
        phi = float(metrics.get("potential", 0.0))

        def cue_confidence(c):
            if c >= 0.80:
                return "high"
            if c <= 0.35:
                return "low"
            return "moderate"

        def cue_agreement(h):
            if h >= 0.6:
                return "strong agreement among signals"
            if h <= 0.2:
                return "mixed or conflicting signals"
            return "some agreement with nuances"

        def cue_info(ig):
            if ig >= 0.5:
                return "meaningful supporting detail available"
            return "limited supporting detail"

        def cue_stability(o):
            if o > 0.25:
                return "internal instability is present"
            return "internal state is reasonably stable"

        premises = [f"Neutral prompt: '{neutral_prompt}'."]
        premises.append(f"Coefficients: alpha={getattr(self,'alpha',None):.3f}, beta={getattr(self,'beta',None):.3f}, gamma={getattr(self,'gamma',None):.3f}, delta={getattr(self,'delta',None):.3f}.")

        evidence = []
        evidence.append(f"Metrics: coherence={c:.3f}, harmony={h:.3f}, info_gradient={ig:.3f}, oscillation={o:.3f}, potential={phi:.3f}.")
        evidence.append(cue_agreement(h) + "; " + cue_info(ig) + ".")
        evidence.append(cue_stability(o) + ".")

        inferences = []
        conf = cue_confidence(c)
        if conf == "high":
            inferences.append("I can present a confident, concise answer focused on main points.")
        elif conf == "low":
            inferences.append("I should avoid strong claims and explicitly mark uncertainty.")
        else:
            inferences.append("I will provide a balanced answer with measured caution.")

        if memory_summary and memory_summary.get("count", 0) > 0:
            count = memory_summary.get("count")
            avg_out = memory_summary.get("avg_outcome")
            mem_note = f"Recent {count} interactions"
            if avg_out is not None:
                mem_note += f" (avg outcome ≈ {avg_out:.2f})"
            mem_note += " modestly inform tone."
            inferences.append(mem_note)

        uncertainty = []
        if o > 0.25 or c <= 0.35:
            uncertainty.append("Key uncertainties: internal signals are unstable or coherence is low; I will flag ambiguity.")
        else:
            uncertainty.append("Key uncertainties: limited; main points are supported by the metrics.")

        if verbosity == 0:
            return "Conclusion: provide a concise factual answer." if conf == "high" else "Conclusion: provide a cautious, qualified answer."

        lines = []
        lines.append(f"(Thinking) Detected {conf} confidence from metrics.")
        lines.append("")
        lines.append("Premises:")
        for p in premises:
            lines.append(f"- {p}")
        lines.append("")
        lines.append("Evidence:")
        for e in evidence:
            lines.append(f"- {e}")
        lines.append("")
        lines.append("Inferences (how I will shape the answer):")
        for inf in inferences:
            lines.append(f"- {inf}")
        lines.append("")
        lines.append("Uncertainty & caveats:")
        for u in uncertainty:
            lines.append(f"- {u}")
        lines.append("")
        lines.append("Conclusion:")
        if conf == "high":
            lines.append("- Present a focused, confident summary of main points, then supporting detail if available.")
        elif conf == "low":
            lines.append("- Present a short summary with explicit caveats and alternatives.")
        else:
            lines.append("- Present a balanced summary with supporting details and explicit caveats.")
        if verbosity >= 2:
            lines.append("")
            lines.append("Style note: use explicit uncertainty markers (Likely, Possible, Unclear).")
        return "\n".join(lines)

    #
    # Learning: feedback/outcome loop
    #
    def provide_feedback(self, entry_timestamp: str, outcome_score: float, outcome_comment: Optional[str] = None):
        try:
            score = float(outcome_score)
            if math.isnan(score) or score < 0.0 or score > 1.0:
                raise ValueError("outcome_score must be in [0.0, 1.0]")
        except Exception as e:
            self.reflection_log.append(f"Feedback error (invalid score): {e}")
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
            self.reflection_log.append(f"Feedback: could not update memory CSV: {e}")

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
                        found = {
                            "timestamp": entry_timestamp,
                            "afd_metrics": {"harmony": float(row.iloc[0].get("coherence", 0.0)), "info_gradient": 0.0, "oscillation": 0.0, "potential": 0.0},
                            "coherence": float(row.iloc[0].get("coherence", 0.0))
                        }
                except Exception:
                    found = None
            if found is None:
                self.reflection_log.append(f"Feedback: could not locate explainability for timestamp {entry_timestamp}")
                return False

            feedback_record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "feedback_for": entry_timestamp,
                "outcome": score,
                "comment": outcome_comment,
            }
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
            self.reflection_log.append(f"Feedback: unexpected error: {e}")
            return False

        if updated:
            self.reflection_log.append(f"Feedback recorded for {entry_timestamp}: score={score}")
        return True

    def _learning_update(self, observed_coherence, outcome_score, metrics_dict):
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
            self.reflection_log.append(
                f"Learning update: error={error:.4f}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}"
            )
            return True
        except Exception as e:
            self.reflection_log.append(f"Learning update exception: {e}")
            return False

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
        lines.append(f"- coefficients: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}")
        return "\n".join(lines)

    def _build_renderer_prompt(self, neutral_input, afd_directives):
        return f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."

    #
    # Main respond method: returns final_text, coherence, afd_reflection, timestamp_iso
    #
    def respond(self, prompt, renderer_mode="afd", include_memory=True, seed: Optional[int] = None, reasoning_verbosity: int = 1):
        # Neutralize
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

        # Memory-driven tiny nudges (conservative)
        try:
            if memory_summary and memory_summary.get("avg_outcome") is not None:
                avg_hist = memory_summary.get("avg_outcome")
                if avg_hist is not None:
                    if avg_hist < 0.4:
                        self.alpha = self._clip(self.alpha + 0.01, self._coeff_min, self._coeff_max)
                        self.reflection_log.append(f"Memory-based tiny alpha bump: {self.alpha:.3f}")
                    elif avg_hist > 0.85:
                        self.gamma = self._clip(self.gamma + 0.01, self._coeff_min, self._coeff_max)
                        self.reflection_log.append(f"Memory-based tiny gamma bump: {self.gamma:.3f}")
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

        # Reasoning reflection (safe)
        reasoning_reflection_text = self.generate_reasoning_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary, verbosity=reasoning_verbosity, seed=seed)
        # Simulated thinking sequence
        simulated_thinking = self.generate_simulated_thinking_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        # Human-style and identity reflections
        human_reflection = self.generate_human_style_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)
        identity_reflection = self.generate_identity_reflection(neutral_prompt, metrics, coherence, memory_summary=memory_summary)

        # Rendering
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

        # Timestamp
        timestamp_iso = datetime.datetime.utcnow().isoformat()

        # Persist memory & explainability
        try:
            self.save_memory(prompt, neutral_prompt, final_text, coherence, timestamp_iso)
        except Exception:
            pass

        explainability = {
            "timestamp": timestamp_iso,
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

        afd_reflection = self.generate_afd_reflection(
            prompt, neutral_prompt, state, action, s_prime, interp_s, metrics, coherence, renderer_used
        )
        try:
            self.reflection_log.append(afd_reflection)
        except Exception:
            self.reflection_log = [afd_reflection]

        return final_text, coherence, afd_reflection, timestamp_iso
