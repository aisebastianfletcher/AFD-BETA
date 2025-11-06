"""
AFDInfinityAMI - Revised to:
 - Use a two-stage LLM workflow: (1) neutralizer/translator that converts user input to
   a concise, non-biased, factual representation and (2) constrained generator that
   produces the final textual output using ONLY the AFD-derived directives + neutral input.
 - Keep the AFD mathematical framework (state/action/coherence/etc.) as the independent
   decision-making core. The LLMs are restricted to (a) neutral translation and (b)
   factual rendering based on AFD outputs.
 - Use Streamlit secrets or environment OPENAI_API_KEY when available.
 - Defensive handling for OpenAI and local HF pipelines.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import openai
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None):
        # Allow automatic detection if an API key is present if caller sets use_openai=False
        api_key = openai_api_key or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.use_openai = bool(api_key) or use_openai
        if self.use_openai and api_key:
            openai.api_key = api_key

        self.memory_file = 'data/response_log.csv'
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5
        self.reflection_log = []

        # The AFD framework must be the independent decision maker.
        # LLM usage is restricted to translating user text to a neutral form
        # and then rendering the final explanation strictly from AFD outputs
        # and the neutralized user input.
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

        # Local model caches (transformers pipelines)
        self._hf_llm = None
        self._hf_neutralizer = None
        self._hf_renderer = None
        self.sentiment_analyzer = None

        # Initialize resources
        try:
            self.sentiment_analyzer = self._cache_sentiment_analyzer()
        except Exception as e:
            # Keep going but log reflection
            self.reflection_log.append(f"Sentiment analyzer init error: {e}")
            self.sentiment_analyzer = None

        # If not using OpenAI, cache local HF pipelines on demand
        if not self.use_openai:
            try:
                self._hf_llm = self._cache_llm()
            except Exception as e:
                self.reflection_log.append(f"Local LLM init error: {e}")
                self._hf_llm = None

        # Ensure memory file exists
        if not os.path.exists(self.memory_file):
            try:
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                pd.DataFrame(columns=['prompt', 'neutral_prompt', 'response', 'coherence']).to_csv(self.memory_file, index=False, encoding='utf-8-sig')
            except Exception as e:
                self.reflection_log.append(f"Error creating memory file: {e}")

    #
    # Caching utilities for HF pipelines (local)
    #
    @st.cache_resource
    def _cache_llm(_self):
        # Provide a causal LM pipeline used for neutralizer and renderer when OpenAI not available.
        model_name = "gpt2"
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
    # LLM wrappers: neutralizer and renderer
    #
    def _neutralize_with_openai(self, user_input):
        # Use a deterministic low-temperature call to translate into neutral text
        try:
            messages = [
                {"role": "system", "content": self.neutralizer_system},
                {"role": "user", "content": user_input}
            ]
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=120
            )
            # robust extraction
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return resp.choices[0].get('text', '').strip()
        except Exception as e:
            self.reflection_log.append(f"OpenAI neutralize error: {e}")
            return ""

    def _render_with_openai(self, neutral_input, afd_directives, max_tokens=250):
        # Provide explicit instruction: use ONLY neutral_input and afd_directives to produce output.
        content = f"Neutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nProduce a single neutral explanation using only the above."
        try:
            messages = [
                {"role": "system", "content": self.renderer_system},
                {"role": "user", "content": content}
            ]
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return resp.choices[0].get('text', '').strip()
        except Exception as e:
            self.reflection_log.append(f"OpenAI render error: {e}")
            return "Unable to render response using OpenAI."

    def _neutralize_with_hf(self, user_input):
        # Use the local HF pipeline to generate a neutral short description.
        # We keep sampling low/temperature low to avoid hallucination.
        if self._hf_llm is None:
            return ""
        prompt = f"{self.neutralizer_system}\n\nUser: {user_input}\n\nNeutral:"
        try:
            out = self._hf_llm(prompt, max_new_tokens=80, do_sample=False, return_full_text=False, num_return_sequences=1)
            # pipeline returns [{'generated_text': '...'}]
            text = out[0].get('generated_text', '') if isinstance(out[0], dict) else str(out[0])
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF neutralize error: {e}")
            return ""

    def _render_with_hf(self, neutral_input, afd_directives):
        if self._hf_llm is None:
            return "Local model not available to render."
        prompt = f"{self.renderer_system}\n\nNeutral input:\n{neutral_input}\n\nAFD directives:\n{afd_directives}\n\nAnswer:"
        try:
            out = self._hf_llm(prompt, max_new_tokens=220, do_sample=True, top_k=40, top_p=0.9, temperature=0.6, return_full_text=False, num_return_sequences=1)
            text = out[0].get('generated_text', '') if isinstance(out[0], dict) else str(out[0])
            # strip any echo of the prompt if present
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF render error: {e}")
            return "Unable to render response locally."

    #
    # AFD framework (unchanged core computations)
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
        return float(score), {'harmony': float(h), 'info_gradient': float(i), 'oscillation': float(o), 'potential': float(phi)}

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
            df = pd.read_csv(self.memory_file, encoding='utf-8-sig')
            new_row = pd.DataFrame({
                'prompt': [prompt],
                'neutral_prompt': [neutral_prompt],
                'response': [response],
                'coherence': [coherence]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.memory_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not save to CSV ({e}).")

    def load_memory(self):
        try:
            return pd.read_csv(self.memory_file, encoding='utf-8-sig')
        except Exception as e:
            self.reflection_log.append(f"Error loading memory file: {e}")
            return pd.DataFrame(columns=['prompt', 'neutral_prompt', 'response', 'coherence'])

    def get_latest_reflection(self):
        return self.reflection_log[-1] if self.reflection_log else "No reflections yet."

    #
    # High-level respond() that enforces the neutral-then-AFD-then-render flow
    #
    def respond(self, prompt):
        # 1) Neutralize / translate the prompt
        neutral_prompt = ""
        if self.use_openai:
            neutral_prompt = self._neutralize_with_openai(prompt)
        else:
            neutral_prompt = self._neutralize_with_hf(prompt)

        if not neutral_prompt:
            # fallback: stripped user prompt but flagged as non-neutralized
            neutral_prompt = prompt.strip()

        # 2) Compute sentiment/state from neutral prompt to drive AFD math.
        try:
            if self.sentiment_analyzer:
                sent = self.sentiment_analyzer(neutral_prompt)[0]
                sentiment_score = float(sent.get('score', 0.5))
                sentiment_label = sent.get('label', 'NEUTRAL')
            else:
                sentiment_score = 0.5
                sentiment_label = 'NEUTRAL'
        except Exception as e:
            self.reflection_log.append(f"Sentiment error in respond: {e}")
            sentiment_score = 0.5
            sentiment_label = 'NEUTRAL'

        # Construct numeric state and action for AFD computations
        state = np.array([sentiment_score] * 5)
        action = np.array([1 if str(sentiment_label).upper().startswith('POS') else -1] * 5)

        # 3) Compute coherence and other AFD metrics (this is the independent core)
        coherence, metrics = self.coherence_score(action, state)
        # Adjust coefficients according to the AFD framework
        self.adjust_coefficients(coherence, metrics)

        # 4) Craft AFD directives (structured, numeric, non-opinionated) to send to renderer
        afd_directives = (
            f"AFD metrics:\n"
            f"- coherence: {coherence:.6f}\n"
            f"- harmony: {metrics.get('harmony'):.6f}\n"
            f"- info_gradient: {metrics.get('info_gradient'):.6f}\n"
            f"- oscillation: {metrics.get('oscillation'):.6f}\n"
            f"- potential: {metrics.get('potential'):.6f}\n"
            f"Coefficients: alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f}\n"
            "The renderer must use ONLY the neutral input and the numeric AFD metrics above to construct the response."
        )

        # 5) Render final text using OpenAI or HF pipeline but constrained to afd_directives + neutral_prompt
        if self.use_openai:
            final_text = self._render_with_openai(neutral_prompt, afd_directives)
        else:
            final_text = self._render_with_hf(neutral_prompt, afd_directives)

        # 6) Save to memory and return
        self.save_memory(prompt, neutral_prompt, final_text, coherence)
        return final_text, coherence, self.get_latest_reflection()
