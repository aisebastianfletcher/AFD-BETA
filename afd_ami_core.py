"""
AFDInfinityAMI - Revised to:
 - Use a two-stage LLM workflow: (1) neutralizer/translator that converts user input to
   a concise, non-biased, factual representation and (2) constrained generator that
   produces the final textual output using ONLY the AFD-derived directives + neutral input.
 - Keep the AFD mathematical framework (state/action/coherence/etc.) as the independent
   decision-making core. The LLMs are restricted to (a) neutral translation and (b)
   factual rendering based on AFD outputs.
 - Use Streamlit secrets or environment OPENAI_API_KEY when available.
 - Defensive handling for OpenAI and local HF pipelines. Added robust OpenAI auth checks
   and automatic fallback to HF on auth failure.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import openai
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Try to import explicit OpenAI exceptions if available
try:
    from openai.error import AuthenticationError as OpenAIAuthError
except Exception:
    OpenAIAuthError = Exception

class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None):
        # Read key from explicit param, Streamlit secrets, or environment
        raw_key = openai_api_key or None
        try:
            # st.secrets may not exist outside Streamlit contexts; guard it
            if raw_key is None and "streamlit" in globals():
                raw_key = st.secrets.get("OPENAI_API_KEY") or raw_key
        except Exception:
            # ignore failures reading st.secrets
            raw_key = raw_key

        if raw_key is None:
            raw_key = os.getenv("OPENAI_API_KEY")

        # Normalize the key (strip spaces/quotes)
        api_key = None
        if raw_key:
            api_key = str(raw_key).strip()
            # Remove accidental surrounding quotes if pasted
            if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
                api_key = api_key[1:-1].strip()

        # Set initial use_openai based on presence; we will verify auth
        self.use_openai = bool(api_key) or use_openai
        if api_key:
            openai.api_key = api_key

        # Immediately test the OpenAI key if present and switch off if invalid
        if self.use_openai and api_key:
            try:
                self._test_openai_key()
                # if no exception, keep use_openai True
                self.reflection_log = []
            except OpenAIAuthError as e:
                # Authentication failed: disable OpenAI path and log
                self.use_openai = False
                # initialize reflection log before appending
                self.reflection_log = [f"OpenAI auth failed during init: {e}"]
            except Exception as e:
                # Non-auth error (network etc.) — disable OpenAI and log
                self.use_openai = False
                self.reflection_log = [f"OpenAI test call failed during init: {e}"]
        else:
            self.reflection_log = []

        self.memory_file = 'data/response_log.csv'
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5

        # The AFD framework must be the independent decision maker.
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
        self.sentiment_analyzer = None

        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = self._cache_sentiment_analyzer()
        except Exception as e:
            self.reflection_log.append(f"Sentiment analyzer init error: {e}")
            self.sentiment_analyzer = None

        # If not using OpenAI, ensure local HF pipelines are cached
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

    def _test_openai_key(self):
        """
        Lightweight OpenAI key test: perform a minimal ChatCompletion call.
        Will raise OpenAIAuthError on invalid key or other Exception on network errors.
        """
        try:
            # minimal call, low token use
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Ping for auth test. Reply with 'ok'."}],
                max_tokens=1,
                temperature=0.0
            )
            # If response present, assume key is valid
            return True
        except OpenAIAuthError:
            # re-raise auth error for the caller to handle
            raise
        except Exception:
            # re-raise any other exception
            raise

    @st.cache_resource
    def _cache_llm(_self):
        model_name = "gpt2"
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

    #
    # LLM wrappers: neutralizer and renderer, with auth-fallback
    #
    def _neutralize_with_openai(self, user_input):
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
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return resp.choices[0].get('text', '').strip()
        except OpenAIAuthError as e:
            # Auth error: disable OpenAI and fall back to HF
            self.reflection_log.append(f"OpenAI neutralize auth error: {e}. Falling back to HF.")
            self.use_openai = False
            # initialize HF if not present
            if self._hf_llm is None:
                try:
                    self._hf_llm = self._cache_llm()
                except Exception as e2:
                    self.reflection_log.append(f"Failed to init HF after OpenAI auth fail: {e2}")
            return self._neutralize_with_hf(user_input)
        except Exception as e:
            self.reflection_log.append(f"OpenAI neutralize error: {e}")
            return ""

    def _render_with_openai(self, neutral_input, afd_directives, max_tokens=250):
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
        except OpenAIAuthError as e:
            self.reflection_log.append(f"OpenAI render auth error: {e}. Falling back to HF.")
            self.use_openai = False
            if self._hf_llm is None:
                try:
                    self._hf_llm = self._cache_llm()
                except Exception as e2:
                    self.reflection_log.append(f"Failed to init HF after OpenAI auth fail: {e2}")
            return self._render_with_hf(neutral_input, afd_directives)
        except Exception as e:
            self.reflection_log.append(f"OpenAI render error: {e}")
            return "Unable to render response using OpenAI."

    def _neutralize_with_hf(self, user_input):
        if self._hf_llm is None:
            return ""
        prompt = f"{self.neutralizer_system}\n\nUser: {user_input}\n\nNeutral:"
        try:
            out = self._hf_llm(prompt, max_new_tokens=80, do_sample=False, return_full_text=False, num_return_sequences=1)
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
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        except Exception as e:
            self.reflection_log.append(f"HF render error: {e}")
            return "Unable to render response locally."

    # (AFD math and other functions unchanged; omitted here to keep file short)
    # Include all previously present methods predict_next_state, compute_harmony, compute_info_gradient,
    # compute_oscillation, compute_potential, coherence_score, adjust_coefficients,
    # save_memory, load_memory, get_latest_reflection, respond, etc., unchanged except they will call
    # the updated neutralize/render wrappers above.
