import os
import numpy as np
import pandas as pd
import streamlit as st
import openai
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None):
        self.use_openai = use_openai
        self.memory_file = 'data/response_log.csv'
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5
        self.reflection_log = []

        # Instruction prefix to encourage longer, coherent answers
        self.instruction_prefix = (
            "You are AFD∞-AMI Ethical Assistant. Provide a thoughtful, clear, and complete answer in full sentences. "
            "When appropriate, briefly explain your reasoning or mention ethical considerations. Be concise but thorough."
        )

        if use_openai and openai_api_key:
            openai.api_key = openai_api_key
            self.llm = self._openai_generate
            self.sentiment_analyzer = self._cache_sentiment_analyzer()
        else:
            self.llm = self._cache_llm()
            self.sentiment_analyzer = self._cache_sentiment_analyzer()

        if not os.path.exists(self.memory_file):
            try:
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                pd.DataFrame(columns=['prompt', 'response', 'coherence']).to_csv(self.memory_file, index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"Error creating memory file: {e}")

    @st.cache_resource
    def _cache_llm(_self):
        """
        Load tokenizer and causal LM, set pad token to eos to avoid generation errors/warnings.
        Return a text-generation pipeline configured for the available device (GPU if available).
        """
        model_name = "gpt2"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            # Ensure tokenizer has a pad token (gpt2 does not by default)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Ensure model config pad token is set
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = model.config.eos_token_id
            device = 0 if torch.cuda.is_available() else -1
            return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            # Fallback to a simple pipeline call; keep the error in reflections
            _self.reflection_log.append(f"LLM cache/load error: {e}. Falling back to default pipeline construction.")
            return pipeline("text-generation", model=model_name)

    @st.cache_resource
    def _cache_sentiment_analyzer(_self):
        # Standard HF SST-2 fine-tuned model
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def _openai_generate(self, prompt, max_tokens=150, **kwargs):
        """
        Call OpenAI ChatCompletion with a system instruction to encourage coherent, full-sentence answers.
        Returns a list similar to HF pipeline output: [{'generated_text': content}]
        """
        try:
            messages = [
                {"role": "system", "content": self.instruction_prefix},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.8,
                top_p=0.95
            )
            # Robust extraction across client versions
            content = ""
            try:
                content = response.choices[0].message.content
            except Exception:
                try:
                    content = response.choices[0]['message']['content']
                except Exception:
                    content = response.choices[0].get('text', '')
            return [{"generated_text": content}]
        except Exception as e:
            self.reflection_log.append(f"OpenAI error: {e}. Falling back to local pipeline.")
            # Fallback to local LLM: call with compatible args
            return self.llm(prompt, max_new_tokens=150, do_sample=True, top_k=50, top_p=0.95, temperature=0.8, num_return_sequences=1, return_full_text=False)

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
        return score, {'harmony': h, 'info_gradient': i, 'oscillation': o, 'potential': phi}

    def adjust_coefficients(self, coherence, metrics):
        log = f"Coherence: {coherence:.2f}, Metrics: {metrics}"
        if coherence < 0.5:
            self.alpha += 0.1
            self.reflection_log.append(f"Increased alpha to {self.alpha:.2f} for better harmony. {log}")
        elif coherence > 0.9:
            self.gamma += 0.1
            self.reflection_log.append(f"Increased gamma to {self.gamma:.2f} to reduce oscillation. {log}")
        else:
            self.reflection_log.append(f"No adjustment needed. {log}")

    def save_memory(self, prompt, response, coherence):
        try:
            df = pd.read_csv(self.memory_file, encoding='utf-8-sig')
            new_row = pd.DataFrame({'prompt': [prompt], 'response': [response], 'coherence': [coherence]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.memory_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not save to CSV ({e}).")

    def load_memory(self):
        try:
            return pd.read_csv(self.memory_file, encoding='utf-8-sig')
        except Exception as e:
            print(f"Error loading memory file: {e}")
            return pd.DataFrame(columns=['prompt', 'response', 'coherence'])

    def get_latest_reflection(self):
        return self.reflection_log[-1] if self.reflection_log else "No reflections yet."

    def respond(self, prompt):
        memory = self.load_memory()
        past_response = ""
        if not memory.empty:
            try:
                similar = memory[memory['prompt'].str.contains(prompt[:20], case=False, na=False)]
                if not similar.empty:
                    best = similar.loc[similar['coherence'].idxmax()]
                    past_response = f"Past high-coherence response: {best['response']} (Coherence: {best['coherence']:.2f})"
            except Exception:
                # If the contains search fails for any reason, ignore it gracefully
                pass

        # Compose prompt with instruction prefix
        full_prompt = f"{self.instruction_prefix}\n\nUser: {prompt}"
        if past_response:
            full_prompt += f"\n\nContext: {past_response}"

        # Choose generation path
        if self.use_openai:
            # Let OpenAI generate up to 150 tokens by default (increase if you want longer answers)
            result = self.llm(full_prompt, max_tokens=150)
            raw_response = result[0].get('generated_text', '')
        else:
            # For transformers pipeline use modern args (max_new_tokens) and sampling parameters
            try:
                result = self.llm(
                    full_prompt,
                    max_new_tokens=150,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    num_return_sequences=1,
                    return_full_text=False
                )
                # pipeline returns list of dicts with 'generated_text'
                raw_response = result[0].get('generated_text', '') if isinstance(result[0], dict) else str(result[0])
            except TypeError:
                # Older transformers may not support max_new_tokens; fall back to max_length
                result = self.llm(
                    full_prompt,
                    max_length=200,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    num_return_sequences=1,
                    return_full_text=False
                )
                raw_response = result[0].get('generated_text', '') if isinstance(result[0], dict) else str(result[0])
            except Exception as e:
                self.reflection_log.append(f"Local generation error: {e}")
                raw_response = "I'm sorry, I couldn't generate a response right now."

        # Post-process: strip any repeated prompt text if returned
        if isinstance(raw_response, str) and full_prompt.strip() and raw_response.startswith(full_prompt.strip()):
            raw_response = raw_response[len(full_prompt):].strip()

        # Sentiment analysis (fallback to neutral on error)
        try:
            sentiment = self.sentiment_analyzer(raw_response)[0]
        except Exception as e:
            self.reflection_log.append(f"Sentiment analyzer error: {e}")
            sentiment = {'label': 'NEUTRAL', 'score': 0.5}

        state = np.array([sentiment.get('score', 0.5)] * 5)
        action = np.array([1 if sentiment.get('label', '').upper().startswith('POS') else -1] * 5)
        coherence, metrics = self.coherence_score(action, state)
        self.adjust_coefficients(coherence, metrics)
        self.save_memory(prompt, raw_response, coherence)
        return raw_response, coherence, self.get_latest_reflection()
