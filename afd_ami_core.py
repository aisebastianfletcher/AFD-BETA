import numpy as np
import pandas as pd
from transformers import pipeline
import os
import openai
import streamlit as st

class AFDInfinityAMI:
    def __init__(self, use_openai=False, openai_api_key=None):
        self.use_openai = use_openai
        self.memory_file = 'data/response_log.csv'
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5
        self.reflection_log = []
        
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
    def _cache_llm(_self):  # Changed self to _self
        return pipeline("text-generation", model="gpt2")

    @st.cache_resource
    def _cache_sentiment_analyzer(_self):  # Changed self to _self
        return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    def _openai_generate(self, prompt, max_length=50, truncation=True, num_return_sequences=1, **kwargs):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=0.7,
                top_p=0.9
            )
            return [{"generated_text": response.choices[0].message.content}]
        except Exception as e:
            self.reflection_log.append(f"OpenAI error: {e}. Falling back to GPT-2.")
            return self.llm(prompt, max_length=max_length, truncation=truncation, num_return_sequences=num_return_sequences, **kwargs)

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
            similar = memory[memory['prompt'].str.contains(prompt[:20], case=False, na=False)]
            if not similar.empty:
                best = similar.loc[similar['coherence'].idxmax()]
                past_response = f"Past high-coherence response: {best['response']} (Coherence: {best['coherence']:.2f})"
        
        raw_response = self.llm(
            prompt + (f"\n\n{past_response}" if past_response else ""),
            max_length=50,
            truncation=True,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9
        )[0]['generated_text']
        
        sentiment = self.sentiment_analyzer(raw_response)[0]
        state = np.array([sentiment['score']] * 5)
        action = np.array([1 if sentiment['label'] == 'POSITIVE' else -1] * 5)
        coherence, metrics = self.coherence_score(action, state)
        self.adjust_coefficients(coherence, metrics)
        self.save_memory(prompt, raw_response, coherence)
        return raw_response, coherence, self.get_latest_reflection()
