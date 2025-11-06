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
