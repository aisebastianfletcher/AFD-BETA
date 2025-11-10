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
        # These rules are deterministic and avoid any LLM generation.
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
        # Keep phrasing deterministic and safe
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
        lines.append(f"- afd_metrics: harmony={metrics.get('harmony'):.6f}, info_gradient={metrics.get('info_gradient'):.6f}, oscillation={metrics.get('oscillation'):.6f}, potential={metrics.get('potential'):.6f}")
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

        # 2) Sentiment/state
        try:
            if self.use_openai:
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

        # numeric state and action
        state = np.array([sentiment_score] * 5)
        action = np.array([1 if str(sentiment_label).upper().startswith("POS") else -1] * 5)

        # 3) AFD math (compute s_prime, interp_s, metrics, coherence)
        s_prime = self.predict_next_state(state, action)
        t = 0.5
        interp_s = state + t * (s_prime - state)
        h = self.compute_harmony(state, interp_s)
        i = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        coherence = float(self.alpha * h + self.beta * i - self.gamma * o + self.delta * phi)
        metrics = {"harmony": float(h), "info_gradient": float(i), "oscillation": float(o), "potential": float(phi)}

        # Adjust coefficients
        self.adjust_coefficients(coherence, metrics)

        # Build AFD directives (for audit)
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

        # 4) Choose renderer mode
        if renderer_mode == "afd":
            final_text = self._afd_renderer(neutral_prompt, afd_directives, metrics, coherence)
            renderer_used = "AFD (deterministic)"
        else:
            # LLM renderer fallback (existing behavior)
            if self.use_openai:
                final_text = self._render_with_openai(neutral_prompt, afd_directives)
                renderer_used = "OpenAI"
            else:
                final_text = self._render_with_hf(neutral_prompt, afd_directives)
                renderer_used = "HF"

        # 5) Save memory and store explainability snapshot
        try:
            self.save_memory(prompt, neutral_prompt, final_text, coherence)
        except Exception:
            pass

        explainability = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "original_prompt": prompt,
            "neutral_prompt": neutral_prompt,
            "renderer_used": renderer_used,
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

        # 6) Build explicit AFD reflection (this will be the returned reflection)
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
        # Append to reflection_log as well
        try:
            self.reflection_log.append(afd_reflection)
        except Exception:
            self.reflection_log = [afd_reflection]

        return final_text, coherence, afd_reflection
