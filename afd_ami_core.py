    def generate_human_style_reflection(self, neutral_prompt, metrics, coherence):
        """
        Produce a deterministic, first-person style reflection derived from AFD metrics.
        This is a simulated 'human-like' reflection (post-hoc summary), NOT raw chain-of-thought.
        """
        # Tone selection from coherence
        c = float(coherence)
        if c >= 0.8:
            confidence_phrase = "I feel confident about the information I will provide."
        elif c <= 0.35:
            confidence_phrase = "I sense uncertainty and will be cautious in my statements."
        else:
            confidence_phrase = "I have moderate confidence and will balance caution with clarity."

        # Harmony and info cues
        h = metrics.get("harmony", 0.0)
        ig = metrics.get("info_gradient", 0.0)
        o = metrics.get("oscillation", 0.0)

        harmony_note = (
            "There is notable agreement across internal signals."
            if h > 0.6
            else "There are some divergent signals to be aware of."
        )
        info_note = "I will surface supporting details where helpful." if ig > 0.5 else "I will keep the explanation concise."
        stability_note = "I will avoid speculation due to instability." if o > 0.2 else "The internal state is reasonably stable."

        # Identity / existence paragraph (safe, deterministic)
        identity = (
            "I am an algorithmic assistant: I do not possess consciousness or feelings, "
            "but I can describe my internal evaluation and use it to guide my response."
        )

        # Build the human-style reflection (first-person, non-sensitive)
        lines = []
        lines.append(identity)
        lines.append("")
        lines.append(f"- Neutral input (summary): {neutral_prompt}")
        lines.append(f"- AFD coherence: {c:.3f}. {confidence_phrase}")
        lines.append(f"- Notes: {harmony_note} {info_note} {stability_note}")
        lines.append("")
        lines.append(
            "This reflection summarizes how my internal AFD assessment will shape the answer: "
            "tone, caution level, and whether to include extra supporting detail."
        )

        # Keep it concise and deterministic
        return "\n".join(lines)
