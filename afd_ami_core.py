    def _rule_neutralize(self, user_input: str) -> str:
        """
        Lightweight rule-based neutralizer used WHEN renderer_mode == 'afd'.
        This avoids any LLM calls and produces a short neutral summary suitable
        for the deterministic AFD renderer.
        """
        s = (user_input or "").strip()
        if not s:
            return ""

        low = s.lower().strip()

        # common question forms -> short neutral summary
        # "what is X?" -> "definition of X"
        m = None
        # strip trailing punctuation
        core = s.rstrip(" ?!.")
        # patterns
        # what is/are/was/does/do/should ...
        m = re.match(r"^(what|who|when|where|why|how)\s+(.+)", low)
        if m:
            qword = m.group(1)
            rest = core[len(qword):].strip()
            # handle "what is X" and "who are you" specially
            if qword == "what":
                return f"ask for a definition or explanation of: {rest}"
            if qword == "who" and "you" in rest:
                return "ask for the agent's identity"
            return f"ask for information about: {rest}"

        # imperative or statement -> short paraphrase
        # e.g., "Explain X" -> "explain X"
        m2 = re.match(r"^(explain|describe|summarize|compare|list)\s+(.+)", low)
        if m2:
            verb = m2.group(1)
            target = m2.group(2).rstrip(" ?!.")
            return f"{verb} {target}"

        # fallback: return a concise neutral echo trimmed to 120 chars
        trimmed = core.strip()
        if len(trimmed) > 120:
            trimmed = trimmed[:117].rstrip() + "…"
        return f"neutral summary: {trimmed}"
