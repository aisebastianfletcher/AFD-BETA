# Streamlit UI for AFD∞-AMI with safe "reasoning as reflection", simulated thinking animation, and feedback loop.
import streamlit as st
import traceback
import os
import pandas as pd
import time

st.set_page_config(page_title="AFD∞-AMI (learning & reflection)", layout="centered")
st.title("AFD∞-AMI — Learning & Reflection")

import_error = None
AFDInfinityAMI = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception:
    import_error = traceback.format_exc()

with st.expander("Debug: import & environment (click to expand)", expanded=True):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.error("Error importing afd_ami_core.py — traceback below (no secrets will be printed):")
        st.code(import_error)
    key_present = bool((st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY present:", key_present)
    st.markdown("This UI supports AFD-mode answers (deterministic + auditable) and an optional LLM renderer. You can submit feedback on responses to help the agent learn conservatively.")

st.markdown("---")

st.header("Ask the assistant")
prompt = st.text_area("Enter your question or prompt:", value="Who am I talking to?", height=140)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    prefer_openai = st.checkbox("Prefer OpenAI if available (LLM rendering)", value=False)
with col2:
    use_stub = st.checkbox("Use stub mode (no models/network calls)", value=False)
with col3:
    renderer_choice = st.selectbox("Renderer mode", ["AFD renderer (deterministic)", "LLM renderer (OpenAI/HF)"], index=0)

temperature = st.slider("Temperature (controls AFD variability & LLM sampling)", min_value=0.0, max_value=3.0, value=0.0, step=0.05)
include_memory = st.checkbox("Include memory context (agent summarizes recent outcomes)", value=True)
memory_window = st.number_input("Memory window (entries to consider)", min_value=1, max_value=1000, value=20, step=1)

# Reasoning/reflection display controls
show_thinking_animation = st.checkbox("Show simulated 'thinking aloud' before answers", value=True)
show_reasoning_reflection = st.checkbox("Show reasoning-as-reflection before the answer", value=True)
reasoning_verbosity = st.selectbox("Reasoning verbosity", ["brief", "normal", "expanded"], index=1)
verbosity_setting = 0 if reasoning_verbosity == "brief" else (2 if reasoning_verbosity == "expanded" else 1)

# Feedback controls
st.markdown("Feedback (0.0 worst — 1.0 best). After generating a response, use the timestamp below to submit feedback and train the agent.")
feedback_score = st.slider("Feedback score", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
feedback_comment = st.text_input("Optional feedback comment", value="")

# Reproducibility seed (optional)
use_seed = st.checkbox("Use reproducibility seed", value=False)
seed = None
if use_seed:
    seed = int(st.number_input("Seed (integer)", min_value=0, max_value=2**31-1, value=0, step=1))

renderer_mode = "afd" if renderer_choice.startswith("AFD") else "llm"
submit = st.button("Generate response")

def make_stub_ami():
    class StubAMI:
        def __init__(self):
            self.use_openai = False
            self.reflection_log = ["stub"]
            self.alpha = self.beta = 1.0
            self.gamma = self.delta = 0.5

        def respond(self, prompt, renderer_mode="afd", include_memory=True, seed=None, reasoning_verbosity=1):
            neutral = f"[stub-neutral] {prompt}"
            coherence = 0.5
            ts = pd.Timestamp.utcnow().isoformat()
            response = f"Neutral input: {neutral}\n\nAnswer: This is stub response."
            return response, coherence, "AFD Reflection (stub)", ts

        def provide_feedback(self, entry_timestamp, outcome_score, outcome_comment=None):
            return True

        def load_memory(self):
            return pd.DataFrame()

        def get_last_explainability(self):
            return None

    return StubAMI()

# instantiate AMI
if use_stub or AFDInfinityAMI is None:
    ami = make_stub_ami()
else:
    try:
        ami = AFDInfinityAMI(use_openai=prefer_openai, temperature=temperature, memory_window=int(memory_window))
    except Exception as e:
        st.error("Failed to instantiate AFDInfinityAMI; using stub.")
        st.exception(e)
        ami = make_stub_ami()

# Show a compact status (do not reveal secrets)
try:
    st.write("Latest reflection available:", bool(ami.get_latest_reflection()))
except Exception:
    st.write("AFD instance created.")

if submit:
    with st.spinner("Generating..."):
        try:
            response, coherence, reflection, ts = ami.respond(prompt, renderer_mode=renderer_mode, include_memory=include_memory, seed=seed if use_seed else None, reasoning_verbosity=verbosity_setting)
        except Exception as e:
            st.error("Error generating response.")
            st.exception(e)
            response, coherence, reflection, ts = "Error", 0.0, "Error", pd.Timestamp.utcnow().isoformat()

    # Retrieve explainability snapshot
    explain = None
    try:
        explain = ami.get_last_explainability()
    except Exception:
        explain = None

    # Simulated thinking animation (short, safe)
    if show_thinking_animation and explain and explain.get("simulated_thinking"):
        thinking_lines = explain.get("simulated_thinking")
        placeholder = st.empty()
        for tline in thinking_lines:
            placeholder.markdown(f"*Thinking...* {tline}")
            # small visual delay
            time.sleep(0.45)
        placeholder.empty()

    # Reasoning-as-reflection before the answer (structured, safe)
    if show_reasoning_reflection and explain and explain.get("reasoning_reflection"):
        st.markdown("**Reasoning (safe, post-hoc reflection):**")
        st.code(explain.get("reasoning_reflection"))

    # Show the final answer
    st.markdown("---")
    st.subheader("Answer")
    st.write(response)

    # AFD reflection (authoritative)
    st.markdown("**AFD reflection (authoritative):**")
    st.code(reflection)

    # Timestamp for feedback reference
    st.markdown("Response timestamp (use this when submitting feedback):")
    st.text(ts)

    # Feedback submission for this response
    if st.button("Submit feedback for this response"):
        ok = ami.provide_feedback(ts, feedback_score, outcome_comment=feedback_comment)
        if ok:
            st.success(f"Feedback recorded (score={feedback_score}). Agent coefficients updated (conservatively).")
            recent = getattr(ami, "reflection_log", [])[-8:]
            st.write("Recent reflection log (last entries):")
            for r in recent:
                st.text(r)
        else:
            st.error("Failed to record feedback. Check Debug expander for details.")
            st.write(getattr(ami, "reflection_log", [])[-8:])

    # Hidden diagnostics block
    if st.checkbox("Show hidden reflections & diagnostics"):
        if explain and explain.get("human_reflection"):
            st.markdown("**Human-style reflection (post-hoc, safe):**")
            st.write(explain.get("human_reflection"))
        if explain and explain.get("identity_reflection"):
            st.markdown("**Identity reflection (who/what I am):**")
            st.write(explain.get("identity_reflection"))

        st.markdown("**Hidden diagnostics**")
        st.write("Using OpenAI:", bool(getattr(ami, "use_openai", False)))
        st.write("Recent reflection log:")
        for r in (getattr(ami, "reflection_log", [])[-12:]):
            st.text(r)

    # Show explainability details on demand
    if st.checkbox("Show detailed explainability (numeric states, prompts, metrics)"):
        if explain is None:
            st.write("No explainability snapshot available.")
        else:
            st.markdown("**Explainability snapshot**")
            st.write("Renderer used:", explain.get("renderer_used"))
            st.write("Neutral prompt:", explain.get("neutral_prompt"))
            st.write("Renderer prompt (exact):")
            st.code(explain.get("renderer_prompt"))
            st.write("Sentiment label / score:", explain.get("sentiment_label"), explain.get("sentiment_score"))
            st.write("State arrays (state, action, s_prime, interp_s):")
            st.json({
                "state": explain.get("state"),
                "action": explain.get("action"),
                "s_prime": explain.get("s_prime"),
                "interp_s": explain.get("interp_s"),
            })
            st.write("AFD metrics and coefficients:")
            st.json({"afd_metrics": explain.get("afd_metrics"), "coefficients": explain.get("coefficients")})
            st.write("Memory summary (recent interactions):")
            st.json(explain.get("memory_summary"))
            st.write("Final text (exact):")
            st.write(explain.get("final_text"))
            st.write("Timestamp (UTC):", explain.get("timestamp"))

# Memory viewing / download
if st.checkbox("Show memory (stored responses)"):
    try:
        mem = ami.load_memory()
        if not mem.empty:
            st.dataframe(mem.tail(100))
            csv = mem.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download memory CSV", csv, file_name="response_memory.csv", mime="text/csv")
        else:
            st.info("Memory is empty.")
    except Exception as e:
        st.warning("Could not read memory.")
        st.exception(e)

st.markdown("---")
st.caption(
    "Notes: Reasoning reflections are safe, post-hoc summaries derived from numeric AFD metrics and aggregated memory. "
    "They are not chain-of-thought. Use the feedback slider and timestamp to teach the agent conservatively over time."
)
