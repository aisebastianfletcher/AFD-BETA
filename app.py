# Streamlit UI with continuous chat plus the AFD single-turn mode.
# Continuous chat is maintained in session_state and persisted to data/chat_history.jsonl via ami.append_chat_entry.
import streamlit as st
import traceback
import os
import pandas as pd
import time

st.set_page_config(page_title="AFD∞-AMI (chat + learning)", layout="wide")
st.title("AFD∞-AMI — Continuous Chat + Learning")

import_error = None
AFDInfinityAMI = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception:
    import_error = traceback.format_exc()

with st.expander("Debug: import & environment (click to expand)", expanded=False):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.error("Error importing afd_ami_core.py — traceback below (no secrets will be printed):")
        st.code(import_error)
    key_present = bool((st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY present:", key_present)
    st.markdown("This UI supports continuous chat (session-based), AFD deterministic rendering, optional LLM rendering, and feedback for learning.")

st.markdown("---")

# Controls
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    prefer_openai = st.checkbox("Prefer OpenAI if available (LLM rendering)", value=False)
with col2:
    use_stub = st.checkbox("Use stub mode (no models/network calls)", value=False)
with col3:
    renderer_choice = st.selectbox("Renderer mode", ["AFD renderer (deterministic)", "LLM renderer (OpenAI/HF)"], index=0)

temperature = st.slider("Temperature (affects AFD variability & LLM sampling)", min_value=0.0, max_value=3.0, value=0.0, step=0.05)
include_memory = st.checkbox("Include memory context (agent summarizes recent outcomes)", value=True)
memory_window = st.number_input("Memory window (entries to consider)", min_value=1, max_value=1000, value=20, step=1)

# Chat controls
continuous_chat = st.checkbox("Enable continuous chat (session)", value=True)
chat_history_length = st.number_input("Chat history length to include in context", min_value=0, max_value=200, value=6, step=1)

# Reasoning/reflection display controls
show_thinking_animation = st.checkbox("Show simulated thinking animation", value=True)
show_reasoning_reflection = st.checkbox("Show reasoning-as-reflection before the answer", value=True)
reasoning_verbosity = st.selectbox("Reasoning verbosity", ["brief", "normal", "expanded"], index=1)
verbosity_setting = 0 if reasoning_verbosity == "brief" else (2 if reasoning_verbosity == "expanded" else 1)

# Feedback controls
st.markdown("Feedback (0.0 worst — 1.0 best). After generating a response, use the timestamp to submit feedback and train the agent.")
feedback_score = st.slider("Feedback score", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
feedback_comment = st.text_input("Optional feedback comment", value="")

# Reproducibility seed (optional)
use_seed = st.checkbox("Use reproducibility seed", value=False)
seed = None
if use_seed:
    seed = int(st.number_input("Seed (integer)", min_value=0, max_value=2**31-1, value=0, step=1))

renderer_mode = "afd" if renderer_choice.startswith("AFD") else "llm"

# instantiate AMI
if use_stub or AFDInfinityAMI is None:
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
        def append_chat_entry(self, role, message, timestamp_iso=None):
            return
    ami = StubAMI()
else:
    try:
        ami = AFDInfinityAMI(use_openai=prefer_openai, temperature=temperature, memory_window=int(memory_window))
    except Exception as e:
        st.error("Failed to instantiate AFDInfinityAMI; using stub.")
        st.exception(e)
        ami = StubAMI()  # type: ignore

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"role","message","ts"}

# Chat input area
st.subheader("Chat")
if continuous_chat:
    user_input = st.text_input("Your message", key="chat_input")
else:
    user_input = st.text_area("Your single-turn prompt", value="", height=120)

send = st.button("Send")

# Reset chat
if st.button("Reset chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared.")

def build_context_from_session(n: int):
    """Build a short context string from session chat history (last n turns)."""
    hist = st.session_state.get("chat_history", [])[-n:]
    parts = []
    for e in hist:
        role = e.get("role")
        msg = e.get("message")
        if role and msg:
            parts.append(f"{role.upper()}: {msg}")
    return "\n".join(parts)

if send and user_input:
    # Build prompt with context if continuous_chat
    context = ""
    if continuous_chat and chat_history_length > 0 and len(st.session_state.chat_history) > 0:
        context = build_context_from_session(chat_history_length)
    if context:
        combined_prompt = f"Conversation context:\n{context}\n\nUser: {user_input}"
    else:
        combined_prompt = user_input

    # record user message in session and persistent chat log
    ts = pd.Timestamp.utcnow().isoformat()
    st.session_state.chat_history.append({"role": "user", "message": user_input, "ts": ts})
    try:
        ami.append_chat_entry("user", user_input, timestamp_iso=ts)
    except Exception:
        pass

    # Call the agent
    try:
        response, coherence, reflection, response_ts = ami.respond(combined_prompt, renderer_mode=renderer_mode, include_memory=include_memory, seed=seed if use_seed else None, reasoning_verbosity=verbosity_setting)
    except Exception as e:
        st.error("Error generating response.")
        st.exception(e)
        response, coherence, reflection, response_ts = "Error generating response.", 0.0, "Error", pd.Timestamp.utcnow().isoformat()

    # If response empty, fallback to explain snapshot
    explain = ami.get_last_explainability()
    final_answer_text = response if response and str(response).strip() else (explain.get("final_text") if explain and explain.get("final_text") else None)
    if not final_answer_text:
        ami.reflection_log.append("DEBUG: No final_text returned by renderer; falling back to placeholder.")
        final_answer_text = "No answer was generated. Check hidden diagnostics."

    # record assistant message
    st.session_state.chat_history.append({"role": "assistant", "message": final_answer_text, "ts": response_ts})
    try:
        ami.append_chat_entry("assistant", final_answer_text, timestamp_iso=response_ts)
    except Exception:
        pass

    # thinking animation
    if show_thinking_animation and explain and explain.get("simulated_thinking"):
        thinking_lines = explain.get("simulated_thinking")
        placeholder = st.empty()
        for tline in thinking_lines:
            placeholder.markdown(f"*Thinking...* {tline}")
            time.sleep(0.18)
        placeholder.empty()

    # reasoning reflection
    if show_reasoning_reflection and explain and explain.get("reasoning_reflection"):
        st.markdown("**Reasoning (safe, post-hoc reflection):**")
        st.code(explain.get("reasoning_reflection"))

    # show assistant reply
    st.markdown("---")
    st.markdown("**Assistant**")
    st.write(final_answer_text)

    st.markdown("**AFD reflection (authoritative):**")
    st.code(reflection)

    st.markdown("Response timestamp (use when submitting feedback):")
    st.text(response_ts)

    # allow submitting feedback for this response
    if st.button("Submit feedback for this response"):
        ok = ami.provide_feedback(response_ts, feedback_score, outcome_comment=feedback_comment)
        if ok:
            st.success(f"Feedback recorded (score={feedback_score}). Agent coefficients updated (conservatively).")
            recent = getattr(ami, "reflection_log", [])[-8:]
            for r in recent:
                st.text(r)
        else:
            st.error("Failed to record feedback. See hidden diagnostics for details.")
            for r in getattr(ami, "reflection_log", [])[-8:]:
                st.text(r)

# Display chat history
st.subheader("Conversation")
chat_df = st.session_state.get("chat_history", [])
if chat_df:
    # render chat in two columns (role / message) with timestamps
    for turn in chat_df[-200:]:
        role = turn.get("role")
        msg = turn.get("message")
        ts = turn.get("ts")
        if role == "user":
            st.markdown(f"**You** ({ts}): {msg}")
        else:
            st.markdown(f"**Assistant** ({ts}): {msg}")
else:
    st.info("No chat history yet. Send a message to start the conversation.")

# Hidden diagnostics & explainability (on demand)
if st.checkbox("Show hidden reflections & diagnostics"):
    explain = ami.get_last_explainability()
    if explain and explain.get("human_reflection"):
        st.markdown("**Human-style reflection (post-hoc, safe):**")
        st.write(explain.get("human_reflection"))
    if explain and explain.get("identity_reflection"):
        st.markdown("**Identity reflection (who/what I am):**")
        st.write(explain.get("identity_reflection"))
    st.markdown("**Hidden diagnostics**")
    st.write("Using OpenAI:", bool(getattr(ami, "use_openai", False)))
    st.write("Detected intent (last explain):", explain.get("detected_intent") if explain else "N/A")
    st.write("Recent reflection log:")
    for r in (getattr(ami, "reflection_log", [])[-20:]):
        st.text(r)

# Show/Download memory and chat files
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

if st.checkbox("Show persisted chat history (audit)"):
    try:
        chat_entries = ami.load_chat_history(200)
        if chat_entries:
            st.json(chat_entries[-200:])
            # download
            data = "\n".join(json.dumps(e, ensure_ascii=False) for e in chat_entries)
            st.download_button("Download chat history", data, file_name="chat_history.jsonl", mime="application/json")
        else:
            st.info("No persisted chat entries.")
    except Exception as e:
        st.warning("Could not load chat history.")
        st.exception(e)

st.markdown("---")
st.caption(
    "Notes: Continuous chat is session-based and persisted to data/chat_history.jsonl for auditing. "
    "AFD reflections and 'reasoning-as-reflection' are safe, post-hoc summaries derived from numeric metrics and aggregated memory (not chain-of-thought)."
)
