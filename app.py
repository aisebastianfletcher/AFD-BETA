# app.py
"""
Streamlit UI: professional continuous chat layout for AFD∞-AMI
- Sidebar holds controls (renderer, temperature, memory options)
- Main pane displays linear conversation with per-assistant explainability expanders
- Chat stored in session_state and persisted to data/chat_history.jsonl
"""
import streamlit as st
import traceback
import os
import pandas as pd
import time
import re
import json

st.set_page_config(page_title="AFD∞-AMI — Chat", layout="wide")
st.title("AFD∞-AMI — Continuous Chat (AFD-driven, auditable)")

# Import agent
import_error = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception as e:
    import_error = traceback.format_exc()
    AFDInfinityAMI = None

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    prefer_openai = st.checkbox("Prefer OpenAI (LLM rendering)", value=False)
    renderer_choice = st.selectbox("Renderer", ["AFD (deterministic)", "LLM (OpenAI/HF)"], index=0)
    renderer_mode = "afd" if renderer_choice.startswith("AFD") else "llm"

    temperature = st.slider("Temperature", 0.0, 3.0, 0.0, step=0.05)
    include_memory = st.checkbox("Include memory in reflections", value=True)
    memory_window = st.number_input("Memory window", min_value=1, max_value=1000, value=20, step=1)

    st.markdown("---")
    st.header("Chat UI")
    continuous_chat = st.checkbox("Enable continuous chat", value=True)
    chat_history_length = st.number_input("Session history turns (for context)", min_value=0, max_value=50, value=6, step=1)

    st.markdown("---")
    st.header("Reflections")
    show_thinking_animation = st.checkbox("Show thinking animation", value=False)
    show_reasoning_reflection = st.checkbox("Show reasoning reflection pre-answer", value=False)
    reasoning_verbosity = st.selectbox("Reasoning verbosity", ["brief", "normal", "expanded"], index=1)
    verbosity_setting = 0 if reasoning_verbosity == "brief" else (2 if reasoning_verbosity == "expanded" else 1)

    st.markdown("---")
    st.header("Feedback & Session")
    feedback_score_default = st.slider("Default feedback", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    use_seed = st.checkbox("Use reproducible seed", value=False)
    seed = None
    if use_seed:
        seed = int(st.number_input("Seed (integer)", min_value=0, max_value=2**31 - 1, value=0, step=1))

    st.markdown("---")
    if import_error:
        st.error("Error importing afd_ami_core.py (see Debug).")
    st.caption("Settings are session-scoped. To persist change code in the repo.")

# Debug expander
with st.expander("Debug", expanded=False):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.exception(import_error)
    st.write("Data persisted in ./data (memory, explainability, chat_history.jsonl).")

# Instantiate agent (or stub)
if import_error or AFDInfinityAMI is None:
    class _StubAMI:
        def __init__(self):
            self.use_openai = False
            self.reflection_log = ["stub"]
            self.alpha = self.beta = 1.0
            self.gamma = self.delta = 0.5
        def respond(self, prompt, renderer_mode="afd", include_memory=True, seed=None, reasoning_verbosity=1, chat_context=None):
            ts = pd.Timestamp.utcnow().isoformat()
            return "Agent not available (import error).", 0.0, "AFD Reflection (stub)", ts
        def append_chat_entry(self, role, message, timestamp_iso=None):
            return
        def provide_feedback(self, entry_timestamp, outcome_score, outcome_comment=None):
            return True
        def load_memory(self):
            return pd.DataFrame()
        def get_last_explainability(self):
            return None
    ami = _StubAMI()
else:
    try:
        ami = AFDInfinityAMI(use_openai=prefer_openai, openai_api_key=None, temperature=temperature, memory_window=int(memory_window))
    except Exception as e:
        st.error("Agent instantiation failed; using stub.")
        st.exception(e)
        class _StubAMI2:
            def __init__(self):
                self.use_openai = False
                self.reflection_log = ["stub2"]
                self.alpha = self.beta = 1.0
                self.gamma = self.delta = 0.5
            def respond(self, prompt, renderer_mode="afd", include_memory=True, seed=None, reasoning_verbosity=1, chat_context=None):
                ts = pd.Timestamp.utcnow().isoformat()
                return "Agent instantiation failed.", 0.0, "AFD Reflection (stub)", ts
            def append_chat_entry(self, role, message, timestamp_iso=None):
                return
            def provide_feedback(self, entry_timestamp, outcome_score, outcome_comment=None):
                return True
            def load_memory(self):
                return pd.DataFrame()
            def get_last_explainability(self):
                return None
        ami = _StubAMI2()

# Session chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role, message, ts, explain}

# Helper functions
def is_greeting(text: str) -> bool:
    if not text:
        return False
    return bool(re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b", text.strip().lower()))

def build_context_from_session(n: int):
    hist = st.session_state.get("chat_history", [])[-n:]
    parts = []
    for e in hist:
        role = e.get("role")
        msg = e.get("message")
        if role and msg:
            parts.append(f"{role.upper()}: {msg}")
    return "\n".join(parts)

# Chat rendering
def render_chat(history):
    st.markdown("### Conversation")
    if not history:
        st.info("No messages yet — send a message to start.")
        return
    for turn in history[-200:]:
        role = turn.get("role", "user")
        msg = turn.get("message", "")
        ts = turn.get("ts", "")
        explain = turn.get("explain")
        if role == "user":
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"**You**")
            with cols[1]:
                st.markdown(f"> {msg}")
        else:
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"**Assistant**")
            with cols[1]:
                st.markdown(msg)
            if explain and isinstance(explain, dict):
                with cols[1].expander("Show assistant reflections & metrics", expanded=False):
                    st.markdown("**Detected intent:** " + str(explain.get("detected_intent", "N/A")))
                    if explain.get("reasoning_reflection"):
                        st.markdown("**Reasoning (post-hoc):**")
                        st.code(explain.get("reasoning_reflection"))
                    if explain.get("human_reflection"):
                        st.markdown("**Human-style reflection:**")
                        st.write(explain.get("human_reflection"))
                    if explain.get("identity_reflection"):
                        st.markdown("**Identity reflection:**")
                        st.write(explain.get("identity_reflection"))
                    st.markdown("**AFD metrics & coefficients**")
                    st.json({"coherence": explain.get("coherence"), "afd_metrics": explain.get("afd_metrics"), "coefficients": explain.get("coefficients")})

# initial render
render_chat(st.session_state.chat_history)

# Input area
st.markdown("---")
if continuous_chat:
    user_input = st.text_input("Message", key="chat_input")
else:
    user_input = st.text_area("Single-turn prompt", value="", height=140)
send = st.button("Send")

# Tools & downloads
with st.sidebar:
    st.markdown("---")
    if st.button("Clear session chat"):
        st.session_state.chat_history = []
        st.success("Session chat cleared.")
    if st.button("Reload agent"):
        try:
            ami = AFDInfinityAMI(use_openai=prefer_openai, openai_api_key=None, temperature=temperature, memory_window=int(memory_window))
            st.success("Agent reloaded.")
        except Exception as e:
            st.error("Reload failed.")
            st.exception(e)
    st.markdown("---")
    st.markdown("### Persisted files")
    if st.button("Download memory CSV"):
        try:
            df = ami.load_memory()
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download memory.csv", csv, file_name="response_memory.csv", mime="text/csv")
        except Exception as e:
            st.error("Could not prepare memory CSV.")
            st.exception(e)
    if st.button("Download chat history (jsonl)"):
        try:
            entries = ami.load_chat_history(1000)
            data = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
            st.download_button("Download chat_history.jsonl", data, file_name="chat_history.jsonl", mime="application/json")
        except Exception as e:
            st.error("Could not prepare chat history.")
            st.exception(e)

# Send handling
if send and user_input and user_input.strip():
    ts_user = pd.Timestamp.utcnow().isoformat()
    st.session_state.chat_history.append({"role": "user", "message": user_input, "ts": ts_user})
    try:
        ami.append_chat_entry("user", user_input, timestamp_iso=ts_user)
    except Exception:
        pass

    context = ""
    if continuous_chat and chat_history_length > 0 and st.session_state.chat_history:
        context = build_context_from_session(chat_history_length)

    # greeting shortcut
    if is_greeting(user_input):
        prompt_for_agent = "Who am I talking to?"
    else:
        prompt_for_agent = user_input

    try:
        response, coherence, afd_reflection, response_ts = ami.respond(
            prompt_for_agent,
            renderer_mode=renderer_mode,
            include_memory=include_memory,
            seed=seed if use_seed else None,
            reasoning_verbosity=verbosity_setting,
            chat_context=context,
        )
    except Exception as e:
        st.error("Agent error during respond()")
        st.exception(e)
        response, coherence, afd_reflection, response_ts = "Error generating response.", 0.0, "Error", pd.Timestamp.utcnow().isoformat()

    explain = ami.get_last_explainability() or {}
    final_text = response if response and str(response).strip() else (explain.get("final_text") or "")

    # greeting defense
    if is_greeting(user_input):
        id_text = None
        if explain and explain.get("identity_reflection"):
            id_text = explain.get("identity_reflection") + "\n\nYou are speaking to AFD∞-AMI — an algorithmic assistant (not conscious)."
        if not id_text and final_text and "Definition:" in final_text and "Life" in final_text:
            id_text = "Hello — you are speaking to AFD∞-AMI, an algorithmic assistant. I process prompts with an auditable AFD pipeline and can learn from feedback."
        if id_text:
            final_text = id_text

    if not final_text or not str(final_text).strip():
        final_text = "No answer was generated. Check hidden diagnostics."

    # record assistant message
    st.session_state.chat_history.append({"role": "assistant", "message": final_text, "ts": response_ts, "explain": explain})
    try:
        ami.append_chat_entry("assistant", final_text, timestamp_iso=response_ts)
    except Exception:
        pass

    # optional thinking animation
    if show_thinking_animation and explain and explain.get("simulated_thinking"):
        pl = st.empty()
        for tline in explain.get("simulated_thinking"):
            pl.markdown(f"*Thinking...* {tline}")
            time.sleep(0.14)
        pl.empty()

    if show_reasoning_reflection and explain and explain.get("reasoning_reflection"):
        st.markdown("**Reasoning (safe, post-hoc):**")
        st.code(explain.get("reasoning_reflection"))

    render_chat(st.session_state.chat_history)

    # show afd reflection & feedback controls
    st.markdown("---")
    st.markdown("**AFD reflection (authoritative)**")
    st.code(afd_reflection)

    st.markdown("Feedback for this response")
    score = st.slider("Score this reply", min_value=0.0, max_value=1.0, value=float(feedback_score_default), step=0.01, key=f"fb_{response_ts}")
    comment = st.text_input("Optional comment", key=f"fb_comment_{response_ts}")
    if st.button("Submit feedback for this response", key=f"submit_fb_{response_ts}"):
        ok = ami.provide_feedback(response_ts, score, outcome_comment=comment)
        if ok:
            st.success("Feedback recorded; conservative learning update applied.")
        else:
            st.error("Failed to record feedback. See diagnostics.")

# Hidden diagnostics
with st.expander("Hidden reflections & diagnostics"):
    explain = ami.get_last_explainability()
    if explain:
        if explain.get("human_reflection"):
            st.markdown("**Human-style reflection (post-hoc, safe)**")
            st.write(explain.get("human_reflection"))
        if explain.get("identity_reflection"):
            st.markdown("**Identity reflection**")
            st.write(explain.get("identity_reflection"))
        if explain.get("reasoning_reflection"):
            st.markdown("**Reasoning reflection**")
            st.code(explain.get("reasoning_reflection"))
        st.markdown("**Detected intent**")
        st.write(explain.get("detected_intent", "N/A"))
    st.markdown("**Recent internal log**")
    for line in getattr(ami, "reflection_log", [])[-50:]:
        st.text(line)

st.markdown("---")
st.caption(
    "Notes: This UI sends only the current user message to the agent; conversation context is kept for explainability only to avoid echoing. "
    "Reflections are safe, post-hoc summaries derived from numeric AFD metrics and aggregated memory; they are not chain-of-thought."
)
