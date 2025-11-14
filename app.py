# Streamlit UI — professional, single-pane continuous chat layout for AFD∞-AMI
# - Sidebar contains controls (renderer, temperature, memory, toggles)
# - Main area is a single chat pane with message bubbles and bottom input
# - Lightweight greeting/identity interception so basic greetings return a friendly identity reply
# - Cleaner hidden diagnostics and explainability access
# - Persists chat to data/chat_history.jsonl via ami.append_chat_entry
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
AFDInfinityAMI = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception:
    import_error = traceback.format_exc()

# Sidebar: tidy controls
with st.sidebar:
    st.header("Controls")
    st.write("Rendering & memory controls")
    prefer_openai = st.checkbox("Prefer OpenAI (for LLM rendering)", value=False)
    renderer_choice = st.selectbox("Renderer", ["AFD (deterministic)", "LLM (OpenAI/HF)"], index=0)
    renderer_mode = "afd" if renderer_choice.startswith("AFD") else "llm"

    temperature = st.slider("Temperature", 0.0, 3.0, 0.0, step=0.05)
    include_memory = st.checkbox("Include memory in reflections", value=True)
    memory_window = st.number_input("Memory window", min_value=1, max_value=1000, value=20, step=1)

    st.markdown("---")
    st.write("Chat & UI options")
    continuous_chat = st.checkbox("Enable continuous chat (session)", value=True)
    chat_history_length = st.number_input("Session history turns (for context)", min_value=0, max_value=50, value=6, step=1)
    show_thinking_animation = st.checkbox("Show simulated thinking animation", value=False)
    show_reasoning_reflection = st.checkbox("Show reasoning reflection (pre-answer)", value=False)
    reasoning_verbosity = st.selectbox("Reasoning verbosity", ["brief", "normal", "expanded"], index=1)
    verbosity_setting = 0 if reasoning_verbosity == "brief" else (2 if reasoning_verbosity == "expanded" else 1)

    st.markdown("---")
    st.write("Feedback")
    feedback_score_default = st.slider("Default feedback slider (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    st.caption("Use the feedback control shown with each message to teach the agent.")

    st.markdown("---")
    st.write("Session")
    use_seed = st.checkbox("Use reproducible seed", value=False)
    seed = None
    if use_seed:
        seed = int(st.number_input("Seed (integer)", min_value=0, max_value=2**31 - 1, value=0, step=1))

    st.markdown("---")
    if import_error:
        st.error("Error importing afd_ami_core.py (see Debug panel).")
    st.caption("Settings are saved only for the current session. For long-term changes edit code in the repository.")

# Top debug expander (collapsed)
with st.expander("Debug & environment", expanded=False):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.exception(import_error)
    st.write("Files persisted under ./data (memory, explainability, chat).")

# Instantiate agent
if import_error:
    # stub that won't raise further exceptions
    class _StubAMI:
        def __init__(self):
            self.use_openai = False
            self.reflection_log = ["stub"]
            self.alpha = self.beta = 1.0
            self.gamma = self.delta = 0.5

        def respond(self, prompt, renderer_mode="afd", include_memory=True, seed=None, reasoning_verbosity=1):
            ts = pd.Timestamp.utcnow().isoformat()
            return "Agent not available (import error).", 0.0, "AFD Reflection: stub", ts

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
        ami = AFDInfinityAMI(use_openai=prefer_openai, temperature=temperature, memory_window=int(memory_window))
    except Exception as e:
        st.error("Failed to instantiate AFDInfinityAMI; using fallback stub.")
        st.exception(e)
        class _StubAMI2:
            def __init__(self):
                self.use_openai = False
                self.reflection_log = ["stub2"]
                self.alpha = self.beta = 1.0
                self.gamma = self.delta = 0.5
            def respond(self, prompt, renderer_mode="afd", include_memory=True, seed=None, reasoning_verbosity=1):
                ts = pd.Timestamp.utcnow().isoformat()
                return "Agent instantiation failed.", 0.0, "AFD Reflection: stub", ts
            def append_chat_entry(self, role, message, timestamp_iso=None):
                return
            def provide_feedback(self, entry_timestamp, outcome_score, outcome_comment=None):
                return True
            def load_memory(self):
                return pd.DataFrame()
            def get_last_explainability(self):
                return None
        ami = _StubAMI2()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts {role, message, ts, explain}

# Layout: main middle column for chat, right column for explainability / actions
main_col, side_col = st.columns([3, 1])

with side_col:
    st.markdown("### Conversation tools")
    if st.button("Clear session chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")
    if st.button("Reload agent (recreate instance)"):
        try:
            ami = AFDInfinityAMI(use_openai=prefer_openai, temperature=temperature, memory_window=int(memory_window))
            st.success("Agent reloaded.")
        except Exception as e:
            st.error("Reload failed.")
            st.exception(e)

    st.markdown("---")
    st.markdown("### Last AFD metrics")
    last_explain = ami.get_last_explainability()
    if last_explain:
        st.write("Detected intent:", last_explain.get("detected_intent", "N/A"))
        afd_metrics = last_explain.get("afd_metrics", {})
        st.json({"coherence": last_explain.get("coherence"), "metrics": afd_metrics, "coefficients": last_explain.get("coefficients")})
    else:
        st.info("No explainability snapshot yet.")

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
            entries = ami.load_chat_history(1000) if hasattr(ami, "load_chat_history") else []
            data = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
            st.download_button("Download chat_history.jsonl", data, file_name="chat_history.jsonl", mime="application/json")
        except Exception as e:
            st.error("Could not prepare chat history.")
            st.exception(e)

# Helper: render chat messages in main column as bubbles
def render_chat(history):
    with main_col:
        st.markdown("## Conversation")
        if not history:
            st.info("No messages yet. Send a message to start the conversation.")
            return
        # Render each turn
        for turn in history[-200:]:
            role = turn.get("role", "user")
            msg = turn.get("message", "")
            ts = turn.get("ts", "")
            explain = turn.get("explain")
            if role == "user":
                cols = st.columns([1, 3])
                with cols[0]:
                    st.markdown(f"**You** ({ts})")
                with cols[1]:
                    st.markdown(f"> {msg}")
            else:
                cols = st.columns([1, 3])
                with cols[0]:
                    st.markdown(f"**Assistant** ({ts})")
                with cols[1]:
                    st.markdown(msg)
            # Optionally show a compact explainability pointer
            if explain and isinstance(explain, dict):
                with main_col.expander(f"Explainability — {ts}", expanded=False):
                    st.json({
                        "detected_intent": explain.get("detected_intent"),
                        "coherence": explain.get("coherence"),
                        "afd_metrics": explain.get("afd_metrics"),
                        "coefficients": explain.get("coefficients"),
                    })

# Render existing chat
render_chat(st.session_state.chat_history)

# Input area anchored below conversation
with main_col:
    st.markdown("---")
    if continuous_chat:
        user_input = st.text_input("Type a message and press Enter", key="chat_input")
    else:
        user_input = st.text_area("Enter a single-turn prompt", value="", height=140)

    send = st.button("Send")

# Lightweight client-side intent detection for greetings
def is_greeting(text: str) -> bool:
    if not text:
        return False
    text = text.strip().lower()
    return bool(re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b", text))

# When sending a message
if send and user_input and user_input.strip():
    # Build context if continuous chat
    context = ""
    if continuous_chat and chat_history_length > 0 and st.session_state.chat_history:
        # include last N turns (user+assistant) as simple context
        recent = st.session_state.chat_history[-chat_history_length:]
        context_parts = []
        for e in recent:
            role = e.get("role")
            msg = e.get("message")
            if role and msg:
                context_parts.append(f"{role.upper()}: {msg}")
        context = "\n".join(context_parts)

    combined_prompt = f"{context}\n\nUser: {user_input}" if context else user_input

    # Record user message in session & persistent chat log
    ts_user = pd.Timestamp.utcnow().isoformat()
    st.session_state.chat_history.append({"role": "user", "message": user_input, "ts": ts_user})
    try:
        ami.append_chat_entry("user", user_input, timestamp_iso=ts_user)
    except Exception:
        # ignore persistence failures; agent still works
        pass

    # If greeting, prefer identity reply via a simple targeted prompt to the agent.
    # This avoids accidental template matches (like "life" templates) for generic greetings.
    if is_greeting(user_input):
        prompt_for_agent = "Who am I talking to?"
    else:
        prompt_for_agent = combined_prompt

    # Call agent
    try:
        response, coherence, afd_reflection, response_ts = ami.respond(
            prompt_for_agent,
            renderer_mode=renderer_mode,
            include_memory=include_memory,
            seed=seed if use_seed else None,
            reasoning_verbosity=verbosity_setting,
        )
    except Exception as e:
        st.error("Agent error during respond()")
        st.exception(e)
        response, coherence, afd_reflection, response_ts = "Error generating response.", 0.0, "Error reflection", pd.Timestamp.utcnow().isoformat()

    # Retrieve explainability snapshot and sanitize final text
    explain = ami.get_last_explainability() or {}
    final_text = response if response and str(response).strip() else (explain.get("final_text") or "")

    # If we used greeting override, and the agent returned content about "Life" or other unrelated topic,
    # fall back to a concise identity message (defensive)
    if is_greeting(user_input):
        # prefer identity_reflection if available
        identity_text = None
        if explain and explain.get("identity_reflection"):
            identity_text = explain.get("identity_reflection") + "\n\nYou are speaking to AFD∞-AMI — an algorithmic assistant (not conscious)."
        # if final_text seems unrelated (contains "Definition: Life"), override
        if not identity_text and final_text and "Definition:" in final_text and "Life" in final_text:
            identity_text = "Hello — you are speaking to AFD∞-AMI, an algorithmic assistant. I process prompts with an auditable AFD pipeline and can learn from feedback."
        if identity_text:
            final_text = identity_text

    # Ensure there is a textual answer
    if not final_text or not str(final_text).strip():
        final_text = "No answer was generated. Check hidden diagnostics for details."

    # Persist assistant message in session and chat log
    st.session_state.chat_history.append({"role": "assistant", "message": final_text, "ts": response_ts, "explain": explain})
    try:
        ami.append_chat_entry("assistant", final_text, timestamp_iso=response_ts)
    except Exception:
        pass

    # Optional: show simulated thinking then reasoning reflection (before final answer)
    if show_thinking_animation and explain and explain.get("simulated_thinking"):
        thinking_lines = explain.get("simulated_thinking")
        placeholder = st.empty()
        for line in thinking_lines:
            placeholder.markdown(f"*Thinking...* {line}")
            time.sleep(0.14)
        placeholder.empty()

    if show_reasoning_reflection and explain and explain.get("reasoning_reflection"):
        with main_col:
            st.markdown("**Reasoning (safe, post-hoc reflection):**")
            st.code(explain.get("reasoning_reflection"))

    # Refresh chat display (render with new messages)
    render_chat(st.session_state.chat_history)

    # Show AFD reflection compact and allow feedback
    with main_col:
        st.markdown("---")
        st.markdown("**AFD reflection (authoritative)**")
        st.code(afd_reflection)

        st.markdown("Feedback for this response")
        score = st.slider("Score this reply (0.0 worst — 1.0 best)", min_value=0.0, max_value=1.0, value=float(feedback_score_default), step=0.01, key=f"fb_{response_ts}")
        comment = st.text_input("Optional comment", key=f"fb_comment_{response_ts}")
        if st.button("Submit feedback for this response", key=f"submit_fb_{response_ts}"):
            ok = ami.provide_feedback(response_ts, score, outcome_comment=comment)
            if ok:
                st.success("Feedback recorded; conservative learning update applied.")
            else:
                st.error("Failed to record feedback. Check hidden diagnostics.")

# Hidden diagnostics in an expander
with st.expander("Hidden reflections & diagnostics"):
    explain = ami.get_last_explainability()
    if explain:
        st.markdown("**Human-style reflection (post-hoc, safe)**")
        if explain.get("human_reflection"):
            st.write(explain.get("human_reflection"))
        st.markdown("**Identity reflection**")
        if explain.get("identity_reflection"):
            st.write(explain.get("identity_reflection"))
        st.markdown("**Reasoning reflection**")
        if explain.get("reasoning_reflection"):
            st.code(explain.get("reasoning_reflection"))
        st.markdown("**Detected intent**")
        st.write(explain.get("detected_intent", "N/A"))
    st.markdown("**Recent internal log (last entries)**")
    for line in getattr(ami, "reflection_log", [])[-30:]:
        st.text(line)

st.markdown("---")
st.caption(
    "Notes: This chat UI is designed for a professional, continuous conversation. "
    "The AFD renderer is deterministic and auditable by default. Toggle the LLM renderer to delegate rendering to OpenAI/HF. "
    "Reflections are post-hoc summaries derived from numeric metrics and are intentionally NOT chain-of-thought."
)
