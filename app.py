# Resilient Streamlit UI for AFD∞-AMI with memory and explainability display
import streamlit as st
import traceback
import os

st.set_page_config(page_title="AFD∞-AMI (resilient)", layout="centered")
st.title("AFD∞-AMI — Resilient UI (AFD renderer default)")

# Attempt import, keep UI alive if import fails
import_error = None
AFDInfinityAMI = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception:
    import_error = traceback.format_exc()

# Diagnostics expander (always visible)
with st.expander("Debug: import & environment (click to expand)", expanded=True):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.error("Error importing afd_ami_core.py — traceback below (no secrets will be printed):")
        st.code(import_error)
    key_present = bool((st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY present:", key_present)
    st.markdown(
        "Default mode uses the deterministic AFD renderer so responses are generated from the AFD math. "
        "If OpenAI is available and you choose the LLM renderer, the LLM will be used for rendering."
    )

st.markdown("---")

st.header("Ask the assistant (neutralized + AFD-driven)")
prompt = st.text_area(
    "Enter your question or prompt. The assistant will neutralize it, run the AFD math, and produce a response.",
    value="What is life?",
    height=160,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    prefer_openai = st.checkbox("Prefer OpenAI if available (avoid HF downloads)", value=True)
with col2:
    use_stub = st.checkbox("Use stub mode (no models or network calls)", value=False)
with col3:
    renderer_choice = st.selectbox(
        "Renderer mode",
        ["AFD renderer (deterministic)", "LLM renderer (OpenAI/HF)"],
        index=0,
    )

# temperature slider for variability
temperature = st.slider("Temperature (controls variability)", min_value=0.0, max_value=1.5, value=0.0, step=0.05)
renderer_mode = "afd" if renderer_choice.startswith("AFD") else "llm"

submit = st.button("Generate response")

def make_stub_ami():
    class StubAMI:
        def __init__(self):
            self.use_openai = False
            self.reflection_log = ["Running in stub mode."]
            self.alpha = self.beta = 1.0
            self.gamma = self.delta = 0.5

        def respond(self, prompt_in, renderer_mode="afd"):
            neutral = f"[stub-neutralized] {prompt_in}"
            coherence = 0.5
            response = (
                f"Neutral input: {neutral}\n\n"
                f"AFD Summary: coherence={coherence:.3f}\n\n"
                f"Answer: This is a stubbed AFD-style response based on the neutral input."
            )
            return response, coherence, self.get_latest_reflection()

        def get_latest_reflection(self):
            return self.reflection_log[-1] if self.reflection_log else "No reflections."

        def load_memory(self):
            import pandas as pd
            return pd.DataFrame()

        def get_last_explainability(self):
            return None

    return StubAMI()

if submit:
    if use_stub or AFDInfinityAMI is None:
        ami = make_stub_ami()
        st.info("Running in stub mode (no models will be loaded).")
    else:
        try:
            ami = AFDInfinityAMI(use_openai=prefer_openai, temperature=temperature)
        except Exception as e:
            st.error("Failed to instantiate AFDInfinityAMI; falling back to stub mode.")
            st.exception(e)
            ami = make_stub_ami()

    # display immediate status
    try:
        st.write("Using OpenAI:", bool(getattr(ami, "use_openai", False)))
        st.write("Latest reflection:", ami.get_latest_reflection())
    except Exception:
        st.write("AFD instance created (could not introspect reflection).")

    with st.spinner("Generating..."):
        try:
            response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode)
        except TypeError:
            try:
                response, coherence, reflection = ami.respond(prompt)
            except Exception as e:
                st.error("An error occurred while generating the response; falling back to stub output.")
                st.exception(e)
                ami = make_stub_ami()
                response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode)
        except Exception as e:
            st.error("An error occurred while generating the response; falling back to stub output.")
            st.exception(e)
            ami = make_stub_ami()
            response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode)

        # Show reflections (AFD reflection is authoritative)
        st.subheader("Reflection (AFD & Human‑style)")
        st.markdown("**AFD reflection (deterministic):**")
        st.code(reflection)

        # human-style reflection from explainability
        explain = None
        try:
            explain = ami.get_last_explainability()
        except Exception:
            explain = None

        if explain and explain.get("human_reflection"):
            st.markdown("**Human‑style reflection (simulated, post‑hoc):**")
            st.write(explain.get("human_reflection"))
        else:
            st.info("No human-style reflection available for this response.")

        st.markdown("---")
        st.subheader("Response")
        st.write(response)

        # AFD outputs
        st.markdown("**AFD outputs**")
        try:
            st.write(f"- Coherence score: {coherence:.6f}")
        except Exception:
            st.write(f"- Coherence score: {coherence}")
        st.write(f"- Renderer used: {explain.get('renderer_used') if explain else 'unknown'}")

        # detailed explainability dump
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
                st.write("Final text (exact):")
                st.write(explain.get("final_text"))
                st.write("Timestamp (UTC):", explain.get("timestamp"))

        # memory display / download
        if st.checkbox("Show memory (stored responses)"):
            try:
                mem = ami.load_memory()
                if not mem.empty:
                    st.dataframe(mem.tail(20))
                    if st.button("Download memory as CSV"):
                        st.download_button("Download CSV", mem.to_csv(index=False).encode("utf-8-sig"), file_name="response_memory.csv", mime="text/csv")
                else:
                    st.info("Memory is empty.")
            except Exception as e:
                st.warning("Could not read memory file.")
                st.exception(e)

st.markdown("---")
st.caption(
    "Notes: Answers default to the deterministic AFD renderer (no LLM). You can select the LLM renderer to compare outputs. "
    "Human-style reflections are simulated, post-hoc summaries derived from numeric AFD metrics (not chain-of-thought). Memory (CSV) and explainability (JSONL) are saved to the data/ directory."
)
