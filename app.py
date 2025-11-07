# Minimal, resilient Streamlit app wrapper for AFDInfinityAMI
# - Shows import traceback if afd_ami_core fails to import
# - Always renders a UI (so page is not blank)
# - Provides a safe "stub" fallback so you can test the UI without models or OpenAI
import streamlit as st
import traceback
import os

st.set_page_config(page_title="AFD∞-AMI (resilient)", layout="centered")
st.title("AFD∞-AMI — Resilient UI")

# Attempt import but never let an import error make the entire page blank.
import_error = None
AFDInfinityAMI = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception as e:
    import_error = traceback.format_exc()

# Top-level diagnostics area (always shown)
with st.expander("Debug: import & environment (click to expand)", expanded=True):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.error("Error importing afd_ami_core.py — traceback below (no secrets will be printed):")
        st.code(import_error)
    key_present = bool((st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY present:", key_present)
    st.markdown("If OpenAI is available, prefer it to avoid heavy local HF model downloads on Streamlit Cloud.")

st.markdown("---")

st.header("Ask the assistant (neutralized + AFD-driven)")

prompt = st.text_area(
    "Enter your question or prompt. The assistant will first neutralize it, run the AFD math, and render a neutral explanation.",
    value="Explain the ethical considerations of using AI in hiring decisions.",
    height=140,
)

col1, col2 = st.columns(2)
with col1:
    prefer_openai = st.checkbox("Prefer OpenAI if available (avoid HF downloads)", value=True)
with col2:
    use_stub = st.checkbox("Use stub mode (no models or network calls)", value=False)

submit = st.button("Generate response")

def make_stub_ami():
    # Minimal stub object with the same respond signature
    class StubAMI:
        def __init__(self):
            self.use_openai = False
            self.reflection_log = ["Running in stub mode."]
            self.alpha = self.beta = 1.0
            self.gamma = self.delta = 0.5

        def respond(self, prompt_in):
            neutral = f"[stub-neutralized] {prompt_in}"
            coherence = 0.5
            response = f"Stubbed response based on neutral input: {neutral}"
            return response, coherence, self.get_latest_reflection()

        def get_latest_reflection(self):
            return self.reflection_log[-1] if self.reflection_log else "No reflections."

        def load_memory(self):
            import pandas as pd
            return pd.DataFrame()
    return StubAMI()

if submit:
    # Always create an ami-like object safely (either real or stub) and show reflection immediately
    if use_stub or AFDInfinityAMI is None:
        ami = make_stub_ami()
        st.info("Running in stub mode (no models will be loaded).")
    else:
        try:
            # Create AFDInfinityAMI; let the class decide about OpenAI vs HF based on env/secrets.
            ami = AFDInfinityAMI(use_openai=prefer_openai)
        except Exception as e:
            st.error("Failed to instantiate AFDInfinityAMI; falling back to stub mode.")
            st.exception(e)
            ami = make_stub_ami()

    # show immediate status
    try:
        st.write("Using OpenAI:", bool(getattr(ami, "use_openai", False)))
        st.write("Latest reflection:", ami.get_latest_reflection())
    except Exception:
        # never let introspection crash the UI
        st.write("AFD instance created (could not introspect reflection).")

    with st.spinner("Generating..."):
        try:
            response, coherence, reflection = ami.respond(prompt)
        except Exception as e:
            st.error("An error occurred while generating the response; falling back to stub output.")
            st.exception(e)
            ami = make_stub_ami()
            response, coherence, reflection = ami.respond(prompt)

        st.subheader("Response")
        st.write(response)

        st.markdown("**AFD outputs**")
        try:
            st.write(f"- Coherence score: {coherence:.6f}")
        except Exception:
            st.write(f"- Coherence score: {coherence}")
        st.write(f"- Latest reflection: {reflection}")

        if st.checkbox("Show stored neutralized prompt (if available)"):
            try:
                mem = ami.load_memory()
                if not mem.empty:
                    last = mem.iloc[-1]
                    neutral = last.get("neutral_prompt", None)
                    if neutral:
                        st.caption("Stored neutralized prompt (most recent):")
                        st.write(neutral)
                    else:
                        st.caption("No neutralized prompt stored in memory.")
                else:
                    st.caption("Memory is empty.")
            except Exception as e:
                st.warning("Could not read memory file.")
                st.exception(e)

st.markdown("---")
st.caption(
    "If the page is blank or hanging, use stub mode or ensure OPENAI_API_KEY is set in Streamlit secrets. "
    "Avoid enabling local HF when running on limited cloud instances because model downloads and Torch init can block or OOM."
)
