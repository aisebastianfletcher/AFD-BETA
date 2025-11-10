# Resilient Streamlit UI for AFD∞-AMI
# - Shows import traceback if afd_ami_core fails to import
# - Default renderer is the deterministic AFD renderer (no LLM) so answers come from AFD math
# - Allows opting into LLM renderer (OpenAI/HF) for comparison
# - Provides stub mode for testing without models or network
import streamlit as st
import traceback
import os

st.set_page_config(page_title="AFD∞-AMI (resilient)", layout="centered")
st.title("AFD∞-AMI — Resilient UI (AFD renderer default)")

# Attempt import but never let an import error make the entire page blank.
import_error = None
AFDInfinityAMI = None
try:
    from afd_ami_core import AFDInfinityAMI  # type: ignore
except Exception:
    import_error = traceback.format_exc()

# Top-level diagnostics area (always shown)
with st.expander("Debug: import & environment (click to expand)", expanded=True):
    st.write("Import OK:", import_error is None)
    if import_error:
        st.error("Error importing afd_ami_core.py — traceback below (no secrets will be printed):")
        st.code(import_error)
    key_present = bool((st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY present:", key_present)
    st.markdown(
        "Notes: Default mode uses the deterministic AFD renderer so responses are generated from the AFD math. "
        "If OpenAI is available and you choose the LLM renderer, the LLM will be used for rendering."
    )

st.markdown("---")

st.header("Ask the assistant (neutralized + AFD-driven)")
prompt = st.text_area(
    "Enter your question or prompt. The assistant will first neutralize it, run the AFD math, and produce a response.",
    value="Explain the ethical considerations of using AI in hiring decisions.",
    height=140,
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
            # deterministic stub AFD-style text
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
    # Always create an ami-like object safely (either real or stub) and show reflection immediately
    if use_stub or AFDInfinityAMI is None:
        ami = make_stub_ami()
        st.info("Running in stub mode (no models will be loaded).")
    else:
        try:
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
        st.write("AFD instance created (could not introspect reflection).")

    with st.spinner("Generating..."):
        try:
            response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode)
        except TypeError:
            # older respond signature without renderer_mode: fall back to default behavior
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

        st.subheader("Response")
        st.write(response)

        st.markdown("**AFD outputs**")
        try:
            st.write(f"- Coherence score: {coherence:.6f}")
        except Exception:
            st.write(f"- Coherence score: {coherence}")
        st.write(f"- Latest reflection (AFD reflection shown by default):")
        st.code(reflection)

        # Optional: detailed explainability dump
        if st.checkbox("Show detailed explainability (numeric states, prompts, metrics)"):
            explain = None
            try:
                explain = ami.get_last_explainability()
            except Exception:
                explain = None

            if explain is None:
                st.write("No explainability snapshot available.")
            else:
                st.markdown("**Explainability snapshot**")
                st.write("Renderer used:", explain.get("renderer_used"))
                st.write("Neutral prompt:", explain.get("neutral_prompt"))
                st.write("Renderer prompt (exact):")
                # renderer_prompt may be long; show as code
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

        # Optionally show stored neutralized prompt saved in memory for audit
        if st.checkbox("Show last neutralized prompt from memory"):
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
    "Notes: Default mode uses the AFD deterministic renderer so the answer is produced from the AFD math and the reflection will be the AFD reflection. "
    "If you explicitly choose the LLM renderer, the LLM will be used for rendering but the AFD reflection will still be recorded and shown."
)
