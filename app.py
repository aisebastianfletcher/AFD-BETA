# diagnostics: replace the import with this to show the full import error in the UI
import streamlit as st
import traceback

try:
    from afd_ami_core import AFDInfinityAMI
except Exception as e:
    st.error("Error importing afd_ami_core.py — showing full traceback below (no secrets will be printed):")
    st.text(str(e))
    st.text(traceback.format_exc())
    # stop execution so the app doesn't continue in a broken state
    st.stop()

# --- normal app code starts here ---
import os

st.set_page_config(page_title="AFD∞-AMI", layout="centered")

st.title("AFD∞-AMI — Neutralized, AFD-driven Assistant")
st.write(
    "This app translates user input into a neutral representation, runs the AFD mathematical "
    "framework, and renders a neutral explanation based only on the AFD metrics and the neutral input."
)

# -------- Diagnostic / Debug (safe) ----------
with st.expander("Debug: OpenAI key & AFD status (click to expand)"):
    key_present = bool(st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY present:", key_present)

    # Optional: instantiate the AFD class for diagnostics (may download HF models if OpenAI not used)
    init_diag = st.checkbox(
        "Initialize AFDInfinityAMI for diagnostics (may download models if OpenAI not used)",
        value=False,
    )
    if init_diag:
        try:
            ami_diag = AFDInfinityAMI()
            st.write("AFDInfinityAMI created.")
            st.write("Using OpenAI:", bool(ami_diag.use_openai))
            st.write("Latest reflection:", ami_diag.get_latest_reflection())
            # Show available public methods (safe introspection)
            st.write("AFD instance methods:", [n for n in dir(ami_diag) if callable(getattr(ami_diag, n)) and not n.startswith("_")])
        except Exception as e:
            st.error("Failed to initialize AFDInfinityAMI for diagnostics.")
            st.exception(e)
    else:
        st.write("Initialization skipped.")

st.markdown("---")

# -------- User Interaction ----------
st.header("Ask the assistant (neutralized + AFD-driven)")
prompt = st.text_area(
    "Enter your question or prompt. The assistant will first neutralize it, run the AFD math, and then render a neutral explanation.",
    value="Explain the ethical considerations of using AI in hiring decisions.",
    height=140,
)

col1, col2 = st.columns([1, 1])
with col1:
    use_openai_checkbox = st.checkbox(
        "Prefer OpenAI if available (fallback to local HF if auth or network fails)",
        value=True,
    )
with col2:
    include_memory_checkbox = st.checkbox("Show last neutralized prompt from memory", value=True)

submit = st.button("Generate response")

if submit:
    # Instantiate the AFD agent when needed. Let the class detect secrets/env internally.
    try:
        # We pass use_openai flag to prefer OpenAI, but AFDInfinityAMI also auto-detects the key.
        ami = AFDInfinityAMI(use_openai=use_openai_checkbox)
        # Immediately show reflection so you can see auth/init status
        st.write("Using OpenAI:", bool(ami.use_openai))
        st.write("Latest reflection:", ami.get_latest_reflection())
    except Exception as e:
        st.error("Failed to initialize the assistant.")
        st.exception(e)
    else:
        with st.spinner("Neutralizing input and generating response..."):
            try:
                response, coherence, reflection = ami.respond(prompt)
            except Exception as e:
                st.error("An error occurred while generating the response.")
                st.exception(e)
                # Show any available reflection info
                try:
                    st.write("Latest reflection:", ami.get_latest_reflection())
                except Exception:
                    pass
            else:
                st.subheader("Response")
                st.write(response)

                st.markdown("**AFD outputs**")
                st.write(f"- Coherence score: {coherence:.6f}")
                st.write(f"- Latest reflection: {reflection}")

                # Optionally show the neutralized prompt saved in memory for audit
                if include_memory_checkbox:
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
    "Notes: The app will not display any API keys. If OpenAI is selected but authentication fails, "
    "the system will automatically fall back to the local HF pipeline if available and record the reason in the reflection log."
)
