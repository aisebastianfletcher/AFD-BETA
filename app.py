# Resilient Streamlit UI for AFD∞-AMI with improved variability and hidden diagnostics
import streamlit as st
import traceback
import os
import pandas as pd

st.set_page_config(page_title="AFD∞-AMI (resilient)", layout="centered")
st.title("AFD∞-AMI — Resilient UI (AFD renderer default)")

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
    # we do not display 'Using OpenAI: True' by default in the main UI
    st.write("OPENAI_API_KEY present:", key_present)
    st.markdown(
        "Default mode uses the deterministic AFD renderer so responses are generated from the AFD math. "
        "You can increase temperature for more variety; enable memory so the agent reflects on recent Q/A."
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

# temperature slider: extended max for larger variety
temperature = st.slider("Temperature (controls variability)", min_value=0.0, max_value=2.5, value=0.0, step=0.05)

# memory controls
include_memory = st.checkbox("Include memory context in reflections and responses", value=True)
memory_window = st.number_input("Memory window (most recent entries to consider)", min_value=1, max_value=500, value=10, step=1)

# reproducibility seed (optional)
use_seed = st.checkbox("Use reproducibility seed", value=False)
seed = None
if use_seed:
    seed = st.number_input("Seed (integer)", min_value=0, max_value=2**31-1, value=0, step=1)

renderer_mode = "afd" if renderer_choice.startswith("AFD") else "llm"
submit = st.button("Generate response")

def make_stub_ami():
    class StubAMI:
        def __init__(self):
            self.use_openai = False
            self.reflection_log = ["Running in stub mode."]
            self.alpha = self.beta = 1.0
            self.gamma = self.delta = 0.5

        def respond(self, prompt_in, renderer_mode="afd", include_memory=True, seed=None):
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
            ami = AFDInfinityAMI(use_openai=prefer_openai, temperature=temperature, memory_window=int(memory_window))
        except Exception as e:
            st.error("Failed to instantiate AFDInfinityAMI; falling back to stub mode.")
            st.exception(e)
            ami = make_stub_ami()

    # Do NOT show "Using OpenAI: True" in main output. Only show when user asks for hidden diagnostics.
    # Provide a compact immediate status without revealing keys or auth.
    try:
        st.write("Latest reflection available:", bool(ami.get_latest_reflection()))
    except Exception:
        st.write("AFD instance created (could not introspect reflection).")

    with st.spinner("Generating..."):
        try:
            response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode, include_memory=include_memory, seed=int(seed) if seed is not None else None)
        except TypeError:
            try:
                response, coherence, reflection = ami.respond(prompt)
            except Exception as e:
                st.error("An error occurred; falling back to stub output.")
                st.exception(e)
                ami = make_stub_ami()
                response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode, include_memory=include_memory, seed=int(seed) if seed is not None else None)
        except Exception as e:
            st.error("An error occurred; falling back to stub output.")
            st.exception(e)
            ami = make_stub_ami()
            response, coherence, reflection = ami.respond(prompt, renderer_mode=renderer_mode, include_memory=include_memory, seed=int(seed) if seed is not None else None)

        # Show the authoritative AFD reflection (keeps it concise)
        st.subheader("AFD reflection")
        st.code(reflection)

        # Human-style reflection and diagnostics are hidden by default. Reveal via checkbox.
        if st.checkbox("Show hidden reflections & diagnostics"):
            explain = None
            try:
                explain = ami.get_last_explainability()
            except Exception:
                explain = None

            # Reveal human-style reflection if available
            if explain and explain.get("human_reflection"):
                st.markdown("**Human-style reflection (simulated, post-hoc)**")
                st.write(explain.get("human_reflection"))

            # Reveal OpenAI usage and other debug lines (only when user asks)
            st.markdown("**Hidden diagnostics**")
            st.write("Using OpenAI:", bool(getattr(ami, "use_openai", False)))
            st.write("Reflection log (recent):")
            for r in (getattr(ami, "reflection_log", [])[-6:]):
                st.text(r)

        st.markdown("---")
        st.subheader("Response")
        st.write(response)

        st.markdown("**AFD outputs**")
        try:
            st.write(f"- Coherence score: {coherence:.6f}")
        except Exception:
            st.write(f"- Coherence score: {coherence}")

        # detailed explainability dump (unchanged)
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

        # memory display / download
        if st.checkbox("Show memory (stored responses)"):
            try:
                mem = ami.load_memory()
                if not mem.empty:
                    st.dataframe(mem.tail(50))
                    csv = mem.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("Download memory as CSV", csv, file_name="response_memory.csv", mime="text/csv")
                else:
                    st.info("Memory is empty.")
            except Exception as e:
                st.warning("Could not read memory file.")
                st.exception(e)

st.markdown("---")
st.caption(
    "Notes: Answers default to the deterministic AFD renderer (no LLM). Temperature increases variability; use the seed for reproducible outputs. Memory influences reflections when enabled. Hidden diagnostics (including 'Using OpenAI') are only shown when you explicitly request them."
)
