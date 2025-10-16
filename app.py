import streamlit as st
import os
from afd_ami_core import AFDInfinityAMI
import pandas as pd
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(page_title="AFD∞-AMI Ethical Assistant", layout="wide")

# Initialize or load the AFDInfinityAMI instance
@st.cache_resource
def get_afd_ami():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return AFDInfinityAMI(use_openai=bool(api_key), openai_api_key=api_key)

afd_ami = get_afd_ami()

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Title and description
st.title("AFD∞-AMI Ethical Assistant")
st.write("A non-reward-based ethical assistant using Autonomous Fletcher Dynamics. Ask me anything ethical!")

# Input prompt
prompt = st.text_input("Enter your question:", key="prompt_input")

if st.button("Submit"):
    prompt = prompt.strip()
    if prompt:
        with st.spinner("Thinking..."):
            try:
                time.sleep(1)  # Simulate thinking delay
                response, coherence, reflection = afd_ami.respond(prompt)

                # Update history
                st.session_state.history.append({
                    "prompt": prompt,
                    "response": response,
                    "coherence": coherence,
                    "reflection": reflection
                })

                # Display results
                st.subheader("Response")
                st.write(response)
                st.markdown(f"**Coherence Score:** `{coherence:.2f}`")
                st.markdown(f"**Reflection Log:** `{reflection}`")

            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a valid prompt.")

    # Display coherence trend
    st.subheader("Coherence Trend")
    try:
        df = afd_ami.load_memory()
        if not df.empty and 'coherence' in df:
            scores = df['coherence'].tail(5).tolist()
            fig, ax = plt.subplots()
            ax.plot(scores, marker='o', linestyle='-', label='Coherence', color='#1f77b4')
            ax.set_title('Ethical Coherence Trend')
            ax.set_xlabel('Recent Interactions')
            ax.set_ylabel('Coherence Score')
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No trend graph available yet.")
    except Exception as e:
        st.error(f"Error loading trend graph: {e}")

# Display conversation history
if st.session_state.history:
    st.subheader("Conversation History")
    for i, entry in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Interaction {len(st.session_state.history) - i}"):
            st.markdown(f"**Prompt:** {entry['prompt']}")
            st.markdown(f"**Response:** {entry['response']}")
            st.markdown(f"**Coherence Score:** `{entry['coherence']:.2f}`")
            st.markdown(f"**Reflection Log:** `{entry['reflection']}`")

# Footer
st.write("© 2025 Sebastian Fletcher | CC BY-NC-ND 4.0")
