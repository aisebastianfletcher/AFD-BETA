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
    if prompt:
        with st.spinner("Thinking..."):
            time.sleep(1)  # Simulate thinking delay
            response, coherence, reflection = afd_ami.respond(prompt)
        
        # Update history
        st.session_state.history.append({"prompt": prompt, "response": response, "coherence": coherence, "reflection": reflection})
        
        # Display results
        st.subheader("Response")
        st.write(response)
        st.write(f"**Coherence Score:** {coherence:.2f}")
        st.write(f"**Reflection Log:** {reflection}")
        
        # Display coherence trend
        st.subheader("Coherence Trend")
        try:
            df = afd_ami.load_memory()
            if not df.empty:
                scores = df['coherence'].tail(5).tolist()
                fig, ax = plt.subplots()
                ax.plot(scores, label='Coherence', color='#1f77b4')
                ax.set_title('Ethical Coherence Trend')
                ax.set_xlabel('Recent Interactions')
                ax.set_ylabel('Coherence Score')
                ax.set_ylim(0, 1)
                ax.legend()
                st.pyplot(fig)
            else:
                st.write("No trend graph available yet (empty CSV).")
        except Exception as e:
            st.error(f"Error loading trend graph: {e}")

# Display conversation history
if st.session_state.history:
    st.subheader("Conversation History")
    for i, entry in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Interaction {len(st.session_state.history) - i}"):
            st.write(f"**Prompt:** {entry['prompt']}")
            st.write(f"**Response:** {entry['response']}")
            st.write(f"**Coherence Score:** {entry['coherence']:.2f}")
            st.write(f"**Reflection Log:** {entry['reflection']}")

# Footer
st.write("© 2025 Sebastian Fletcher | CC BY-NC-ND 4.0")
