import streamlit as st
import os
import sys
import re

class StreamlitCapture:
    """
    Hijacks sys.stdout to pipe terminal logs directly into the Streamlit UI.
    """
    def __init__(self, st_placeholder):
        self.st_placeholder = st_placeholder
        self.buffer = ""
        # CrewAI uses ANSI escape codes for terminal colors (red, green, etc.)
        # This regex strips them out so they don't look like gibberish in the browser.
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        clean_data = self.ansi_escape.sub('', data)
        if clean_data:
            self.buffer += clean_data
            # Only display the last 2000 characters to prevent UI lag
            display_text = self.buffer[-2000:] if len(self.buffer) > 2000 else self.buffer
            self.st_placeholder.code(display_text, language="text")

    def flush(self):
        pass

# Set up the visual page configuration
st.set_page_config(
    page_title="ResearchAgent OS",
    page_icon="🔬",
    layout="wide"
)

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown("Ensure your `.env` file contains your `GEMINI_API_KEY`.")
    st.markdown("---")
    st.info("This system autonomously searches arXiv, reads PDFs, designs architectures, and writes PyTorch code.")

# Main UI
st.title("🔬 ResearchAgent OS")
st.subheader("Autonomous Machine Learning Pipeline")

# User Input
topic = st.text_input("Research Topic", value="predicting customer churn using deep learning")

if st.button("🚀 Initialize Multi-Agent Pipeline"):
    st.markdown("---")
    
    status_text = st.info("System Booting... Please wait.")
    
    # Create an open drop-down box to hold the live logs
    with st.expander("🕵️‍♂️ Live Agent Telemetry", expanded=True):
        terminal_output = st.empty()
    
    try:
        from main import agent_core, search_task
        
        search_task.description = f"Search arXiv for exactly two recent papers on '{topic}'. Return their PDF URLs."
        status_text.info(f"Agents deployed. Researching: {topic}")
        
        # --- The Hijack ---
        original_stdout = sys.stdout
        sys.stdout = StreamlitCapture(terminal_output)
        
        # Execute the Crew
        result = agent_core.kickoff()
        
        # --- Restore the Terminal ---
        sys.stdout = original_stdout
        
        status_text.success("Pipeline Execution Complete!")
        
        st.markdown("### 📋 Final System Output")
        st.write(result)
        
        st.markdown("### 📂 Generated Artifacts")
        if os.path.exists("experiments/churn_lit_review_v1.yaml"):
            with open("experiments/churn_lit_review_v1.yaml", "r") as f:
                with st.expander("View YAML Configuration"):
                    st.code(f.read(), language="yaml")
                    
        if os.path.exists("experiments/train.py"):
            with open("experiments/train.py", "r") as f:
                with st.expander("View Generated PyTorch Code"):
                    st.code(f.read(), language="python")

    except Exception as e:
        # If it crashes, make sure we still restore the terminal!
        sys.stdout = sys.__stdout__
        status_text.error(f"Pipeline crashed: {str(e)}")