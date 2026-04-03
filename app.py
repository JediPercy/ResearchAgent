import streamlit as st
import os
import sys
import re
import io 
import zipfile

# --- Terminal Capture Engine ---
class StreamlitCapture:
    def __init__(self, st_placeholder):
        self.st_placeholder = st_placeholder
        self.buffer = ""
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        clean_data = self.ansi_escape.sub('', data)
        if clean_data:
            self.buffer += clean_data
            display_text = self.buffer[-2000:] if len(self.buffer) > 2000 else self.buffer
            self.st_placeholder.code(display_text, language="text")

    def flush(self):
        pass
    
def create_zip_of_experiments():
    """Packages the experiments folder into an in-memory zip file."""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists("experiments"):
            for root, dirs, files in os.walk("experiments"):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Keep the archive clean by removing the absolute path
                    archive_name = os.path.relpath(file_path, "experiments")
                    zf.write(file_path, archive_name)
    memory_file.seek(0)
    return memory_file

# --- Initialize Session State ---
if 'pipeline_stage' not in st.session_state:
    st.session_state.pipeline_stage = 'setup'
if 'yaml_content' not in st.session_state:
    st.session_state.yaml_content = ""

# --- UI Setup ---
st.set_page_config(page_title="ResearchAgent OS", page_icon="🔬", layout="wide")

with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown("Ensure your `.env` file contains your `GEMINI_API_KEY`.")
    st.markdown("---")
    st.info("Status: **Human-In-The-Loop Enabled**")
    
    # A reset button to start over
    if st.button("🔄 Reset Pipeline"):
        st.session_state.pipeline_stage = 'setup'
        st.rerun()

st.title("🔬 ResearchAgent OS")

# ==========================================
# STAGE 1: Setup & Research
# ==========================================
if st.session_state.pipeline_stage == 'setup':
    st.subheader("Phase 1: Autonomous Literature Review")
    topic = st.text_input("Research Topic", value="predicting customer churn using deep learning")

    if st.button("🚀 Start Phase 1 (Research)"):
        st.markdown("---")
        with st.expander("🕵️‍♂️ Live Agent Telemetry", expanded=True):
            terminal_output = st.empty()
        
        try:
            # Import Phase 1 logic
            from main import research_crew, search_task
            search_task.description = f"Search arXiv for exactly two recent papers on '{topic}'. Return their PDF URLs."
            
            # Hijack Terminal
            original_stdout = sys.stdout
            sys.stdout = StreamlitCapture(terminal_output)
            
            # Execute Research Team
            research_crew.kickoff()
            
            sys.stdout = original_stdout
            
            # Read the generated YAML to present it to the user
            with open("experiments/churn_lit_review_v1.yaml", "r") as f:
                st.session_state.yaml_content = f.read()
                
            st.session_state.pipeline_stage = 'review'
            st.rerun()
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"Pipeline crashed: {str(e)}")

# ==========================================
# STAGE 2: Human Review (HITL)
# ==========================================
elif st.session_state.pipeline_stage == 'review':
    st.subheader("Phase 2: Human Review & Approval")
    st.warning("⚠️ The Research Team has drafted the experiment architecture. Please review and edit the parameters below before sending it to the ML Engineer.")
    
    # Editable Text Area for the YAML
    edited_yaml = st.text_area("Edit Configuration (YAML)", value=st.session_state.yaml_content, height=400)
    
    if st.button("✅ Approve & Generate PyTorch Code"):
        st.markdown("---")
        with st.expander("💻 Live Engineering Telemetry", expanded=True):
            terminal_output = st.empty()
            
        try:
            # 1. Save the user's edits back to the disk so the Engineer reads the updated version!
            with open("experiments/churn_lit_review_v1.yaml", "w") as f:
                f.write(edited_yaml)
                
            # 2. Import Phase 2 logic
            from main import engineering_crew
            
            # 3. Hijack Terminal
            original_stdout = sys.stdout
            sys.stdout = StreamlitCapture(terminal_output)
            
            # 4. Execute Engineering Team
            engineering_crew.kickoff()
            
            sys.stdout = original_stdout
            st.session_state.pipeline_stage = 'complete'
            st.rerun()
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"Pipeline crashed: {str(e)}")

# ==========================================
# STAGE 3: Complete
# ==========================================
elif st.session_state.pipeline_stage == 'complete':
    st.subheader("🎉 Experiment Generation Complete")
    
    if os.path.exists("experiments/train.py"):
        with open("experiments/train.py", "r") as f:
            with st.expander("View Generated PyTorch Code", expanded=False):
                st.code(f.read(), language="python")

    st.markdown("### 📦 Download Your Assets")
    zip_buffer = create_zip_of_experiments()
    
    st.download_button(
        label="⬇️ Download Experiments Folder (.zip)",
        data=zip_buffer,
        file_name="research_experiment.zip",
        mime="application/zip"
    )