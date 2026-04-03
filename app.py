import streamlit as st
import os

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

# Execution Button
if st.button("🚀 Initialize Multi-Agent Pipeline"):
    st.markdown("---")
    
    # Visual feedback placeholders
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.info("System Booting... Please wait.")
    
    try:
        # Import the core logic from your main.py file
        # We import here so it only loads when the button is clicked
        from main import agent_core, search_task
        
        # Dynamically update the search task with the user's input from the web UI
        search_task.description = f"Search arXiv for exactly two recent papers on '{topic}'. Return their PDF URLs."
        
        status_text.info(f"Agents deployed. Researching: {topic}")
        progress_bar.progress(25)
        
        # Kick off the crew!
        result = agent_core.kickoff()
        progress_bar.progress(100)
        
        status_text.success("Pipeline Execution Complete!")
        
        # Display the final output
        st.markdown("### 📋 Final System Output")
        st.write(result)
        
        # Provide a way to view the generated files
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
        status_text.error(f"Pipeline crashed: {str(e)}")