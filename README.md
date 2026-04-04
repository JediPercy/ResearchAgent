# ResearchAgent OS: Autonomous Multi-Agent ML Research Infrastructure

![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-containerized-0db7ed)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Framework](https://img.shields.io/badge/orchestration-CrewAI-orange)
![ML Stack](https://img.shields.io/badge/ML-PyTorch%20%7C%20Optuna-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**ResearchAgent OS** is a distributed, containerized multi-agent system engineered to automate the end-to-end machine learning research lifecycle. 

Transitioning away from monolithic LLM usage, ResearchAgent leverages a decoupled topology of specialized agents to execute complex applied science workflows. The system autonomously handles literature ingestion, neural network architecture design, model training loops, and technical documentation—packaged within a stateless web interface featuring Human-In-The-Loop (HITL) oversight.

---

## Key Capabilities

* **Autonomous Literature Review:** Agents dynamically scrape arXiv, parse PDFs, and extract complex mathematical architectures.
* **Dynamic Code Generation:** Translates academic mathematical formulations directly into custom, executable `PyTorch` neural network classes.
* **Human-In-The-Loop (HITL):** A stateless UI pipeline that pauses execution, allowing human engineers to review and modify YAML experiment configurations before compute resources are expended.
* **Containerized Deployment:** Fully Dockerized architecture ready for immediate deployment to Google Cloud Run or AWS ECS.

---

## Core Architecture & Agent Topology

The system operates on a multi-phase, hub-and-spoke model via the `AgentCore` communication protocol.

### Phase 1: The Research Team
* **Academic Librarian Agent:** Navigates academic databases and retrieves contextually relevant PDF URLs.
* **Summarizer Agent:** Ingests raw PDFs, extracting core methodologies, loss functions, and dataset requirements.
* **Senior ML Researcher Agent:** Synthesizes extracted data into a deterministic `YAML` MLOps configuration file representing the experiment.

### Phase 2: Human Oversight
* **Streamlit State Manager:** Pauses agent execution, surfacing the generated `YAML` for human validation and hyperparameter tuning.

### Phase 3: The Engineering Team
* **ML Engineer Agent:** Reads the approved `YAML` and dynamically writes the PyTorch training scripts, including custom data loaders, complex neural network classes, and evaluation metrics.

---

## Getting Started

ResearchAgent OS can be run via Docker (Recommended) or as a local Python environment.

### Prerequisites
* Valid LLM API Key (Gemini, Anthropic, OpenAI, etc.)
* Docker Desktop (If containerizing)
* Python 3.11+ (If running locally)

### Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ResearchAgent.git](https://github.com/yourusername/ResearchAgent.git)
   cd ResearchAgent
   touch .env
   GEMINI_API_KEY=your_actual_api_key_here (or your choice of model)
   ```
2. **Run via Docker**
   ```bash
   # Build the image
   docker build -t research-agent-os .

   # Run the container locally on port 8501
   docker run --env-file .env -p 8501:8501 research-agent-os
   ```

3. **Run via Local Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv research_env
   source research_env/bin/activate  # On Windows: research_env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Launch the Streamlit OS
   streamlit run app.py
   ```

## Technology Stack
* **UI & Deployment:** Streamlit, Docker
* **Agent Orchestration:** Python, CrewAI
* **Machine Learning:** PyTorch, Scikit-Learn
* **Data Engineering (Planned):** Pandas, Numpy, PySpark 
* **Infrastructure (Planned):** AWS S3, Kubernetes (EKS)