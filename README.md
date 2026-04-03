# ResearchAgent: Autonomous Multi-Agent ML Research Infrastructure

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/orchestration-CrewAI-orange)
![Memory](https://img.shields.io/badge/memory-ChromaDB-green)
![ML Stack](https://img.shields.io/badge/ML-PyTorch%20%7C%20Optuna-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**ResearchAgent** is a distributed, multi-agent system engineered to automate the end-to-end machine learning research lifecycle. 

Transitioning away from monolithic LLM usage, ResearchAgent leverages a decoupled topology of specialized agents to execute complex applied science workflows. The system is designed to autonomously handle literature ingestion, big data processing, neural network architecture, model training loops, and technical documentation—all synchronized via shared vector memory and scalable cloud storage.

---

## Core Architecture & Agent Topology

The system operates on a hub-and-spoke model, governed by a Main Agent that manages state, memory, and task delegation via the `AgentCore` communication protocol.

### 0. The Main Agent (Orchestrator)
The central nervous system of the cluster.
* **State & Memory:** Local vector embeddings via `ChromaDB` (with production architecture mapped for `Pinecone`).
* **Storage:** Unified data lake and artifact storage via AWS S3 containers.
* **Routing:** `AgentCore` protocol for inter-agent context sharing.

### 1. Literature Review Agent
Responsible for grounding experiments in current academic research.
* **Document Processing:** Ingests and parses academic papers using `PyPDF`.
* **Data Gathering:** Executes custom web scraping scripts and interfaces with academic APIs.
* **Processing:** Utilizes frontier LLMs for deep summarization and dense vector embeddings.

### 2. Data Agent
Handles raw data ingestion, ETL pipelines, and feature engineering.
* **Big Data Compute:** Designed for `PySpark` and `Databricks` environments.
* **Local Processing:** `numpy` and `pandas` for structured dataframe manipulation.
* **API Integration:** Pulls dynamic datasets from endpoints like the NBA API, HuggingFace, GitHub, and Spotify.
* **Storage:** Reads/writes cleaned datasets directly to the Main Agent's shared S3 bucket.

### 3. Experiment Agent
Focuses entirely on model design and configuration management.
* **Framework:** Architects deep learning models in `PyTorch`.
* **Optimization:** Configures hyperparameter tuning studies using `Optuna`.
* **State Tracking:** Generates deterministic `YAML` configuration files for every experimental run to ensure reproducibility.

### 4. Train/Evaluate Agent
The heavy compute executor for the ML pipelines.
* **Execution Environment:** Triggers code via Bash scripts or Jupyter Notebooks.
* **Compute Infrastructure:** Designed to deploy on Kubernetes Clusters (EKS) for optimized GPU orchestration or AWS EC2 instances.
* **ML Stack:** `PyTorch`, `Scikit-Learn`, `Keras`.
* **Experiment Tracking:** Logs metrics via AWS SageMaker.
* **Evaluation:** Generates performance visualization using `Seaborn` and `Matplotlib`.

### 5. Reporting & Writing Agent
Translates quantitative results into digestible technical documentation.
* **Document Generation:** Native Markdown writer with LaTeX rendering capabilities for mathematical formulation.
* **Visual Integration:** Embeds `Matplotlib` and `Seaborn` plots directly into reports.
* **Presentation Layer:** Automates slide deck design via the Canva API.
* **Knowledge Management:** Syncs final reports to organizational wikis (Confluence, Notion, Obsidian) and exports via `PyPDF`.

### 6. CI/CD & Workflow Agents (Auxiliary)
* **GitHub Agent:** Automates version control by pushing code changes and verified experiment configurations to existing repositories via the GitHub API, facilitating seamless collaboration across applied science teams.
* **JIRA Agent:** Manages Agile workflows, creates tickets for data pipeline blockers, and keeps the SWE/ML research team aligned on deployment schedules.

---

## Technology Stack

* **Agent Orchestration:** Python, CrewAI, LangChain
* **Vector Memory:** ChromaDB (Local Prototype) -> Pinecone (Production)
* **Machine Learning:** PyTorch, Keras, Scikit-Learn, Optuna
* **Data Engineering:** Pandas, Numpy, PySpark (Planned)
* **Infrastructure & Compute:** AWS S3, AWS EC2, AWS SageMaker, Kubernetes (Planned)
* **Documentation & Vis:** PyPDF, PyYAML, Seaborn, Canva API, LaTeX

---

## Getting Started (Local Prototyping)

Currently, ResearchAgent is configured for local execution using ChromaDB for zero-latency, cost-free vector memory during the prototyping phase.

### Prerequisites
* Python 3.10+
* Valid LLM API Key (Gemini, Anthropic, etc.)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/ResearchAgent.git](https://github.com/yourusername/ResearchAgent.git)
   ```

   ```
   cd ResearchAgent
   ```

2. Create an isolated virtual environment
This prevents dependency conflicts with your system-wide Python installation.

On macOS and Linux:
```python -m venv research_env```

On Windows:
```python -m venv research_env```

3. Activate the virtual environment
Ensure your terminal is actively using the isolated environment before installing packages.

On macOS and Linux:
```source research_env/bin/activate```

On Windows:
```research_env\Scripts\activate```

4. Install project dependencies
Install the required packages for the orchestration framework and local vector memory.

```pip install -r requirements.txt```

(If you are building from scratch and do not have a requirements.txt file yet, run): 
```pip install crewai langchain-google-genai chromadb pandas numpy python-dotenv)```

5. Configure environment variables
Create a hidden .env file in the root directory of the project to securely store your API keys.

```touch .env```

Open the .env file in your text editor and add:

[MODEL_CHOICE]_API_KEY=your_actual_api_key_here
(I used Google Gemini initially.)

Execution
Run the base orchestrator to initialize the core Data -> Experiment pipeline. Ensure your virtual environment is activated before running.

```python main.py```