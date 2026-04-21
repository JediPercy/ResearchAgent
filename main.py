import os
import requests
import arxiv
from io import BytesIO
from pypdf import PdfReader
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import logging
import warnings

# ==========================================
# 0. Environment Setup
# ==========================================
load_dotenv()

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
warnings.filterwarnings("ignore")

logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
# ==========================================
# 1. Define Custom Tools
# ==========================================

class SaveYAMLTool(BaseTool):
    name: str = "Save YAML Configuration File"
    description: str = (
        "Saves the generated YAML configuration content to a local file. "
        "Inputs must be 'content' (the YAML string) and 'filename' (e.g., 'model.yaml')."
    )

    def _run(self, content: str, filename: str) -> str:
        directory = "experiments"
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully saved experiment configuration to {filepath}"

class ArxivSearchTool(BaseTool):
    name: str = "Search arXiv for ML Papers"
    description: str = (
        "Searches the arXiv database for academic papers. "
        "Input should be a specific search query (e.g., 'customer churn prediction deep learning'). "
        "Returns a list of relevant papers including their titles, summaries, and PDF links."
    )

    def _run(self, query: str) -> str:
        # Explicitly configure the client to respect rate limits
        client = arxiv.Client(
            page_size=2,          # Match our max_results to prevent 100-item page requests
            delay_seconds=3.0,    # Mandatory 3-second delay between API calls
            num_retries=3         # Automatically retry if we still hit a 429
        )
        
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in client.results(search):
            paper_info = (
                f"Title: {paper.title}\n"
                f"Summary: {paper.summary}\n"
                f"PDF Link: {paper.pdf_url}\n"
                "-----------------"
            )
            results.append(paper_info)

        if not results:
            return "No papers found for that query."
        return "\n\n".join(results)

class PDFReaderTool(BaseTool):
    name: str = "Read PDF from URL"
    description: str = (
        "Reads and extracts text from a PDF given its URL. "
        "Input should be the direct URL to the PDF file."
    )

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            
            text = ""
            num_pages = min(len(reader.pages), 3) 
            for i in range(num_pages):
                text += reader.pages[i].extract_text() + "\n\n"
                
            return text
        except Exception as e:
            return f"Failed to read PDF: {str(e)}"
        
class ReadYAMLTool(BaseTool):
    name: str = "Read YAML Configuration File"
    description: str = (
        "Reads a YAML file from the experiments directory. "
        "Input should be the filename (e.g., 'churn_lit_review_v1.yaml')."
    )

    def _run(self, filename: str) -> str:
        filepath = os.path.join("experiments", filename)
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            return f"Failed to read file: {str(e)}"

class SavePythonTool(BaseTool):
    name: str = "Save Python Script"
    description: str = (
        "Saves the generated Python code to a local file. "
        "Inputs must be 'content' (the python code) and 'filename' (e.g., 'train.py')."
    )

    def _run(self, content: str, filename: str) -> str:
        content = content.replace("```python", "").replace("```", "").strip()
        
        directory = "experiments"
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully saved Python script to {filepath}"

# Instantiate the tools
save_yaml_tool = SaveYAMLTool()
arxiv_search_tool = ArxivSearchTool()
pdf_reader_tool = PDFReaderTool()
read_yaml_tool = ReadYAMLTool()
save_python_tool = SavePythonTool()


# ==========================================
# 2. Define the Agents
# ==========================================

llm_string = "gemini/gemini-2.0-flash"

librarian_agent = Agent(
    role='Academic Librarian',
    goal='Search the arXiv database to find the most relevant PDF links for machine learning methodologies.',
    backstory="You are an expert at navigating academic databases. You find the best paper URLs and pass them down the chain.",
    verbose=True,
    allow_delegation=False,
    tools=[arxiv_search_tool], 
    llm=llm_string,
    max_retry_limit=5
)

summarizer_agent = Agent(
    role='Senior Summarizer',
    goal='Take PDF URLs, read the documents, and summarize the mathematical methodologies into a markdown report.',
    backstory="You are a speed-reading researcher. You extract exact PyTorch layers, loss functions, and hyperparameters from dense texts.",
    verbose=True,
    allow_delegation=False,
    tools=[pdf_reader_tool], 
    llm=llm_string,
    max_retry_limit=5
)

researcher_agent = Agent(
    role='Lead ML Researcher',
    goal='Take literature summaries and design machine learning experiments, saving them as YAML configurations.',
    backstory="You are the lead architect. You do not search or read. You take summaries and design production-ready model configurations.",
    verbose=True,
    allow_delegation=False,
    tools=[save_yaml_tool], 
    llm=llm_string,
    max_retry_limit=5
)

ml_engineer_agent = Agent(
    role='Senior ML Engineer',
    goal='Read YAML configurations and write executable PyTorch training scripts.',
    backstory=(
        "You are a brilliant Python developer specializing in PyTorch. You take "
        "architectural blueprints and turn them into bug-free, production-ready code."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[read_yaml_tool, save_python_tool], 
    llm=llm_string,
    max_retry_limit=5
)

data_engineer_agent = Agent(
    role="Senior Data Engineer",
    goal="Build robust Python data pipelines using Pandas to ingest, clean, and preprocess raw datasets for machine learning.",
    backstory="You are a veteran Data Engineer. You excel at handling missing values, encoding categorical variables, and scaling numerical features. You write clean, well-documented Pandas code that saves processed datasets as CSVs.",
    verbose=True,
    allow_delegation=False,
    tools=[save_python_tool], 
    llm=llm_string,
    max_retry_limit=5 
)

# ==========================================
# 3. Define the Tasks
# ==========================================

search_task = Task(
    description="Search arXiv for exactly two recent papers on 'predicting customer churn using deep learning'. Return their PDF URLs.",
    expected_output="A list of two PDF URLs.",
    agent=librarian_agent
)

read_task = Task(
    description="Using the PDF URLs from the Librarian, read the papers. Extract the specific neural network architectures (e.g., LSTM, FeedForward) and hyperparameters they used.",
    expected_output="A dense markdown summary of the architectures and tuning parameters found in the papers.",
    agent=summarizer_agent
)

experiment_task = Task(
    description=(
        "Based on the literature summary, write a complete YAML configuration file for a new churn prediction experiment. "
        "Incorporate the architecture trends you see in the summary. "
        "CRITICAL: You MUST use the 'Save YAML Configuration File' tool to save your output. Name the file 'churn_lit_review_v1.yaml'."
    ),
    expected_output="A confirmation string stating the YAML file has been saved to the disk.",
    agent=researcher_agent
)

data_pipeline_task = Task(
    description=(
        "Write a Python script named `experiments/data_pipeline.py`.\n"
        "This script must:\n"
        "1. Use pandas to download the Telco Customer Churn dataset from a public URL (e.g., 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv').\n"
        "2. Handle missing values appropriately.\n"
        "3. Encode categorical variables using One-Hot Encoding or Label Encoding.\n"
        "4. Scale numerical features using StandardScaler.\n"
        "5. Save the final cleaned dataframe as `experiments/cleaned_churn_data.csv`.\n"
        "CRITICAL: Use the 'Save Python Script' tool to save your output as 'data_pipeline.py'."
    ),
    expected_output="A fully functioning Python script saved as `experiments/data_pipeline.py` that outputs a cleaned CSV.",
    agent=data_engineer_agent
)

coding_task = Task(
    description=(
        "Use the 'Read YAML Configuration File' tool to read 'churn_lit_review_v1.yaml'. "
        "Based EXACTLY on that configuration, write a complete, executable PyTorch script. "
        "CRITICAL DATA INSTRUCTION: Do NOT generate dummy data. You must load the dataset "
        "from 'experiments/cleaned_churn_data.csv' using pandas, convert the features and "
        "target labels into PyTorch tensors, and build a proper DataLoader for the training loop. "
        "CRITICAL: Use the 'Save Python Script' tool to save your output as 'train.py'."
    ),
    expected_output="A confirmation string stating the Python script has been saved, utilizing the cleaned CSV data.",
    agent=ml_engineer_agent
)

# ==========================================
# 4. Assemble the Crew
# ==========================================

# Phase 1: Generates the YAML
research_crew = Crew(
    agents=[librarian_agent, summarizer_agent, researcher_agent],
    tasks=[search_task, read_task, experiment_task],
    process=Process.sequential, 
    memory=False,                
    verbose=True,
    max_rpm=100
)

# Phase 2: Generates the Python Code
engineering_crew = Crew(
    agents=[data_engineer_agent, ml_engineer_agent],
    tasks=[data_pipeline_task, coding_task],
    process=Process.sequential, 
    memory=False,                
    verbose=True,
    max_rpm=100
)