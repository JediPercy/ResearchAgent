import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# 1. Load the variables from the .env file
load_dotenv()

# 2. Safely retrieve the key (Optional validation step)
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

# ==========================================
# 1. Define the Agents
# ==========================================

data_agent = Agent(
    role='Senior Data Engineer',
    goal='Gather, clean, and preprocess datasets using numpy and pandas, and retrieve external data from APIs.',
    backstory=(
        "You are a meticulous data engineer. Your job is to fetch raw data, "
        "handle missing values, and structure it perfectly into pandas DataFrames "
        "so the research team can train models without data pipeline errors."
    ),
    verbose=True,
    allow_delegation=False
)

# ==========================================
# 2. Define the Tasks
# ==========================================

data_task = Task(
    description=(
        "Simulate fetching a dataset for a binary classification problem "
        "(e.g., predicting customer churn). Outline the python code using pandas "
        "to clean the data, handle nulls, and normalize the features."
    ),
    expected_output="A Python script using pandas and numpy to clean the simulated dataset.",
    agent=data_agent
)

experiment_task = Task(
    description=(
        "Based on the cleaned data pipeline from the Data Engineer, write a complete "
        "YAML configuration file for this experiment. It must include hyperparameters "
        "for a PyTorch model and specify an Optuna study setup for tuning."
    ),
    expected_output="A well-formatted YAML file containing model architecture, hyperparameters, and Optuna ranges.",
    agent=experiment_agent
)

# ==========================================
# 3. Assemble the Crew (AgentCore)
# ==========================================

agent_core = Crew(
    agents=[data_agent, experiment_agent],
    tasks=[data_task, experiment_task],
    process=Process.sequential, 
    memory=True,                # Activates ChromaDB locally
    verbose=True
)

# ==========================================
# 4. Run the Pipeline
# ==========================================

if __name__ == "__main__":
    print("Initializing AgentCore ML Research Team...")
    result = agent_core.kickoff()
    
    print("\n========================================")
    print("FINAL EXPERIMENT OUTPUT:")
    print("========================================")
    print(result)