# Use an official, lightweight Python image
FROM python:3.11-slim

# Set environment variables to keep Python behavior predictable
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 
# Ensures console output is not buffered so we see logs instantly
ENV PYTHONUNBUFFERED=1 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies securely and keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the standard Streamlit port for local development
EXPOSE 8501

# Command to run the app. 
# We use 'sh -c' so it can read the $PORT environment variable if deployed to GCP.
CMD sh -c "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"