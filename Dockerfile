# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies from packages.txt
COPY packages.txt .
RUN apt-get update && apt-get install -y --no-install-recommends $(cat packages.txt) && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local NLTK data into the container
COPY nltk_data/ /usr/local/share/nltk_data/

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# Use environment variable for port and add headless flags for deployment
CMD streamlit run main_app.py --server.port $PORT --server.headless true --browser.gatherUsageStats false 