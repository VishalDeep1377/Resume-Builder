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

# Create a dedicated directory for NLTK data and set permissions
RUN mkdir -p /app/nltk_data && chmod -R 755 /app/nltk_data

# Download NLTK data into the dedicated directory
RUN python -m nltk.downloader -d /app/nltk_data punkt stopwords wordnet

# Copy the NLTK config to the root user's home directory
COPY nltk.cfg /root/.nltk.cfg

# Set the NLTK_DATA environment variable to ensure it's found
ENV NLTK_DATA /app/nltk_data

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# Use environment variable for port to be compatible with cloud platforms
CMD ["/bin/sh", "-c", "echo '--- ENVIRONMENT VARIABLES ---' && printenv && echo '--- STARTING APP ---' && streamlit run main_app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false"] 