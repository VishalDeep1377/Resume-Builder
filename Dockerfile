# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies from packages.txt
# This is useful for libraries that need to be compiled
COPY packages.txt .
RUN xargs -a packages.txt apt-get install -y --no-install-recommends

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 