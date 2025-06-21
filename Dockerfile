# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
COPY packages.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    $(cat packages.txt) \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 