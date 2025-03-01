# Use slim Python image for minimal size
FROM python:3.10-slim  

# Set working directory
WORKDIR /app  

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libmagic1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*  

# Copy requirements file
COPY requirements.txt .  

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt  

# Copy application code
COPY . .  

# Expose Streamlit default port
EXPOSE 8501  

# Streamlit Health Check (More stable)
HEALTHCHECK CMD curl --fail http://localhost:8501/healthz || exit 1  

# Set entrypoint for Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]  
