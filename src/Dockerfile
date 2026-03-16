FROM python:3.10-slim
# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
#It sets the default directory inside the container where all following commands (like COPY, RUN, and CMD) will be executed.
WORKDIR /app
#copy the requirements from the app folder
COPY src/requirements.txt .
RUN pip install -r requirements.txt
# Copy the CONTENTS of the local src/app folder into the container's WORKDIR (/app)
COPY src/ .
#directory is already set to app so now run app.py
CMD ["python", "main.py"]