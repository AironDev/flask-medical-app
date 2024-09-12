# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5002

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5002"]
