# Use the official Python image from the Docker Hub
FROM python:3.11

RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the working directory
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV PORT=8080

# Expose the port on which your app will run
EXPOSE 8080

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]