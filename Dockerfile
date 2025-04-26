# Use the official Python image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port your Flask app will run on (default is 5000 for Flask)
EXPOSE 5000

# Set environment variables (optional, but recommended for Flask)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]