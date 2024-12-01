# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Upgrade pip and install necessary dependencies
RUN pip install --upgrade pip
RUN pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost umap-learn

# Command to run your script
CMD ["python", "Credit_Card_Fraud.py"]

