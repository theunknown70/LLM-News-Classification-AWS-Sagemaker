# End-to-End Multi-Class Text Classification with AWS SageMaker

This project demonstrates a complete MLOps pipeline for training, deploying, and invoking a multi-class text classification model using AWS SageMaker and Hugging Face Transformers. The model is trained to classify news headlines into four categories: Business, Science & Technology, Entertainment, and Health.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Step-by-Step Execution](#step-by-step-execution)
  - [1. Data Preparation & Exploration](#1-data-preparation--exploration)
  - [2. Model Training](#2-model-training)
  - [3. Real-Time Deployment](#3-real-time-deployment)
  - [4. Serverless Inference with Lambda](#4-serverless-inference-with-lambda)
  - [5. Batch Predictions (Optional)](#5-batch-predictions-optional)
- [Cleaning Up](#cleaning-up)

## Project Overview
The goal of this project is to build an end-to-end system that can classify news headlines. It covers all major stages of a machine learning lifecycle:
1.  **Data Analysis**: Exploring and preparing the raw dataset.
2.  **Training**: Using a managed AWS SageMaker Training Job to train a DistilBERT model from Hugging Face.
3.  **Deployment**: Deploying the trained model to a persistent, real-time SageMaker Endpoint.
4.  **Inference**: Invoking the endpoint using two methods:
    * A serverless API using AWS Lambda and API Gateway for low-latency, on-demand predictions.
    * A SageMaker Batch Transform job for offline, large-scale predictions.

## Architecture

### Real-Time Inference Flow
User --> API Gateway --> AWS Lambda --> SageMaker Endpoint --> Prediction

### Training & Batch Flow
Data (CSV on S3) --> SageMaker Training Job --> Trained Model (model.tar.gz on S3)
Input Data (JSONs on S3) --> SageMaker Batch Transform Job --> Predictions (JSONs on S3)

<!-- end list -->

## Features
- **Exploratory Data Analysis (EDA)** in a Jupyter Notebook.
- **Managed Training** with the SageMaker Python SDK and a Hugging Face Estimator.
- **Custom Scripts** for both training (`script.py`) and inference (`inference.py`).
- **Real-Time Deployment** to a scalable, managed SageMaker Endpoint.
- **Serverless Inference** using an AWS Lambda function for cost-effective and scalable predictions.
- **Batch Transform** for offline processing of large datasets.

## Prerequisites
1.  An **AWS Account**.
2.  An **IAM User** with permissions for S3, SageMaker, Lambda, API Gateway, and CloudWatch.
3.  The **AWS CLI** configured locally (optional but recommended).
4.  A dataset of news headlines. The one used for this project can be found on Kaggle: [News Aggregator Dataset](https://www.kaggle.com/datasets/uciml/news-aggregator-dataset). You will need the `newsCorpora.csv` file.

## Step-by-Step Execution

### 1. Data Preparation & Exploration
1.  **Create an S3 Bucket** in your AWS account to store data and model artifacts.
2.  **Upload your dataset** (e.g., `newsCorpora.csv`) to a folder within the bucket.
3.  Open `EDA_MultiClassTextClassification.ipynb` in a SageMaker Notebook instance.
4.  **Update the `s3_path` variable** in the first code cell to point to your CSV file in S3.
5.  Run all cells in the notebook to analyze the data.

### 2. Model Training
1.  Open `script.py` and **update the `s3_path` variable** to the same S3 location of your dataset.
2.  Open `TrainingNotebook.ipynb`.
3.  Review the hyperparameters in the `hyperparameters` dictionary. Adjust if needed.
4.  In the `.fit()` command, update the S3 URI to point to your dataset location.
5.  Run all cells in the notebook. This will start a SageMaker Training Job, which may take 15-20 minutes. You can monitor its progress in the AWS SageMaker console.

### 3. Real-Time Deployment
1.  Once the training job is complete, open `Deployment.ipynb`.
2.  The notebook will automatically retrieve the S3 path of the `model.tar.gz` from the previous training job.
3.  Define a unique `endpoint_name`.
4.  Run all cells to deploy the model. This will create a live SageMaker Endpoint, which may take 5-10 minutes to become active.
5.  The final cells will test the endpoint with a sample headline and then delete it to save costs. **Do not run the `predictor.delete_endpoint()` cell if you plan to proceed to the next step.**

### 4. Serverless Inference with Lambda
1.  **Create an IAM Role** for Lambda. Attach the `AWSLambdaBasicExecutionRole` policy and create a new inline policy granting `sagemaker:InvokeEndpoint` permission.
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sagemaker:InvokeEndpoint",
                "Resource": "arn:aws:sagemaker:<your-region>:<your-account-id>:endpoint/<your-endpoint-name>"
            }
        ]
    }
    ```
2.  In the AWS Lambda console, **create a new function** (Author from scratch, Python 3.9+ runtime). Assign the IAM role you just created.
3.  Copy the code from `aws-lambda-llm-endpoint-invoke-function.py` into the Lambda code editor.
4.  **Update the `endpoint_name` variable** in the Lambda code to match the endpoint you deployed.
5.  **Deploy** the Lambda function.
6.  (Optional but Recommended) In the Lambda console, add a trigger and select **API Gateway**. Create a new HTTP API. This will give you a public URL to invoke your function.

### 5. Batch Predictions (Optional)
1.  Run the `load-test.py` script locally (`python load-test.py`). This creates `inputs.tar.gz`.
2.  Upload `inputs.tar.gz` to your S3 bucket.
3.  In the AWS SageMaker console, go to **Batch Transform Jobs** and create a new job.
4.  Select the model trained earlier.
5.  For the input, specify the S3 path to your `inputs.tar.gz` file and set the content type to `application/json`.
6.  Specify an S3 output path for the predictions.
7.  Create and run the job. The results will appear in the output path as `inputs.tar.gz.out`.

## Cleaning Up
To avoid ongoing charges, make sure to delete the AWS resources you created:
1.  **Delete the SageMaker Endpoint**: If you haven't already, run `predictor.delete_endpoint()` in the `Deployment.ipynb` notebook or delete it from the SageMaker console.
2.  **Delete the SageMaker Model**: In the SageMaker console, go to `Inference > Models` and delete the model artifact.
3.  **Stop/Delete the Notebook Instance**: Stop or delete the SageMaker Notebook instance to prevent charges for the compute instance.
4.  **Delete the Lambda Function and API Gateway**: Go to the Lambda and API Gateway consoles to delete the functions and APIs you created.
5.  **Empty and Delete the S3 Bucket**: Delete all objects from your S3 bucket and then delete the bucket itself.
