# AttritionGuard---Predictive-Customer-Attrition
AttritionGuard: Predict and prevent customer churn with our Logistic Regression model. FastAPI-powered server for real-time predictions. Code includes EDA, model training, and local deployment. Explore and deploy with confidence

# Predictive Attrition

## Overview

This repository contains a machine learning model developed to predict customer attrition, accompanied by a FastAPI server to serve prediction requests. The model is based on logistic regression, chosen for its predictive capabilities.

## Project Structure

- **Jupyter Notebook (ML Model Development):** `CustomerAttrition_Model.ipynb`
  - Data preprocessing, exploratory data analysis (EDA), logistic regression model training, and evaluation.

- **FastAPI Server (Model Serving):** `app.py`
  - Implements a local server using FastAPI to serve predictions.
  - Defines a POST endpoint (`/predict`) for receiving prediction requests in JSON format.

## How to Use

### Machine Learning Model

1. Open and run `AttritionPredict Server.ipynb` in a Jupyter environment.
2. Follow the step-by-step instructions in the notebook for data preprocessing, model training, and evaluation.

### FastAPI Server

1. Ensure you have FastAPI installed (`pip install fastapi`).
2. Run the FastAPI server using `uvicorn app:app --reload`.
3. Access the server at `http://127.0.0.1:8000`.

#### Prediction Request Example

```bash
curl -X POST -H "Content-Type: application/json" -d '{"ClientID": 123, "Gender": 0, ..., "TotalSubscriptionCost": 1500}' http://127.0.0.1:8000/predict
