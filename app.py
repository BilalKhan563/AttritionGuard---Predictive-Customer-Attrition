#All the process of Data science and Machine Learnig was done on jupyter notebook.
# For the purpose of serving the model we are using VSCode.
# The model trained and saved is opened here as well as the file of cleaned data.
# We are using FastAPI


# Importing necessary packages
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import joblib
import pandas as pd  #pandas for loading the csv

app = FastAPI()

# Loading the saved model
model = joblib.load('C:\\Users\\Bilal\\ml_project\\logistic_regression_model.pkl')

# Loading the cleaned and preprocessed data file
cleaned_data = pd.read_csv('C:\\Users\\Bilal\\ml_project\\cleaned_data.csv')  # Update with the actual path

class PredictionInput(BaseModel):

    ClientID: int
    Gender: int
    IsSenior: int
    HasPartner: int
    HasDependents: int
    ServiceDuration: int
    HasPhoneService: int
    HasMultiplePhoneServices: int
    InternetServiceType: int
    HasCloudSecurity: int
    HasCloudBackup: int
    HasDeviceCoverage: int
    HasTechSupport: int
    HasStreamingTV: int
    HasStreamingMovies: int
    SubscriptionType: int
    HasElectronicBilling: int
    PaymentMethod: int
    MonthlySubscriptionFee: int
    TotalSubscriptionCost: int

class PredictionOutput(BaseModel):
    # Here we are defining the output data structure using "Pydantic BaseModel"
    prediction: int

@app.get("/")
def read_root():
    return {"FastAPI App": "Running!"}


@app.post("/predict")
async def predict(data: PredictionInput, background_tasks: BackgroundTasks):
    try:
        # Here making predictions using the loaded model
        features = [data.Gender, data.IsSenior, data.HasPartner, data.HasDependents,
                    data.ServiceDuration, data.HasPhoneService, data.HasMultiplePhoneServices,
                    data.InternetServiceType, data.HasCloudSecurity, data.HasCloudBackup, data.HasDeviceCoverage,
                    data.HasTechSupport, data.HasStreamingTV, data.HasStreamingMovies, data.SubscriptionType,
                    data.HasElectronicBilling, data.PaymentMethod, data.MonthlySubscriptionFee, data.TotalSubscriptionCost]

        #features for debugging(the process of identifying and removing errors from computer hardware or software)
        print("Input Features:", features)

        predictions = model.predict([features])

        # Preparin the response
        response = {'prediction': int(predictions[0])}

        # Executing background tasks 'if needed'
        background_tasks.add_task(save_prediction_to_database, features, predictions[0])
        
        return response

    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=f'Internal Server Error: {str(e)}')
