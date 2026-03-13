import pandas as pd
import numpy as np
import joblib


model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")


new_patient = np.array([2, 120, 70, 28, 100, 25.0, 0.5, 30])


columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age']


new_patient_df = pd.DataFrame([new_patient], columns=columns)


new_patient_scaled = scaler.transform(new_patient_df)
prediction = model.predict(new_patient_scaled)

print("Prediction:", prediction[0])
