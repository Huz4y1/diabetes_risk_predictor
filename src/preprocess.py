import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib



df = pd.read_csv('data/raw/diabetes.csv')


cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_invalid_zeros:
    median = df[col].median()
    df[col] = df[col].replace(0, median)


columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_cleaned = df.dropna(subset=columns_to_check)


X = df_cleaned.drop("Outcome", axis=1)
y = df_cleaned["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")


os.makedirs("data/preptraining_data", exist_ok=True)


pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/preptraining_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/preptraining_data/X_test_scaled.csv", index=False)
y_train.to_csv("data/preptraining_data/y_train.csv", index=False)
y_test.to_csv("data/preptraining_data/y_test.csv", index=False)
