import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


X_test = pd.read_csv("data/preptraining_data/X_test_scaled.csv")
y_test = pd.read_csv("data/preptraining_data/y_test.csv")


models = {
    "Logistic Regression": "models/logistic_model.pkl",
    "Random Forest": "models/RandomForest_model.pkl",
    "Gradient Boosting": "models/GradientBoosting_model.pkl",
    "Logistic Regression Tuned": "models/logistic_model_tuned.pkl"
}

results = []  


for model_name, model_path in models.items():
    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)



    
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


results_df = pd.DataFrame(results)
results_df.to_csv("data/results/model_results.csv", index=False)
