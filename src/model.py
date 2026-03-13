import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


X_test = pd.read_csv("data/preptraining_data/X_test_scaled.csv")
X_train = pd.read_csv("data/preptraining_data/X_train_scaled.csv")
y_test = pd.read_csv("data/preptraining_data/y_test.csv")
y_train = pd.read_csv("data/preptraining_data/y_train.csv")



y_train = y_train.values.ravel()
y_test = y_test.values.ravel()



model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "models/GradientBoosting_model.pkl")


predictions = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
