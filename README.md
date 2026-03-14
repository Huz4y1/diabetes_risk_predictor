# Diabetes Risk Predictor

A binary classification machine learning model, which predicts if a person is at risk of getting diabetes

## Features
- **Multi-Model Prediction:** Switch between Logistic Regression, Random Forest, and Gradient Boosting models.
- **Visual Analytics:** Compare your health data against dataset averages in real-time.
- **Data Scaling:** Integrated with a pre-trained Scaler to ensure prediction accuracy.
- **Report Export:** Save your prediction results and inputs directly to a CSV file.

## Project Structure
- `main.py`: The entry point for the application.
- `src/gui.py`: Contains the PyQt5 interface logic and data visualization.
- `models/`: Directory containing trained `.pkl` models and the scaler.
- `data/`: Contains the processed dataset used for background averages.

## Installation

1. Clone this repository:
   "git clone https://github.com/Huz4y1/diabetes_risk_predictor.git"

2. Install the required dependencies:
   "pip install -r requirements.txt"

3. Run the application using Python:
   "python main.py"

## Video Showcase
![DiabetesAI_showcase](https://github.com/user-attachments/assets/58470232-bd14-4219-bcf6-43deac91a784)

## Data Set Used
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
