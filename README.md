# Diabetes Risk Assistant

An interactive Machine Learning dashboard built with Python and PyQt5. This application allows users to input health metrics and receive a diabetes risk assessment using multiple trained models.

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
   "git clone https://github.com/Huz4y1/health_risk_predictor.git"

2. Install the required dependencies:
   "pip install -r requirements.txt"

3. Run the application using Python:
   "python main.py"
