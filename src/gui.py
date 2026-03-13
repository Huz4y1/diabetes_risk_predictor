import joblib
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, 
    QGridLayout, QMessageBox, QComboBox, QFrame
)
from PyQt5.QtCore import Qt

MODELS = {
    "Logistic Regression": "models/logistic_model_tuned.pkl",
    "Random Forest": "models/RandomForest_model.pkl",
    "Gradient Boosting": "models/GradientBoosting_model.pkl"
}
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/processed/diabetes_cleaned.csv"

FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin', 'BMI', 
    'DiabetesPedigreeFunction', 'Age'
]

FEATURE_INFO = {
    "Pregnancies": ("Ex: 2", "Number of times pregnant"),
    "Glucose": ("Ex: 120", "Plasma glucose (mg/dL)"),
    "BloodPressure": ("Ex: 70", "Diastolic pressure (mm Hg)"),
    "SkinThickness": ("Ex: 20", "Triceps fold thickness (mm)"),
    "Insulin": ("Ex: 85", "2-hour serum insulin (mu U/ml)"),
    "BMI": ("Ex: 30.5", "Body mass index (weight/height²)"),
    "DiabetesPedigreeFunction": ("Ex: 0.5", "Genetic likelihood score"),
    "Age": ("Ex: 45", "Age in years")
}

class DiabetesGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetes Risk Assistant")
        self.inputs = {}
        self.current_model = None
        try:
            self.scaler = joblib.load(SCALER_PATH)
        except:
            self.scaler = None
        
        self.init_ui()
        self.load_dataset()
        self.change_model("Logistic Regression")

    def init_ui(self):
        layout = QGridLayout()
        layout.setSpacing(10)
        
        title = QLabel("Diabetes Health Dashboard")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(title, 0, 0, 1, 2, Qt.AlignCenter)

        layout.addWidget(QLabel("<b>Prediction Engine:</b>"), 1, 0)
        self.model_selector = QComboBox()
        self.model_selector.addItems(MODELS.keys())
        self.model_selector.currentTextChanged.connect(self.change_model)
        layout.addWidget(self.model_selector, 1, 1)

        row = 2
        for feat in FEATURES:
            label = QLabel(f"<b>{feat}</b>")
            edit = QLineEdit()
            edit.setPlaceholderText(FEATURE_INFO[feat][0])
            desc = QLabel(FEATURE_INFO[feat][1])
            desc.setStyleSheet("color: #7f8c8d; font-size: 11px; font-style: italic;")
            
            self.inputs[feat] = edit
            layout.addWidget(label, row, 0)
            layout.addWidget(edit, row, 1)
            row += 1
            layout.addWidget(desc, row, 0, 1, 2)
            row += 1

        self.predict_btn = QPushButton("Analyze Health Data")
        self.predict_btn.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.predict_btn.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_btn, row, 0, 1, 2)
        row += 1

        self.result_frame = QFrame()
        self.result_frame.setStyleSheet("background-color: #ecf0f1; border-radius: 10px;")
        res_layout = QGridLayout(self.result_frame)
        self.result_label = QLabel("Waiting for input...")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #34495e;")
        res_layout.addWidget(self.result_label, 0, 0, Qt.AlignCenter)
        layout.addWidget(self.result_frame, row, 0, 1, 2)
        row += 1

        self.comparison_plot = pg.PlotWidget(title="Your Levels vs. Healthy Averages")
        self.comparison_plot.setBackground('w')
        self.comparison_plot.showGrid(x=True, y=True)
        layout.addWidget(self.comparison_plot, row, 0, 1, 2)
        row += 1

        self.save_btn = QPushButton("Save Report (CSV)")
        self.save_btn.clicked.connect(self.export_prediction)
        layout.addWidget(self.save_btn, row, 0, 1, 2)
        
        self.setLayout(layout)

    def change_model(self, model_name):
        try:
            self.current_model = joblib.load(MODELS[model_name])
        except:
            pass

    def load_dataset(self):
        try:
            self.df = pd.read_csv(DATA_PATH)
        except:
            self.df = None

    def make_prediction(self):
        try:
            raw_values = [float(self.inputs[f].text()) for f in FEATURES]
            input_df = pd.DataFrame([raw_values], columns=FEATURES)
            
            data_to_predict = self.scaler.transform(input_df) if self.scaler else input_df
            prob = self.current_model.predict_proba(data_to_predict)[0][1]
            
            status = "High Risk" if prob > 0.5 else "Low Risk"
            color = "#e74c3c" if prob > 0.5 else "#27ae60"
            
            self.result_label.setText(f"Result: {status} ({prob*100:.1f}%)")
            self.result_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
            self.update_comparison_graph(raw_values)
            
        except Exception as e:
            QMessageBox.warning(self, "Entry Error", "Please ensure all fields are filled correctly.")

    def update_comparison_graph(self, values):
        if self.df is None: return
        self.comparison_plot.clear()
        
        means = self.df[FEATURES].mean().values
        x = np.arange(len(FEATURES))
        
        bg_avg = pg.BarGraphItem(x=x-0.2, height=means, width=0.3, brush='#bdc3c7', name="Average")
        bg_user = pg.BarGraphItem(x=x+0.2, height=values, width=0.3, brush='#3498db', name="You")
        
        self.comparison_plot.addItem(bg_avg)
        self.comparison_plot.addItem(bg_user)
        self.comparison_plot.getAxis('bottom').setTicks([[(i, f[:5]) for i, f in enumerate(FEATURES)]])

    def export_prediction(self):
        try:
            data = {f: [self.inputs[f].text()] for f in FEATURES}
            data['Assessment'] = [self.result_label.text()]
            pd.DataFrame(data).to_csv("health_report.csv", index=False)
            QMessageBox.information(self, "Report Saved", "Success! Data saved to health_report.csv")
        except:
            pass