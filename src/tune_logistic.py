

import pandas as pd                                
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV    
import joblib                                      

X_train = pd.read_csv("data/preptraining_data/X_train_scaled.csv")  
y_train = pd.read_csv("data/preptraining_data/y_train.csv")          


y_train = y_train.values.ravel()  


model = LogisticRegression(max_iter=1000)


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],             
    'penalty': ['l1', 'l2'],                   
    'solver': ['liblinear', 'saga']            
}


grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='accuracy',    
                           cv=5,                 
                           verbose=1)


grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


joblib.dump(grid_search.best_estimator_, "models/logistic_model_tuned.pkl")
