import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import json

best_model, best_params = tune_xgb(X_train_transformed, y_train, X_val_transformed, y_val, max_evals=30)

with open("models_results/XGBoost/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("Best params saved:", best_params)