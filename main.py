import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, confusion_matrix, log_loss
)
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train)

prediction = model.predict(X_test)      
proba = model.predict_proba(X_test)[:, 1] 
# -----------------------------------------------

# ЗАДАНИЕ: Заведите словарь со всеми метриками
metrics = {}

_, err1, _, err2 = confusion_matrix(y_test, prediction, normalize='all').ravel()

auc = roc_auc_score(y_test, proba)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
logloss = log_loss(y_test, proba)

# Запишите значения метрик в словарь
metrics["err1"] = err1
metrics["err2"] = err2
metrics["auc"] = auc
metrics["precision"] = precision
metrics["recall"] = recall
metrics["f1"] = f1
metrics["logloss"] = logloss


EXPERIMENT_NAME = "churn_polyakov"
REGISTRY_MODEL_NAME = "churn_model_evgeniyPol"

client = MlflowClient()
try:
    model_details = client.get_registered_model(REGISTRY_MODEL_NAME)
    next_version = len(model_details.latest_versions) + 1 
    next_version = max([int(v.version) for v in model_details.latest_versions]) + 1
except:
    next_version = 1

RUN_NAME = f"model_registry_V{next_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

mlflow.set_tracking_uri("http://127.0.0.1:5000")

pip_requirements = ["scikit-learn==1.3.1", "pandas", "mlflow", "boto3"]

signature = infer_signature(X_test, prediction)

input_example = X_test[:5]

metadata = {
    "model_type": "tuned_logistic_regression",
    "data_version": "2025-07-17",
    "team": "mle_students"
}

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.log_metrics(metrics)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",                
        pip_requirements=pip_requirements,    
        signature=signature,                  
        input_example=input_example,          
        registered_model_name=REGISTRY_MODEL_NAME, 
        metadata=metadata,
        await_registration_for=60                     
    )