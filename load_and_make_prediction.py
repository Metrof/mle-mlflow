import mlflow
import numpy as np
import os
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

model_name = "churn_model_evgeniyPol"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

loaded_model = mlflow.pyfunc.load_model(model_uri)

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_predictions = loaded_model.predict(X_test)
model_predictions = model_predictions.astype(int)

assert model_predictions.dtype == int

print("Первые 10 предсказаний:")
print(model_predictions[:10])