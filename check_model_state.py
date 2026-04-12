from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

client = MlflowClient()
original_name = "churn_model_evgeniyPol"
filter_string = f"name LIKE '%{original_name}%'"
search_results = client.search_registered_models(filter_string=filter_string)
new_name = f"{original_name}_b2c"

if not search_results:
    print(f"Модель с упоминанием '{original_name}' не найдена!")
else:
    model_object = search_results[0]
    current_full_name = model_object.name
    print(f"Найдена модель: {current_full_name}")

    versions = client.get_latest_versions(original_name)
    versions = sorted(versions, key=lambda x: int(x.version))

versions = client.get_latest_versions(original_name)
versions = sorted(versions, key=lambda x: int(x.version))

last_version = versions[-1].version      
prev_version = versions[-2].version      

print("Текущие состояния версий:")
for v in versions:
    print(f"Версия {v.version}: статус {v.current_stage}")

client.transition_model_version_stage(
    name=original_name,
    version=last_version,
    stage="Production"
)

client.transition_model_version_stage(
    name=original_name,
    version=prev_version,
    stage="Staging"
)

client.rename_registered_model(
    name=original_name,
    new_name=new_name
)

print(f"\nГотово! Модель '{original_name}' переименована в '{new_name}'")
print(f"Версия {last_version} теперь в Production, версия {prev_version} — в Staging.")