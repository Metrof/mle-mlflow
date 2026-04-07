import os
import pandas as pd
import psycopg
import mlflow
from dotenv import load_dotenv

load_dotenv()

# 1. Настройка окружения для MLflow (чтобы он знал, куда класть артефакт в S3)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Данные для подключения к Postgres
connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"), 
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

connection.update(postgres_credentials)

# определим название таблицы, в которой хранятся наши данные.
TABLE_NAME = "users_churn"

# 3. Получение данных
with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 10") 
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

file_name = "columns_sol.txt"
with open(file_name, "w", encoding="utf-8") as fio:
    fio.write(",".join(df.columns.tolist()))

EXPERIMENT_NAME = "artifact_logging_experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="log_columns_artifact"):
    mlflow.log_artifact(file_name)
    print(f"Файл {file_name} успешно отправлен в MLflow!")