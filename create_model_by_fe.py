import os
import psycopg
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

num_columns = ["monthly_charges", "total_charges"]

#-----------
TABLE_NAME = "users_churn"

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

connection.update(postgres_credentials)

with psycopg.connect(**connection) as conn:

    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# --- Настройки окружения (те же, что и раньше) ---
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "users_churn_exp"

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# --- 1. Загрузка препроцессора из MLflow ---
# Укажите здесь RUN_ID того запуска, где вы сохранили column_transformer
LOGGED_PREPROCESSOR_RUN_ID = "8f3a042a1fe643d8a5ef1ab3afadda93" 
model_uri = f"runs:/{LOGGED_PREPROCESSOR_RUN_ID}/column_transformer"

print(f"Загрузка препроцессора из {model_uri}...")
preprocessor = mlflow.sklearn.load_model(model_uri)

# --- 2. Подготовка данных ---
# Загружаем ваш исходный датафрейм (df) здесь
# Не забудьте обработать NaN в числовых колонках, как мы обсуждали ранее!
df[num_columns] = df[num_columns].fillna(df[num_columns].median())

# Трансформируем данные загруженным препроцессором
X = preprocessor.transform(df)
y = df['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Обучение и регистрация модели ---
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

with mlflow.start_run(run_name="Training_with_Loaded_Preprocessor") as run:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    # Логируем параметры и метрики
    mlflow.log_param("preprocessor_run_id", LOGGED_PREPROCESSOR_RUN_ID)
    mlflow.log_metric("auc_roc", auc_score)
    
    # Регистрируем финальную модель
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Telecom_Churn_Final_Model"
    )

    print(f"Готово! AUC-ROC: {auc_score:.4f}")