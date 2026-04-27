import os
import pandas as pd
import psycopg
import mlflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from autofeat import AutoFeatClassifier

# 1. ЗАГРУЗКА ОКРУЖЕНИЯ
load_dotenv()

# Настройки MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "users_churn_exp"
PREPROCESSOR_RUN_ID = "8f3a042a1fe643d8a5ef1ab3afadda93" 

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# 2. ПОДКЛЮЧЕНИЕ К БД И ЗАГРУЗКА ДАННЫХ
TABLE_NAME = "users_churn"
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
    "sslmode": "require",
}

print("Извлечение данных из Postgres...")
with psycopg.connect(**postgres_credentials) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# 3. ПРЕДОБРАБОТКА И ОЧИСТКА (NaN)
print("Очистка данных...")
# Исправляем типы и заполняем пропуски в критичных колонках
num_features = ["monthly_charges", "total_charges"]
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Сортировка по времени перед разделением
df = df.sort_values(by="begin_date")
target_col = 'target'
X_raw = df.drop(columns=[target_col, "begin_date"]) # Убираем таргет и дату из признаков
y = df[target_col].astype(int)

# 4. ПРИМЕНЕНИЕ ЗАГРУЖЕННОГО ПРЕПРОЦЕССОРА
print(f"Загрузка препроцессора из MLflow (ID: {PREPROCESSOR_RUN_ID})...")
model_uri = f"runs:/{PREPROCESSOR_RUN_ID}/column_transformer"
preprocessor = mlflow.sklearn.load_model(model_uri)

X_transformed = preprocessor.transform(X_raw)

# Превращаем результат обратно в DataFrame для AutoFeat
try:
    cols_out = preprocessor.get_feature_names_out()
    X_prep = pd.DataFrame(X_transformed, columns=cols_out)
except AttributeError:
    X_prep = pd.DataFrame(X_transformed)

# 5. ГЕНЕРАЦИЯ ПРИЗНАКОВ ЧЕРЕЗ AUTOFEAT
print("Запуск AutoFeat (генерация признаков)...")
# Разделим на трейн/тест уже подготовленные данные
X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, shuffle=False)

afc = AutoFeatClassifier(
    transformations=('log', '1/', 'sqrt', 'abs'), 
    feateng_steps=1, 
    n_jobs=-1
)

X_train_final = afc.fit_transform(X_train, y_train)
X_test_final = afc.transform(X_test)

# 6. ЛОГИРОВАНИЕ ИТОГОВОЙ МОДЕЛИ
print("Логирование в MLflow...")
try:
    exp_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
except AttributeError:
    exp_id = mlflow.create_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="final_autofeat_run", experiment_id=exp_id) as run:
    # Сохраняем AutoFeat как трансформер
    mlflow.sklearn.log_model(afc, artifact_path="autofeat_model")
    
    # Обучаем простую логистическую регрессию на новых признаках
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_final, y_train)
    
    # Логируем итоговую модель и метрики
    mlflow.sklearn.log_model(clf, artifact_path="logistic_regression_model")
    mlflow.log_metric("accuracy", clf.score(X_test_final, y_test))
    mlflow.log_param("total_features", X_train_final.shape[1])

    print(f"Готово! Run ID: {run.info.run_id}")