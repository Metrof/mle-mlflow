import os
import pandas as pd
import psycopg
import mlflow
import ast # Для корректного чтения списков из CSV
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. ЗАГРУЗКА ОКРУЖЕНИЯ
load_dotenv()

# Настройки MLflow и S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

RUN_ID = "042f9d59867c4f64ad2178ecb01b2dc5" # ID запуска с отбором
PREPROCESSOR_RUN_ID = "8f3a042a1fe643d8a5ef1ab3afadda93" 
mlflow.set_tracking_uri("http://127.0.0.1:5000")

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

# 3. ПРЕДОБРАБОТКА
print("Очистка и трансформация...")
num_features = ["monthly_charges", "total_charges"]
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

df = df.sort_values(by="begin_date")
X_raw = df.drop(columns=['target', "begin_date"])
y = df['target'].astype(int)

# 4. ПРИМЕНЕНИЕ ПРЕПРОЦЕССОРА
model_uri = f"runs:/{PREPROCESSOR_RUN_ID}/column_transformer"
preprocessor = mlflow.sklearn.load_model(model_uri)
X_transformed = preprocessor.transform(X_raw)

try:
    cols_out = preprocessor.get_feature_names_out()
    X_prep = pd.DataFrame(X_transformed, columns=cols_out)
except:
    X_prep = pd.DataFrame(X_transformed)

# Разделение (важно: shuffle=False как в прошлом скрипте)
X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, shuffle=False)

# 5. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ АРТЕФАКТОВ
print("Загрузка списков признаков...")
local_sfs = mlflow.artifacts.download_artifacts(run_id=RUN_ID, artifact_path="feature_selection_plots/sfs.csv")
local_sbs = mlflow.artifacts.download_artifacts(run_id=RUN_ID, artifact_path="feature_selection_plots/sbs.csv")

sfs_res = pd.read_csv(local_sfs, index_col=0)
sbs_res = pd.read_csv(local_sbs, index_col=0)

# Превращаем строки типа "('a', 'b')" в реальные списки
top_sfs = list(ast.literal_eval(sfs_res.iloc[-1]['feature_names']))
top_sbs = list(ast.literal_eval(sbs_res.iloc[-1]['feature_names']))

# Формируем пересечение и объединение (Задание 2)
interc_features = list(set(top_sfs) & set(top_sbs))
union_features = list(set(top_sfs) | set(top_sbs))

# 6. ФУНКЦИЯ ОБУЧЕНИЯ
def train_and_register(feature_list, experiment_name, model_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name): 
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train[feature_list], y_train)
        
        auc = roc_auc_score(y_test, model.predict_proba(X_test[feature_list])[:, 1])
        
        mlflow.log_params({"num_features": len(feature_list)})
        mlflow.log_metric("roc_auc", auc)
        
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        
        run = mlflow.active_run()
        print(f"Эксперимент: {experiment_name}")
        print(f"Имя запуска (RUN NAME): {model_name}")
        print(f"ID запуска (RUN ID): {run.info.run_id}")

# 7. ЗАПУСК
train_and_register(interc_features, "feature_selection_intersection", "model_inter_and_union")
train_and_register(union_features, "feature_selection_union", "model_inter_and_union")