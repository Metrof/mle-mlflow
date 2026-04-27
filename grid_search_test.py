import os
import pandas as pd
import psycopg
import mlflow
import mlflow.catboost
import mlflow.sklearn
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, log_loss, confusion_matrix
)
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# 1. КОНФИГУРАЦИЯ
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "churn_grid_search_exp"
RUN_NAME = 'model_grid_search'
REGISTRY_MODEL_NAME = "catboost_churn_model"

# ID запуска, где лежит твой ColumnTransformer
PREPROCESSOR_RUN_ID = "8f3a042a1fe643d8a5ef1ab3afadda93" 

# Настройки S3 и MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 
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

# 3. ПРЕДОБРАБОТКА (базовая очистка)
print("Очистка данных...")
num_features = ["monthly_charges", "total_charges"]
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Сортировка по времени ПЕРЕД применением препроцессора и сплитом
df = df.sort_values(by="begin_date")
X_raw = df.drop(columns=['target', "begin_date"])
y = df['target'].astype(int)

# 4. ПРИМЕНЕНИЕ ЗАГРУЖЕННОГО ПРЕПРОЦЕССОРА
print(f"Загрузка препроцессора из MLflow (ID: {PREPROCESSOR_RUN_ID})...")
model_uri = f"runs:/{PREPROCESSOR_RUN_ID}/column_transformer"
preprocessor = mlflow.sklearn.load_model(model_uri)
X_transformed = preprocessor.transform(X_raw)

# Восстанавливаем DataFrame после препроцессинга
try:
    cols_out = preprocessor.get_feature_names_out()
    X_prep = pd.DataFrame(X_transformed, columns=cols_out)
except:
    X_prep = pd.DataFrame(X_transformed)

# 5. РАЗДЕЛЕНИЕ НА TRAIN/TEST
# Используем X_prep (обработанные признаки)
X_train, X_test, y_train, y_test = train_test_split(
    X_prep, y, test_size=0.2, shuffle=False
)

# 6. ИНИЦИАЛИЗАЦИЯ ПОИСКА (GRID SEARCH)
print("Запуск Grid Search...")
params = {
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 3]
}

model_base = CatBoostClassifier(
    iterations=300,
    loss_function="Logloss",
    random_seed=0,
    task_type='CPU',
    verbose=False
)

cv = GridSearchCV(
    estimator=model_base,
    param_grid=params,
    scoring='roc_auc',
    cv=2,
    n_jobs=-1
)

clf = cv.fit(X_train, y_train)
cv_results = pd.DataFrame(clf.cv_results_)
best_params = clf.best_params_

# 7. ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ
model_best = CatBoostClassifier(
    **best_params,
    iterations=300,
    loss_function="Logloss",
    random_seed=0,
    task_type='CPU',
    verbose=False
)
model_best.fit(X_train, y_train)

# Предсказания
prediction = model_best.predict(X_test)
probas = model_best.predict_proba(X_test)[:, 1]

# 8. РАСЧЕТ МЕТРИК
metrics = {}
_, err1, _, err2 = confusion_matrix(y_test, prediction, normalize='all').ravel()

metrics["err1"] = err1
metrics["err2"] = err2
metrics["auc"] = roc_auc_score(y_test, probas)
metrics["precision"] = precision_score(y_test, prediction)
metrics["recall"] = recall_score(y_test, prediction)
metrics["f1"] = f1_score(y_test, prediction)
metrics["logloss"] = log_loss(y_test, probas) # Считаем по вероятностям!

# Метрики кросс-валидации для лучшей модели
best_idx = clf.best_index_
metrics['mean_fit_time'] = cv_results.loc[best_idx, 'mean_fit_time']
metrics['std_fit_time'] = cv_results.loc[best_idx, 'std_fit_time']
metrics['mean_test_score'] = cv_results.loc[best_idx, 'mean_test_score']
metrics['std_test_score'] = cv_results.loc[best_idx, 'std_test_score']
metrics["best_score"] = clf.best_score_

# 9. ЛОГИРОВАНИЕ В MLFLOW
pip_requirements = ["catboost", "scikit-learn", "pandas", "psycopg", "python-dotenv"]
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]

# Получаем ID эксперимента
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    # Параметры и метрики
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    
    # Модель CatBoost
    mlflow.catboost.log_model(
        cb_model=model_best,
        artifact_path='models',
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements
    )
    
    # Объект поиска и таблица результатов
    mlflow.sklearn.log_model(cv, artifact_path='cv')
    cv_results.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")

    print(f"Успех! Run ID: {run.info.run_id}")
    print(f"Best ROC-AUC: {metrics['auc']:.4f}")