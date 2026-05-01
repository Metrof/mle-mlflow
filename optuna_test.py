import os
import pandas as pd
import psycopg
import mlflow
import numpy as np
import optuna
from dotenv import load_dotenv
from collections import defaultdict
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, 
    recall_score, f1_score, log_loss
)
from optuna.samplers import TPESampler
from optuna.integration.mlflow import MLflowCallback

# 1. ЗАГРУЗКА ОКРУЖЕНИЯ И КОНФИГУРАЦИЯ
load_dotenv()

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_prediction_tpe"
RUN_NAME = "model_bayesian_search"
STUDY_DB_NAME = "sqlite:///local.study.db"
STUDY_NAME = "churn_model"

# Настройки S3 и MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

# 2. ПОДКЛЮЧЕНИЕ К БД И ЗАГРУЗКА ДАННЫХ
TABLE_NAME = "users_churn_transformed"
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

# Подготовка данных (замените 'target' на реальное имя колонки, если оно другое)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ФУНКЦИЯ OBJECTIVE ДЛЯ OPTUNA
def objective(trial: optuna.Trial) -> float:
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5),
        "loss_function": "Logloss",
        "task_type": "CPU",
        "random_seed": 0,
        "iterations": 300,
        "verbose": False,
    }
    
    model = CatBoostClassifier(**param)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    metrics = defaultdict(list)

    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        train_x, val_x = X_train.iloc[train_index], X_train.iloc[val_index]
        train_y, val_y = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(train_x, train_y)
        mlflow.catboost.log_model(model, artifact_path="cv")
        
        prediction = model.predict(val_x)
        probas = model.predict_proba(val_x)[:, 1]

        # Извлекаем ошибки 1 и 2 рода
        _, err1, _, err2 = confusion_matrix(val_y, prediction, normalize='all').ravel()
        
        metrics["err1"].append(err1)
        metrics["err2"].append(err2)
        metrics["auc"].append(roc_auc_score(val_y, probas))
        metrics["precision"].append(precision_score(val_y, prediction))
        metrics["recall"].append(recall_score(val_y, prediction))
        metrics["f1"].append(f1_score(val_y, prediction))
        metrics["logloss"].append(log_loss(val_y, probas))

    auc = np.mean(metrics["auc"])

    return auc

# 4. ЗАПУСК ОПТИМИЗАЦИИ
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# Запускаем основной процесс поиска
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id, nested=True) as run:
    run_id = run.info.run_id

    # Настройка Callback для связи дочерних испытаний с родительским Run
    mlflc = MLflowCallback(
        tracking_uri=f'http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}',
        metric_name='auc',
        create_experiment=False,
        mlflow_kwargs={
            'experiment_id': experiment_id, 
            'tags': {'mlflow.parentRunId': run_id},
            'nested': True 
        }
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_DB_NAME,
        direction="maximize",
        sampler=TPESampler(seed=0),
        load_if_exists=True
    )

    study.optimize(objective, n_trials=10, callbacks=[mlflc])

# 5. ФИНАЛЬНОЕ ЛОГИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ
with mlflow.start_run(run_name="final_model_log", experiment_id=experiment_id):
    best_params = study.best_params
    final_params = {
        **best_params, 
        "iterations": 300, 
        "loss_function": "Logloss", 
        "random_seed": 0, 
        "verbose": False
    }
    
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_train, y_train)
    
    mlflow.log_params(best_params)
    mlflow.catboost.log_model(final_model, artifact_path="model")
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best params: {best_params}")