import os
import pandas as pd
import psycopg
import mlflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# 1. ЗАГРУЗКА ОКРУЖЕНИЯ
load_dotenv()

# Настройки MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
PREPROCESSOR_RUN_ID = "8f3a042a1fe643d8a5ef1ab3afadda93" 

EXPERIMENT_NAME = "mlxtend_exp"
RUN_NAME = "feature_selection"
REGISTRY_MODEL_NAME = "users_churn_logistic_regression"
FS_ASSETS = "fs_assets" 

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

#-------------------------------------------------------------

# 6. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И СЕЛЕКТОРОВ
# Создаем классификатор по условию задачи
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

# Настройка SFS (Forward Selection)
sfs_selector = SFS(
    rf,
    k_features=10,             # Условие: 10 лучших признаков
    forward=True,              # Forward Selection
    floating=False,            # Выключено
    scoring='roc_auc',         # Основная метрика
    cv=4,                      # Кросс-валидация
    n_jobs=-1
)

# Настройка SBS (Backward Selection)
sbs_selector = SFS(
    rf,
    k_features=10,             # Условие: 10 лучших признаков
    forward=False,             # Backward Selection
    floating=False,            # Выключено
    scoring='roc_auc',         # Основная метрика
    cv=4,                      # Кросс-валидация
    n_jobs=-1
)

# 7. ОБУЧЕНИЕ СЕЛЕКТОРОВ
# Важно: отбор проводим только на обучающем наборе данных (X_train)
print("Запуск SFS (Forward)...")
sfs_selector = sfs_selector.fit(X_train, y_train)

print("Запуск SBS (Backward)...")
sbs_selector = sbs_selector.fit(X_train, y_train)

# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# Получаем имена признаков через атрибут k_feature_names_
top_sfs = list(sfs_selector.k_feature_names_)
top_sbs = list(sbs_selector.k_feature_names_)

print(f"Top SFS features: {top_sfs}")
print(f"Top SBS features: {top_sbs}")

#--------------------------------------

sfs_df = pd.DataFrame.from_dict(sfs_selector.get_metric_dict()).T
sbs_df = pd.DataFrame.from_dict(sbs_selector.get_metric_dict()).T 

os.makedirs(FS_ASSETS, exist_ok=True)

sfs_df.to_csv(f"{FS_ASSETS}/sfs.csv")
sbs_df.to_csv(f"{FS_ASSETS}/sbs.csv") 

#--------------------------------------

fig = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()

plt.savefig(f"{FS_ASSETS}/sfs.png")
plt.show()

#--

fig = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')

plt.title('Sequential Backward Selection (w. StdDev)')
plt.grid()

plt.savefig(f"{FS_ASSETS}/sbs.png")
plt.show()

#------------------------------------

interc_features = list(set(top_sbs) & set(top_sfs))
union_features = list(set(top_sbs) | set(top_sfs))

#------------------------------------

with mlflow.start_run(run_name="feature_selection_results") as run:
    
    # 1. Логируем параметры (например, количество признаков)
    mlflow.log_param("num_interc_features", len(interc_features))
    mlflow.log_param("num_union_features", len(union_features))
    
    # 2. Сохраняем все содержимое папки FS_ASSETS в S3
    # Это отправит sbs.csv, sbs.png, sfs.csv и sfs.png одним махом
    mlflow.log_artifacts(FS_ASSETS, artifact_path="feature_selection_plots")
    
    print(f"Все артефакты из {FS_ASSETS} успешно загружены в S3.")
    print(f"Run ID: {run.info.run_id}")