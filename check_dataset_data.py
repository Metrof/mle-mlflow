import os

import psycopg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "users_churn" # таблица с данными в postgres 

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_UC" 
RUN_NAME = "eda"

ASSETS_DIR = "assets"

if not os.path.exists(ASSETS_DIR):
    os.mkdir(ASSETS_DIR)

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64

sns.set_style("white")
sns.set_theme(style="whitegrid")

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

print(df.head(2))

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(16.5, 12.5, forward=True)
fig.tight_layout(pad=4.0) # чуть увеличил отступ

# Мы используем строковые названия колонок
y_col = "customer_id"

# 1. Type
x_col = "type"
agg_df = df.groupby(x_col)[y_col].nunique().reset_index()
sns.barplot(data=agg_df, x=x_col, y=y_col, ax=axs[0, 0])
axs[0, 0].set_title(f'Count {y_col} by {x_col}')

# 2. Payment Method
x_col = "payment_method"
agg_df = df.groupby(x_col)[y_col].nunique().reset_index()
sns.barplot(data=agg_df, x=x_col, y=y_col, ax=axs[1, 0])
axs[1, 0].set_title(f'Count {y_col} by {x_col}')
axs[1, 0].tick_params(axis='x', rotation=45)

# 3. Internet Service
x_col = "internet_service"
agg_df = df.groupby(x_col)[y_col].nunique().reset_index()
sns.barplot(data=agg_df, x=x_col, y=y_col, ax=axs[0, 1])
axs[0, 1].set_title(f'Count {y_col} by {x_col}')

# 4. Gender
x_col = "gender"
agg_df = df.groupby(x_col)[y_col].nunique().reset_index()
sns.barplot(data=agg_df, x=x_col, y=y_col, ax=axs[1, 1])
axs[1, 1].set_title(f'Count {y_col} by {x_col}')

file_path = os.path.join(ASSETS_DIR, "cat_features_1.png")
plt.savefig(file_path)
plt.show()

#------------------------------------------

x = "customer_id"
binary_columns = [
    "online_security", 
    "online_backup", 
    "device_protection", 
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "senior_citizen",
    "partner",
    "dependents",
]
stat = ['count']

print(df.groupby(binary_columns).agg(stat[0])[x].reset_index().sort_values(by=x, ascending=False).head(10))

#------------------------------------------

# 1. Считаем количество 0 и 1 для каждой колонки
# apply(pd.Series.value_counts) применит подсчет к каждой колонке отдельно
heat_df = df[binary_columns].apply(pd.Series.value_counts).T
sns.heatmap(heat_df)
plt.savefig(os.path.join(ASSETS_DIR, 'cat_features_2_binary_heatmap'))

#------------------------------------------

x = "begin_date"
charges_columns = ["monthly_charges", "total_charges"]

# 1. Подготовка данных
df.dropna(subset=charges_columns, how='any', inplace=True)
stats = ["mean", "median", lambda x: x.mode().iloc[0]]

# 2. Агрегация для Monthly Charges
# Важно: передаем stats в agg
charges_monthly_agg = df.groupby(x)[charges_columns[0]].agg(stats).reset_index()
charges_monthly_agg.columns = [x, "monthly_mean", "monthly_median", "monthly_mode"]

# 3. Агрегация для Total Charges
charges_total_agg = df.groupby(x)[charges_columns[1]].agg(stats).reset_index()
charges_total_agg.columns = [x, "total_mean", "total_median", "total_mode"]

# 4. Создание фигуры (2 графика по вертикали)
fig, axs = plt.subplots(2, 1) # Исправлено на 2 графика
fig.set_size_inches(12.5, 10.5, forward=True) # Немного увеличим размер для наглядности
fig.tight_layout(pad=5.0)

# 5. Линейные графики для Monthly
sns.lineplot(data=charges_monthly_agg, x=x, y="monthly_mean", ax=axs[0], label='Mean')
sns.lineplot(data=charges_monthly_agg, x=x, y="monthly_median", ax=axs[0], label='Median')
sns.lineplot(data=charges_monthly_agg, x=x, y="monthly_mode", ax=axs[0], label='Mode')
axs[0].set_title(f"Statistics for {charges_columns[0]} by {x}")
axs[0].tick_params(axis='x', rotation=45)

# 6. Линейные графики для Total
sns.lineplot(data=charges_total_agg, x=x, y="total_mean", ax=axs[1], label='Mean')
sns.lineplot(data=charges_total_agg, x=x, y="total_median", ax=axs[1], label='Median')
sns.lineplot(data=charges_total_agg, x=x, y="total_mode", ax=axs[1], label='Mode')
axs[1].set_title(f"Statistics for {charges_columns[1]} by {x}")
axs[1].tick_params(axis='x', rotation=45)

# 7. Сохранение
file_path = os.path.join(ASSETS_DIR, "charges_by_date.png")
plt.savefig(file_path)
plt.show()

#------------------------------------------

import os
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Установка названия колонки
x = "target"

# 2. Агрегация: считаем количество вхождений каждого значения
# Мы группируем по target и считаем количество строк (size)
target_agg = df.groupby(x).size().reset_index(name='count')

# 3. Создание графика
# Так как мы работаем с одним графиком, axs не нужен, используем plt или sns напрямую
plt.figure(figsize=(8, 6))
sns.barplot(data=target_agg, x=x, y='count')

# 4. Установка заголовка
plt.title(f"{x} total distribution")

# 5. Сохранение (не забудь расширение .png)
file_path = os.path.join(ASSETS_DIR, "target_count.png")
plt.savefig(file_path)
plt.show()

#------------------------------------------

# 1. Установка переменных для анализа
x = "begin_date"
target = "target"
stat = ["count"]

# 2. Агрегация количества целей (только единиц) по датам
# Фильтруем df, оставляя только target == 1
target_agg_by_date = df[[x, target]].groupby([x]).agg(stat).reset_index()
target_agg_by_date.columns = target_agg_by_date.columns.droplevel()
target_agg_by_date.columns = [x, "target_count"]

# 3. Подсчёт количества клиентов для каждого значения цели (0 и 1)
target_agg = df[[x, target, 'customer_id']].groupby([x, target]).count().reset_index()

# 4. Расчёт конверсии по датам
# sum — это количество единиц, count — общее количество
conversion_agg = df[[x, target]].groupby([x])['target'].agg(['sum', 'count']).reset_index()
conversion_agg['conv'] = (conversion_agg['sum'] / conversion_agg['count']).round(2)

# 5. Расчет конверсии по датам и по полу
conversion_agg_gender = df[[x, target, 'gender']].groupby([x, 'gender'])[target].agg(['sum', 'count']).reset_index()
conversion_agg_gender['conv'] = (conversion_agg_gender['sum'] / conversion_agg_gender['count']).round(2)

# 6. Инициализация фигуры 2x2
fig, axs = plt.subplots(2, 2)
fig.tight_layout(pad=5.0) 
fig.set_size_inches(16.5, 12.5, forward=True)

# График 1: Общее количество целей (единиц)
sns.lineplot(data=target_agg_by_date, x=x, y="target_count", ax=axs[0, 0])
axs[0, 0].set_title("Target count by begin date")
axs[0, 0].tick_params(axis='x', rotation=45)

# График 2: Количество 0 и 1 (разделение через hue)
sns.lineplot(data=target_agg, x=x, y="customer_id", hue=target, ax=axs[0, 1])
axs[0, 1].set_title("Target count type by begin date")
axs[0, 1].tick_params(axis='x', rotation=45)

# График 3: Коэффициент конверсии общая
sns.lineplot(data=conversion_agg, x=x, y="conv", ax=axs[1, 0])
axs[1, 0].set_title("Conversion value")
axs[1, 0].tick_params(axis='x', rotation=45)

# График 4: Конверсия по полу
sns.lineplot(data=conversion_agg_gender, x=x, y="conv", hue="gender", ax=axs[1, 1])
axs[1, 1].set_title("Conversion value by gender")
axs[1, 1].tick_params(axis='x', rotation=45)

# 7. Сохранение
plt.savefig(os.path.join(ASSETS_DIR, 'target_by_date'))

#------------------------------------------

# определение списка столбцов с данными о платежах и целевой переменной
charges = ["monthly_charges", "total_charges"]
target = "target"

# инициализация фигуры для отображения гистограмм (2 графика вертикально)
fig, axs = plt.subplots(2, 1)
fig.tight_layout(pad=3.0) 
fig.set_size_inches(10.5, 10.5, forward=True)

# 1. Распределение ежемесячных платежей (Monthly Charges)
sns.histplot(
    data=df,              # датафрейм с данными
    x=charges[0],         # первый вид платежей
    hue=target,           # разделение данных по целевой переменной
    kde=True,             # включение оценки плотности распределения
    ax=axs[0]             # отобразить на первом подграфике
)
axs[0].set_title(f"{charges[0]} distribution")

# 2. Распределение общих платежей (Total Charges)
sns.histplot(
    data=df,              # датафрейм с данными
    x=charges[1],         # второй вид платежей
    hue=target,           # разделение данных по целевой переменной
    kde=True,             # включение оценки плотности распределения
    ax=axs[1]             # отобразить на втором подграфике
)
axs[1].set_title(f"{charges[1]} distribution")

# сохранение фигуры в файл
plt.savefig(os.path.join(ASSETS_DIR, 'chargest_by_target_dist'))




os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    print(run_id)
    mlflow.log_artifacts(ASSETS_DIR) 