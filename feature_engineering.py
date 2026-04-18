import os
import psycopg

import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, 
    SplineTransformer, 
    QuantileTransformer, 
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)

from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "users_churn"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "users_churn_exp"
RUN_NAME = "preprocessing" 

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64

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

#---------------------------------------------------

# определение категориальных колонок, которые будут преобразованы
cat_columns = ["type", "payment_method", "internet_service", "gender"]

# создание объекта OneHotEncoder для преобразования категориальных переменных
# auto - автоматическое определение категорий
# ignore - игнорировать ошибки, если встречается неизвестная категория
# max_categories - максимальное количество уникальных категорий
# sparse_output - вывод в виде разреженной матрицы, если False, то в виде обычного массива
# drop="first" - удаляет первую категорию, чтобы избежать ловушки мультиколлинеарности

encoder_oh = OneHotEncoder(
    categories='auto',       
    drop='first',           
    max_categories=10,      
    handle_unknown='ignore',
    sparse_output=False      
)

# преобразование полученных признаков в DataFrame и установка названий колонок
# get_feature_names_out() - получение имён признаков после преобразования
encoded_features = encoder_oh.fit_transform(df[cat_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_oh.get_feature_names_out(cat_columns),
    index=df.index 
)

# конкатенация исходного DataFrame с новым DataFrame, содержащим закодированные категориальные признаки
# axis=1 означает конкатенацию по колонкам
obj_df = pd.concat([df, encoded_df], axis=1)

# print(obj_df.head(2))

#-----------------------------------------------

num_columns = ["monthly_charges", "total_charges"]
df[num_columns] = df[num_columns].fillna(df[num_columns].median())

num_df = df[num_columns].copy()

n_knots = 3
degree_spline = 4
n_quantiles=100
degree = 3
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'
subsample = None


# SplineTransformer
encoder_spl = SplineTransformer(n_knots=n_knots, degree=degree_spline)
encoded_features = encoder_spl.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_spl.get_feature_names_out(num_columns)
)
num_df = pd.concat([num_df, encoded_df], axis=1)


# QuantileTransformer
encoder_q = QuantileTransformer(n_quantiles=n_quantiles)
encoded_features = encoder_q.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_q.get_feature_names_out(num_columns)
)
encoded_df.columns = [col + f"_q_{n_quantiles}" for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)


# RobustScaler
encoder_rb = RobustScaler()
encoded_features = encoder_rb.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_rb.get_feature_names_out(num_columns)
)
encoded_df.columns = [col + f"_robust" for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)


# PolynomialFeatures
encoder_pol = PolynomialFeatures(degree=degree, include_bias=False)
encoded_features = encoder_pol.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_pol.get_feature_names_out(num_columns),
    index=df.index
)

encoded_df = encoded_df.drop(columns=num_columns)
encoded_df.columns = [col + "_poly" for col in encoded_df.columns]
num_df = pd.concat([num_df, encoded_df], axis=1)

# KBinsDiscretizer
encoder_kbd = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample)
encoded_features = encoder_kbd.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_kbd.get_feature_names_out(num_columns)
)
encoded_df.columns = [col + f"_bin" for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)


# print(num_df.head(2))

#-----------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_transformer = ColumnTransformer(
    transformers=[('spl', encoder_spl, num_columns), 
                  ('q', encoder_q, num_columns), 
                  ('rb', encoder_rb, num_columns), 
                  ('pol', encoder_pol, num_columns),
                  ('kbd', encoder_kbd, num_columns)]
)

categorical_transformer = Pipeline(steps=[('encoder', encoder_oh)])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_columns), 
        ('cat', categorical_transformer, cat_columns)], n_jobs=-1)

encoded_features = preprocessor.fit_transform(df)

transformed_df = pd.DataFrame(encoded_features, columns=preprocessor.get_feature_names_out())

df = pd.concat([df, transformed_df], axis=1)

# print(df.head(2))

#-----------------------------------------------

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
    mlflow.sklearn.log_model(preprocessor, "column_transformer") 