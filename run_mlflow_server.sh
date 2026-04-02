# 1. Указываем значения БЕЗ знака $ в начале
export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=YCAJE3Nlz8iDILW5VTYM1ihQB
export AWS_SECRET_ACCESS_KEY=YCPjvS7uwhvJpUj3bKm8X-IX4QAwBIVsvX61IL44
export AWS_BUCKET_NAME=s3-student-mle-20250717-d331044a6c-freetrack

# 2. В команде запуска используем переменные через $ там, где мы их создали выше,
# но ПРЯМЫЕ данные (пароли, хосты) пишем как текст БЕЗ $.
mlflow server \
  --backend-store-uri postgresql://mle_20250717_d331044a6c_freetrack:b09eee00787a4a95b004369cc22c9a1c@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250717_d331044a6c \
  --default-artifact-root s3://$AWS_BUCKET_NAME \
  --no-serve-artifacts