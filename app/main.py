from typing import List
from joblib import load
from os import getenv
from functools import lru_cache

from pandas.core.frame import DataFrame

import boto3
import botocore
from fastapi import FastAPI
import pandas as pd

BUCKET_NAME = getenv("BUCKET_NAME")
FEATURES_KEY = getenv("FEATURES_KEY")
FEATURES = getenv("FEATURES")
MODEL_KEY = getenv("MODEL_KEY")
MODEL = getenv("MODEL")

app = FastAPI()

@lru_cache
def download_file(bucket_name: str, file_key: str, file_local: str):
    print(f"Downloading file from S3 {bucket_name}/{file_key}")
    s3 = boto3.resource("s3")
    try:
        s3.Bucket(bucket_name).download_file(file_key, file_local)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


def format_features(
    user_id: int, drop_columns: List[str], features: DataFrame
) -> DataFrame:
    print("Formatting features")
    features = features[features["id"] == user_id]
    features.sort_values(by=["loan_date"], ascending=False, inplace=True)
    first_row = features.head(1).copy()
    first_row.drop(drop_columns, axis="columns", inplace=True)
    return first_row


@app.get("/features/{user_id}")
def get_features(user_id: int):
    download_file(BUCKET_NAME, FEATURES_KEY, FEATURES)
    df = pd.read_parquet(FEATURES)
    if not user_id in df["id"].unique():
        return {"Message": "User not found"}
    df = format_features(user_id, ["id", "loan_date"], df)
    return {"features": df.to_dict("records")[0]}


@app.get("/predict/{user_id}")
def predict(user_id: int):
    download_file(BUCKET_NAME, FEATURES_KEY, FEATURES)
    download_file(BUCKET_NAME, MODEL_KEY, MODEL)
    df = pd.read_parquet(FEATURES)
    if not user_id in df["id"].unique():
        return {"Message": "User not found"}

    df = format_features(user_id, ["status", "loan_date"], df)
    print("Loading Model...")
    model = load(MODEL)
    print("Predicting...")
    prediction = model.predict(df)
    return {"prediction": prediction.item(0)}
