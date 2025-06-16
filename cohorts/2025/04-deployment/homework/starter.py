#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import argparse
import boto3
import os
from prefect import task, flow
from typing import Optional

# Define the categorical features
CATEGORICAL = ['PULocationID', 'DOLocationID']

# Function to read and preprocess the data
@task(log_prints=True, retries=2, retry_delay_seconds=5)
def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df

@task(log_prints=True)
def load_model(model_file='model.bin'):
    # Load the pre-trained model and DictVectorizer
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

@task(log_prints=True, retries=2)
def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, bucket, s3_file)
    print(f"Uploaded {local_file} to s3://{bucket}/{s3_file}")

@task(log_prints=True)
def make_predictions(df, dv, model, year, month):
    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f"Standard deviation of y_pred: {np.std(y_pred)}")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = df[['ride_id']].copy()
    df_result['prediction'] = y_pred
    return df_result, y_pred

@flow(name="Taxi Duration Prediction Flow")
def main_flow(year: int, month: int, model_file: str = 'model.bin', bucket: Optional[str] = None):
    # Load the model and DictVectorizer
    dv, model = load_model(model_file)

    print(f"Loaded model for year: {year}, month: {month}")
    print(f"Model file: {model_file}")

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(f"Reading Input file: {input_file}")

    # Read and preprocess the data
    df = read_data(input_file)
    
    df_result, y_pred = make_predictions(df, dv, model, year, month)

    output_file = f'yellow_tripdata_{year:04d}-{month:02d}_predicted.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    print(f"Output saved to {output_file}")

    # Upload to S3 if bucket is provided
    if bucket:
        s3_file = os.path.basename(output_file)
        upload_to_s3(output_file, bucket, s3_file)

    # return the mean predicted duration
    mean_pred = y_pred.mean()
    print(f"Mean predicted duration: {mean_pred}")
    return mean_pred

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run a pre-trained model to predict taxi trip duration for a given year and month.')
    print("Example usage: python starter.py --year 2023 --month 3  --bucket my-s3-bucket")
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--bucket', type=str, required=False, help='S3 bucket to upload the result')
    args = parser.parse_args()
    main_flow(year=args.year, month=args.month, bucket=args.bucket)