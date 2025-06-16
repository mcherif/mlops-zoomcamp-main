#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import argparse
import boto3  # <-- Add this import
import os

def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, bucket, s3_file)
    print(f"Uploaded {local_file} to s3://{bucket}/{s3_file}")

# Function to read and preprocess the data
def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    # Define the categorical features
    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def load_model(model_file='model.bin'):
    # Load the pre-trained model and DictVectorizer
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def run(year, month, model_file='model.bin', bucket=None):
    # Load the model and DictVectorizer
    dv, model = load_model(model_file)

    print(f"Loaded model for year: {year}, month: {month}")
    print(f"Model file: {model_file}")

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(f"Reading Input file: {input_file}")

    # Read and preprocess the data
    df = read_data(input_file)
    # TODO: improve by removing this (Redefine the categorical features)
    categorical = ['PULocationID', 'DOLocationID']
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Calculate and print the standard deviation of the predictions
    y_pred_std = np.std(y_pred)
    print(f"Standard deviation of y_pred: {y_pred_std}")

    # Create a ride_id column in the format 'YYYY/MM_ride_index'
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index  .astype('str')


    df_result = df[['ride_id']].copy()
    df_result['prediction'] = y_pred

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
    return str(mean_pred)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run a pre-trained model to predict taxi trip duration for a given year and month.')
    print("Example usage: python starter.py --year 2023 --month 3  --bucket my-bucket")
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--bucket', type=str, required=False, help='S3 bucket to upload the result')
    args = parser.parse_args()
    
    # return the mean predicted duration and upload output to s3
    mean_pred = run(year=args.year, month=args.month, bucket=args.bucket)
