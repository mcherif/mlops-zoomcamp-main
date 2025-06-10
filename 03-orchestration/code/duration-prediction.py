#!/usr/bin/env python
# coding: utf-8
import pickle
from pathlib import Path

import pandas as pd
#import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
#from mlflow.entities import ViewType

import os
import argparse


import prefect
from prefect import task, flow

# This script trains a linear regression model to predict taxi trip duration
# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models') # local directory to save artidacts
models_folder.mkdir(exist_ok=True)

@task(name='Read Dataframe', retries=3, retry_delay_seconds=2, log_prints=True)
def read_dataframe(year, month, color):
    print(f"Running training for year: {year}, month: {month}, color: {color}")
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    
    print(f"Data shape: {df.shape}")
    #df = df.head(100_000)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    print(f"Data shape after transformation: {df.shape}")
    
    # Keep only the columns needed for modeling to save memory as I was getting memory errors
    used_columns = ['PULocationID', 'DOLocationID', 'trip_distance', 'duration']
    df = df[used_columns]

    return df

@task(name='Add features', retries=3, retry_delay_seconds=2, log_prints=True)
def prepare_features(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task(log_prints=True)
def train_lr(X_train, y_train, dv):

    with mlflow.start_run() as run:

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        
        mlflow.log_metric("intercept", lr.intercept_)
        mlflow.log_metric("rmse", rmse)
        
        print(f"RMSE: {rmse}")
        print(f"Intercept: {lr.intercept_}")

        Path('models').mkdir(exist_ok=True)
        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('models/preprocessor.b')

        # Ensure model is saved with support for sparse input
        mlflow.sklearn.log_model(lr, artifact_path='model')
        #mlflow.lrlog_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

@task(name='Register model', retries=3, retry_delay_seconds=2, log_prints=True)
def register_model(run_id):
    model_path = 'models/preprocessor.b'
    with mlflow.start_run(run_id=run_id):
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path)
            mlflow.log_metric("model_size_bytes", model_size)
            print(f"Model size (bytes): {model_size}")
        client = MlflowClient()

        experiment = client.get_experiment_by_name('nyc-taxi-experiment')

        model_name = f"trained-model-{run_id[-4:]}"
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name
        )

@flow(name='Main flow')
def run(year, month, color='yellow'):
    # Read the training data
    df_train = read_dataframe(year=year, month=month, color=color)

    # Prepare features
    X_train, dv = prepare_features(df_train)
    y_train = df_train['duration'].values

    # Train linear regression model and log to MLflow
    run_id = train_lr(X_train, y_train, dv)
    
    # Register the trained model in MLflow Model Registry
    register_model(run_id)
    
    print(f"MLflow run_id: {run_id}")
    return run_id
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--color', type=str, required=True, help='yellow (default) or green (specifies taxi data)')
    args = parser.parse_args()
    
    run_id = run(year=args.year, month=args.month, color=args.color)
    # Save the run_id to a file
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)