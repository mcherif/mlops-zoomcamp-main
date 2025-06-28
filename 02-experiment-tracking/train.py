import os
import pickle
import click
#import numpy as np
import mlflow
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-greentaxi-experiment")

# Enable autologging to track parameters, metrics, and models
mlflow.sklearn.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():

        # Load the preprocessed data
        # from 02-experiment-tracking/preprocess_data.py
        # The data is expected to be in the form of pickled files
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)


if __name__ == '__main__':
    run_train()