import pandas as pd
from logger import Logger
import sys
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import pickle
import dvc.api


app_logger = Logger("preprocessing.log").get_app_logger()

class Preprocessing:
    def __init__(self) -> None:
        try:
            pass
        except Exception:
            sys.exit(1)

    def read_csv(self,csv_path,missing_values=[]) -> pd.DataFrame:
        # open and read csv files given the path to the file
        try:
            df = pd.read_csv(csv_path, na_values=missing_values)
            print("file read as csv")
            self.logger.info(f"file read as csv from {csv_path}")
            return df
        except FileNotFoundError:
            print("file not found")
            self.logger.error(f"file not found, path:{csv_path}")

    def save_csv(self, df, csv_path):
        try:
            df.to_csv(csv_path, index=False)
            print('File Successfully Saved.!!!')
            self.logger.info(f"File Successfully Saved to {csv_path}")

        except Exception:
            print("Save failed...")
            self.logger.error(f"saving failed")

        return df

    def get_data_from_remote(tag, path='data/train.csv', repo='https://github.com/YohansSamuel/pharmaceutical_sales_prediction'):
        rev = tag
        data_url = dvc.api.get_url(path=path, repo=repo, rev=rev)
        df = pd.read_csv(data_url)
        app_logger.info(f"Read data from {path}, version {tag}")

        return df

    
