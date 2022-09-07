import pandas as pd
import numpy as np
from logger import Logger
import sys
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import pickle
import dvc.api

import io

app_logger = Logger("preprocessing.log").get_app_logger()

class Preprocessing:
    def __init__(self) -> None:
        try:
            self.logger = Logger("preprocessing.log").get_app_logger()
        except Exception:
            sys.exit(1)

    # open and read csv files given the path to the file
    def read_csv(self,csv_path,missing_values=[]) -> pd.DataFrame:
        
        try:
            df = pd.read_csv(csv_path, na_values=missing_values)
            print("file read as csv")
            self.logger.info(f"file read as csv from {csv_path}")
            return df
        except FileNotFoundError:
            print("file not found")
            self.logger.error(f"file not found, path:{csv_path}")

     # save the csv file into the given path
    def save_csv(self, df, csv_path):
        try:
            df.to_csv(csv_path, index=False)
            print('File Successfully Saved.!!!')
            self.logger.info(f"File Successfully Saved to {csv_path}")

        except Exception:
            print("Save failed...")
            self.logger.error(f"saving failed")

        return df
     
     # retrieve data from the remote path given the data version(tag) 
    def get_data_from_dvc(tag, path='../data/train.csv', repo='../'):#https://github.com/YohansSamuel/pharmaceutical_sales_prediction
        rev = tag
        data_url = dvc.api.read(path=path, repo=repo, rev=rev)
        df = pd.read_csv(io.StringIO(data_url),sep=",")
        app_logger.info(f"Read data from {path}, version {tag}")
        return df
    
    # save a trained model in the given path
    def save_model(self, file_name, model):
        with open(f"../models/{file_name}.pkl", "wb") as f:
            self.logger.info(f"Model dumped to {file_name}.pkl")
            pickle.dump(model, f)
    
    # read a trained model from the given path
    def read_model(self, file_name):
        with open(f"../models/{file_name}.pkl", "rb") as f:
            self.logger.info(f"Model loaded from {file_name}.pkl")
            return pickle.load(f)
    
     # display a missing value of each column in percentage
    def percent_missing(self, df: pd.DataFrame) -> float:
        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMissing = missingCount.sum()
        return round((totalMissing / totalCells) * 100, 2)
    
    # display a missing value of a given column in percentage
    def percent_missing_for_col(self, df: pd.DataFrame, col_name: str) -> float:
        total_count = len(df[col_name])
        if total_count <= 0:
            return 0.0
        missing_count = df[col_name].isnull().sum()

        return round((missing_count / total_count) * 100, 2)
        
    # get missing data percentage for every column
    def get_missing_data_percentage(self, df):
        
        try:
            self.logger.info('Getting Missing Data Percentage')
            total = df.isnull().sum().sort_values(ascending=False)
            percent_1 = total/df.isnull().count()*100
            percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
            missing_data = pd.concat(
                [total, percent_2], axis=1, keys=['Total', '%'])
            return missing_data
        except Exception:
            self.logger.exception('Failed to Get Missing Data Percentage')
            sys.exit(1)
    
    # get the numerical columns
    def get_numerical_columns(self, df):
        """Get numerical columns from dataframe."""
        try:
            self.logger.info('Getting Numerical Columns from Dataframe')
            num_col = df.select_dtypes(
                exclude="object").columns.tolist()
            return num_col
        except Exception:
            self.logger.error(f"fetchig numerical columns failed")
            sys.exit(1)
    # get the categorical columns
    def get_categorical_columns(self, df):
        """Get categorical columns from dataframe."""
        try:
            self.logger.info('Getting Categorical Columns from Dataframe')
            return df.select_dtypes(
                include="object").columns.tolist()
        except Exception:
            self.logger.exception('fetchig categorical columns failed')
            sys.exit(1)
    # convert a given column datatype into a datetime
    def convert_to_datetime(self, df, column):
        """Convert column to datetime."""
        try:
            self.logger.info('Converting column to Datetime')
            df[column] = pd.to_datetime(df[column])
            return df
        except Exception:
            self.logger.exception('Failed to convert column to Datetime')
            sys.exit(1)
    # join two given dataframes
    def join_dataframes(self, df1, df2, on, how="inner"):
        """Join two dataframes."""
        try:
            self.logger.info('Joining two Dataframes')
            return pd.merge(df1, df2, on=on)
        except Exception:
            self.logger.exception('joining dataframes failed')
            sys.exit(1)

    # extract the required fields from the timestamp column
    def extract_fields_date(self, df, date_column):
        try:
            self.logger.info('Extracting Fields from Date Column')
            df['Year'] = df[date_column].dt.year
            df['Month'] = df[date_column].dt.month
            df['Day'] = df[date_column].dt.day
            df['DayOfWeek'] = df[date_column].dt.dayofweek
            df['weekday'] = df[date_column].dt.weekday
            df['weekofyear'] = df[date_column].dt.weekofyear
            df['weekend'] = df[date_column].apply(self.is_weekend)
            return df
        except Exception:
            self.logger.exception('Failed to Extract Fields from Date Column')
            sys.exit(1)

    # returning the number of rows columns and column information
    def get_info(self,df):
        row_count, col_count = df.shape
    
        print(f"Number of rows: {row_count}")
        print(f"Number of columns: {col_count}")
        print("================================")

        return (row_count, col_count), df.info()

    # count values for a given column
    def get_count(self, df,column_name):
        return pd.DataFrame(df[column_name].value_counts())