import pandas as pd
from logger import Logger
import sys

class Preprocessing:
    def __init__(self) -> None:
        try:
            self.logger = Logger("logger.log").get_app_longer()
            self.logger.info("class is successfully instantiated")
        except Exception:
            self.logger.info("class is not instantiated")
            sys.exit(1)

    def read_csv(self,path) -> pd.DataFrame:
        # open and read csv files given the path to the file
        return pd.read_csv(path)
