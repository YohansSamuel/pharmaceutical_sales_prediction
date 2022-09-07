import os
import sys
import unittest

import numpy
import pandas as pd
import pandas.api.types as ptypes
from pandas.api import types
from pandas._libs.tslibs.timestamps import Timestamp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from preprocessing import Preprocessing
from logger import Logger

class TestPreprocessing(unittest.TestCase):

     def setUp(self) -> pd.DataFrame:
        self.preprocess = Preprocessing()


     def test_get_categorical_columns(self):
        """Test get categorical columns module."""
        df = self.preprocess.get_categorical_columns(self.preprocess)
        assert df == ['Date', 'StoreType', 'Assortment']
     

     def test_convert_to_datetime(self):
        df = pd.DataFrame({'col1': ["2018-01-05", "2018-01-06"]})
        df = self.cleaner.convert_to_datetime(df, ['col1'])
        self.assertTrue(type(df['col1'].dtype == ptypes.DatetimeTZDtype))


if __name__ == '__main__':
    unittest.main()