from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine


class DataLoader(ABC):
    """ Signature for a data loader.

    Note that the return type is a tuple of two pandas DataFrames. The first is
    the training data, the second is the test data.
    """
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class CSVDataLoader(DataLoader):
    def __init__(self, path: Path):
        self.path = path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return pd.read_csv(self.path / "train.csv"), pd.read_csv(self.path / "test.csv")


class SQLDataLoader(DataLoader):
    def __init__(self, connection_string: str, train_table_name: str, test_table_name: str):
        self.engine = create_engine(connection_string)
        self.train_table_name = train_table_name
        self.test_table_name = test_table_name

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        query_train = f"""
            SELECT *
            FROM {self.train_table_name}
        """
        query_test = f"""
            SELECT *
            FROM {self.test_table_name}
        """

        with self.engine.connect() as connection:
            return pd.read_sql(query_train, connection), pd.read_sql(query_test, connection)
