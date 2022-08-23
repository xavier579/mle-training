"""
ingest_data.py
"""
import argparse
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib

# import logging
# import logging.config
from sklearn.model_selection import train_test_split


class Ingest:

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "../../data"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(self, housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        """
        Download the housing dataset.

        Parameters
        ----------
        housing_url : str
            URL to download data
        housing_path : str
            Path where data will be stored

    .. highlight:: rst
    .. code-block:: python

        os.makedirs(housing_path, exist_ok=True)
        housing_path_raw = os.path.join(housing_path, "raw")
        tgz_path = os.path.join(housing_path_raw, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path_raw)
        housing_tgz.close()

        """
        os.makedirs(housing_path, exist_ok=True)
        housing_path_raw = os.path.join(housing_path, "raw")
        tgz_path = os.path.join(housing_path_raw, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path_raw)
        housing_tgz.close()

    def load_housing_data(self, housing_path):
        """
        Load the housing data into a DataFrame

    .. highlight:: rst
    .. code-block:: python

        csv_path = os.path.join(housing_path, "raw", "housing.csv")
        return pd.read_csv(csv_path)

        """
        csv_path = os.path.join(housing_path, "raw", "housing.csv")
        return pd.read_csv(csv_path)


    def split_train_test(self, housing, opt):
        """
        Split the data into train and test and save in processed folder

        Parameters
        ----------
        housing : DataFrame
            Dataframe containing all the data
        opt : Namespace
            Containing the arguements passed from ArgeParser

    .. highlight:: rst
    .. code-block:: python

        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
        train_set.to_csv(os.path.join(opt.data_folder, "processed", "train.csv"))
        test_set.to_csv(os.path.join(opt.data_folder, "processed", "val.csv"))

        """
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
        train_set = train_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        print(train_set.head())
        # exit(0)
        train_set.to_csv(os.path.join(opt.data_folder, "processed", "train.csv"), index=False)
        test_set.to_csv(os.path.join(opt.data_folder, "processed", "val.csv"), index=False)

    def parse_opt(self, known=False):
        """
        Parse the arguements from command line
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_folder",
            default=self.HOUSING_PATH,
            help="Folder path where dataset will be saved",
        )

        parser.add_argument("--log_level", default="DEBUG", help="Log level")
        parser.add_argument(
            "--log_path",
            default=None,
            help="Path to store logs, if empty logs would not be written to a file",
        )
        parser.add_argument(
            "--no_console_log",
            default="false",
            help="true if logs to be written to console, else false",
        )

        return parser.parse_known_args()[0] if known else parser.parse_args()

    def main(self, opt):
        self.fetch_housing_data(self.HOUSING_URL, opt.data_folder)
        housing = self.load_housing_data(opt.data_folder)
        self.split_train_test(housing, opt)


if __name__ == "__main__":
    ing = Ingest()
    opt = ing.parse_opt()
    ing.main(opt)
