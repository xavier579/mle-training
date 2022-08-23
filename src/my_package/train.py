"""
train.py

"""
import argparse
import logging
import logging.config
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from my_package import log, utils

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def columns(self):
        if self.add_bedrooms_per_room:
            cols = [
                "rooms_per_household",
                "population_per_household",
                "bedrooms_per_room",
            ]
        else:
            cols = ["rooms_per_household", "population_per_household"]
        return cols


class Train:
    """
    Class to train the models
    """

    def parse_opt(known=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder", default="../../data", help="Path for input data")
        parser.add_argument(
            "--output_folder", default="../../artifacts", help="Path for model output"
        )

        parser.add_argument("--log_level", default="DEBUG", help="Log level")
        parser.add_argument(
            "--log_path",
            default=None,
            help="Path to store logs, if empty logs would not be written to a file",
        )
        # parser.add_argument('--no_console_log', default='true', help='true if logs to be written to console, else false')
        parser.add_argument(
            "--no_console_log",
            action=argparse.BooleanOptionalAction,
            help="true if logs to be written to console, else false",
        )

        return parser.parse_known_args()[0] if known else parser.parse_args()

    def income_cat_proportions(self, data):
        return data["income_cat"].value_counts() / len(data)

    def save_model(self, model, model_folder=""):
        """
        Save model to the desired folder

        Parameters
        ----------
        model : Model object
            Dataframe containing all the data
        opt : Namespace
            Containing the arguements passed from ArgeParser
        model_folder: str
            Path where model will be saved

    .. highlight:: rst
    .. code-block:: python

        model_path = os.path.join(opt.output_folder, model_folder)
        # print("Model Path", model_path)
        os.makedirs(os.path.join(model_path), exist_ok=True)
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)


        """
        model_path = os.path.join(opt.output_folder, model_folder)
        # print("Model Path", model_path)
        os.makedirs(os.path.join(model_path), exist_ok=True)
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

    def train_LR(self, X_train, y_train):
        """
        Trains Linear Regression model

        Parameters
        ----------
        X_train: DataFrame
            Independent variables of Training dataset
        y_train: Series
            Dependent variables of Training dataset

    .. highlight:: rst
    .. code-block:: python

        start_time = time.time()
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        logger.debug("Linear Regression model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(lr_model, "LinearRegModel")

        """
        start_time = time.time()
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        logger.debug("Linear Regression model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(lr_model, "LinearRegModel")

    def train_tree(self, X_train, y_train):
        """
        Trains Decision Tree Regressor model

        Parameters
        ----------
        X_train: DataFrame
            Independent variables of Training dataset
        y_train: Series
            Dependent variables of Training dataset

    .. highlight:: rst
    .. code-block:: python

        start_time = time.time()
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(X_train, y_train)
        logger.debug("Decision tree model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(tree_reg, "DecisionTreeModel")

        """
        start_time = time.time()
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(X_train, y_train)
        logger.debug("Decision tree model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(tree_reg, "DecisionTreeModel")

    def forest_reg_rand(self, X_train, y_train):
        """
        Trains RandomForest Regressor model and
        choosing best hyperparameters usng RandomizedSearchCV

        Parameters
        ----------
        X_train: DataFrame
            Independent variables of Training dataset
        y_train: Series
            Dependent variables of Training dataset

    .. highlight:: rst
    .. code-block:: python

        start_time = time.time()
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(X_train, y_train)
        final_model = rnd_search.best_estimator_
        logger.debug("Random Forest using randomizedSearch model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(final_model, "RandomForest_rand")


        """
        start_time = time.time()
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(X_train, y_train)
        final_model = rnd_search.best_estimator_
        logger.debug("Random Forest using randomizedSearch model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(final_model, "RandomForest_rand")

    def forest_reg_grid(self, X_train, y_train):
        """
        Trains RandomForest Regressor model and
        choosing best hyperparameters usng GridSearchCV

        Parameters
        ----------
        X_train: DataFrame
            Independent variables of Training dataset
        y_train: Series
            Dependent variables of Training dataset

    .. highlight:: rst
    .. code-block:: python

        start_time = time.time()
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        grid_search.best_params_
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, X_train.columns), reverse=True)
        final_model = grid_search.best_estimator_
        logger.debug("Random Forest using GridSearch model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(final_model, "RandomForest_grid")


        """
        start_time = time.time()
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        grid_search.best_params_
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, X_train.columns), reverse=True)
        final_model = grid_search.best_estimator_
        logger.debug("Random Forest using GridSearch model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(final_model, "RandomForest_grid")



    def process_df(self, df):
        """
        Process the input variables

        Parameters
        ----------
        opt: Namespace
            Containing the arguements passed from ArgeParser
        df: DataFrame
            The housing price dataset

        Return
        ----------
        X: Processed independent variables
        y: Processed dependent variables

        """
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"].copy()
        # imputer = SimpleImputer(strategy="median")


        housing_num = df.drop("ocean_proximity", axis=1)

        attr_adder = CombinedAttributesAdder()
        cols = attr_adder.columns()

        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("attribs_adder", CombinedAttributesAdder()),
                ("std_scaler", StandardScaler()),
            ]
        )

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer(
            [
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ]
        )

        housing_prepared_numpyarray = full_pipeline.fit_transform(df)

        column_names = utils.get_feature_names_from_column_transformer(full_pipeline)
        logger.info("ColumnTransformer Columns: "+ str(column_names))
        logger.info("CombinedAttributesAdder Columns: "+ str(cols))
        house_prep = (
            pd.DataFrame(housing_prepared_numpyarray[:, :8], columns=column_names[:8])
        ).join(
            (pd.DataFrame(housing_prepared_numpyarray[:, 8:11], columns=cols)).join(
                pd.DataFrame(housing_prepared_numpyarray[:, 11:], columns=column_names[8:])
            )
        )

        for i in range(len(house_prep.columns)):
            if "num" in house_prep.columns[i]:
                house_prep.rename(
                    columns={
                        house_prep.columns[i]: re.sub("num_", "", house_prep.columns[i])
                    },
                    inplace=True,
                )

        X_final = house_prep


        # housing_tr = pd.DataFrame(X_impute, columns=housing_num.columns, index=X.index)
        # housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
        # housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
        # housing_tr["population_per_household"] = (
        #     housing_tr["population"] / housing_tr["households"]
        # )
        # housing_cat = X[["ocean_proximity"]]
        # X = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
        return X_final, y

    def train(self, opt):
        """
        Trains ML models

        Parameters
        ----------
        opt: Namespace
            Containing the arguements passed from ArgeParser

    .. highlight:: rst
    .. code-block:: python

        train_df = pd.read_csv(os.path.join(opt.data_folder, "processed", "train.csv"))
        X_train, y_train = self.process_df(train_df)
        self.train_LR(X_train, y_train)
        self.train_tree(X_train, y_train)
        self.forest_reg_rand(X_train, y_train)
        self.forest_reg_grid(X_train, y_train)
        logger.info("Training done for all models")

        """
        # Train all models
        train_df = pd.read_csv(os.path.join(opt.data_folder, "processed", "train.csv"))
        X_train, y_train = self.process_df(train_df)
        self.train_LR(X_train, y_train)
        self.train_tree(X_train, y_train)
        self.forest_reg_rand(X_train, y_train)
        self.forest_reg_grid(X_train, y_train)
        logger.info("Training done for all models")

    def main(self, opt):
        logger.info(f"Logging: Train - Start")
        self.train(opt)


if __name__ == "__main__":

    tr = Train()
    global opt
    opt = tr.parse_opt()
    global logger
    if opt.log_path is not None:
        logger = log.configure_logger(log_file=os.path.join(opt.log_path, "house_prediction.log"))
    else:
        logger = logging
    if opt.no_console_log:
        logger.disabled = True
    tr.main(opt)
