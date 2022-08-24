"""
train.py

"""
import argparse
import logging
import logging.config
import os
import pickle
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from my_package import log, utils
from my_package.score import Score

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class Train:
    """
    Class to train the models
    """

    scr = Score()

    def __init__(self) -> None:
        global opt
        global logger
        opt = self.parse_opt()
        if opt.log_path is not None:
            logger = log.configure_logger(
                log_file=os.path.join(opt.log_path, "house_prediction.log")
            )
        else:
            logger = logging
        if opt.no_console_log:
            logger.disabled = True

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

        """
        start_time = time.time()
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        logger.debug("Linear Regression model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(lr_model, "LinearRegModel")
        mlflow.sklearn.log_model(lr_model, "model")
        self.scr.evaluate_mlfow(lr_model)

    def train_tree(self, X_train, y_train):
        """
        Trains Decision Tree Regressor model

        Parameters
        ----------
        X_train: DataFrame
            Independent variables of Training dataset
        y_train: Series
            Dependent variables of Training dataset

        """
        start_time = time.time()
        max_depth = 6
        tree_reg = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree_reg.fit(X_train, y_train)
        logger.debug("Decision tree model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(tree_reg, "DecisionTreeModel")
        # Log parameter, metrics, and model to MLflow
        mlflow.log_param(key="max_depth", value=max_depth)
        mlflow.sklearn.log_model(tree_reg, "model")
        self.scr.evaluate_mlfow(tree_reg)

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

        """
        start_time = time.time()
        # mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
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
        best_param = rnd_search.best_params_
        mlflow.log_param(key="n_estimators", value=best_param["n_estimators"])
        mlflow.log_param(key="max_features", value=best_param["max_features"])
        mlflow.sklearn.log_model(final_model, "model")
        self.scr.evaluate_mlfow(final_model)

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

        """
        start_time = time.time()
        # mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
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

        cvres = grid_search.cv_results_
        # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        #     print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, X_train.columns), reverse=True)
        final_model = grid_search.best_estimator_
        logger.debug("Random Forest using GridSearch model trained")
        logger.debug(" Trained in: " + str(time.time() - start_time))
        self.save_model(final_model, "RandomForest_grid")
        best_param = grid_search.best_params_
        mlflow.log_param(key="n_estimators", value=best_param["n_estimators"])
        mlflow.log_param(key="max_features", value=best_param["max_features"])
        if "bootstrap" in best_param.keys():
            mlflow.log_param(key="bootstrap", value=best_param["bootstrap"])
        mlflow.sklearn.log_model(final_model, "model")
        self.scr.evaluate_mlfow(final_model)

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
        X_train, y_train = utils.process_df(train_df)
        self.train_LR(X_train, y_train)
        self.train_tree(X_train, y_train)
        self.forest_reg_rand(X_train, y_train)
        self.forest_reg_grid(X_train, y_train)
        logger.info("Training done for all models")

        """
        # Train all models
        train_df = pd.read_csv(os.path.join(opt.data_folder, "processed", "train.csv"))
        X_train, y_train = utils.process_df(train_df)
        with mlflow.start_run(
            run_name="CHILD_TRAIN",
            # experiment_id=experiment_id,
            description="Train",
            nested=True,
        ) as train_parent:
            mlflow.log_param("train_parent", "yes")
            with mlflow.start_run(
                run_name="CHILD_RUN_LR",
                # experiment_id=experiment_id,
                description="Train LR",
                nested=True,
            ) as child_train_run:
                mlflow.log_param("train_child", "yes")
                self.train_LR(X_train, y_train)
            with mlflow.start_run(
                run_name="CHILD_RUN_Tree",
                # experiment_id=experiment_id,
                description="Train Decision Tree",
                nested=True,
            ) as child_train_run:
                mlflow.log_param("train_child", "yes")
                self.train_tree(X_train, y_train)
            # with mlflow.start_run(
            #     run_name="CHILD_RUN_RF_rnd",
            #     # experiment_id=experiment_id,
            #     description="Train_RF_rnd",
            #     nested=True,
            # ) as child_train_run:
            #     mlflow.log_param("train_child", "yes")
            #     self.forest_reg_rand(X_train, y_train)
            # with mlflow.start_run(
            #     run_name="CHILD_RUN_RF_grid",
            #     # experiment_id=experiment_id,
            #     description="Train_RF_grid",
            #     nested=True,
            # ) as child_train_run:
            #     mlflow.log_param("train_child", "yes")
            #     self.forest_reg_grid(X_train, y_train)
        logger.info("Training done for all models")

        print("parent train run:")
        print("run_id: {}".format(train_parent.info.run_id))
        print("description: {}".format(train_parent.data.tags.get("mlflow.note.content")))
        print("--")

        # Search all child runs with a parent id
        query = "tags.mlflow.parentRunId = '{}'".format(train_parent.info.run_id)
        results = mlflow.search_runs(filter_string=query)
        print("train child runs:")
        print(results.columns)
        print(results)

    def main(self):
        global opt
        opt = self.parse_opt()
        global logger
        if opt.log_path is not None:
            logger = log.configure_logger(
                log_file=os.path.join(opt.log_path, "house_prediction.log")
            )
        else:
            logger = logging
        if opt.no_console_log:
            logger.disabled = True
            logger.info(f"Logging: Train - Start")
        self.train(opt)


if __name__ == "__main__":

    tr = Train()
    tr.main()
    print("Training Done")
