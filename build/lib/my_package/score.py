import argparse
import json
import logging
import logging.config
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from my_package import log, utils


class Score:
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
        parser.add_argument(
            "--no_console_log",
            action=argparse.BooleanOptionalAction,
            help="true if logs to be written to console, else false",
        )

        return parser.parse_known_args()[0] if known else parser.parse_args()

    def evaluate(self):
        """
        Evaluate all the models performance by calculating MSE, MAE, RMSE

        Parameters
        ----------
        opt: Namespace
            Containing the arguements passed from ArgeParser

        Return
        ----------
        result: dict
            Containing model values of rmse, mse, mae

        """
        model_list = [
            name
            for name in os.listdir(opt.output_folder)
            if os.path.isdir(os.path.join(opt.output_folder, name))
        ]
        logger.debug("List of models to be evaluated: " + str(model_list))
        val_df = pd.read_csv(os.path.join(opt.data_folder, "processed", "val.csv"))
        X_test, y_test = utils.process_df(val_df)
        result = dict()
        for model_name in model_list:
            result[model_name] = dict()
            model = pickle.load(
                open(os.path.join(opt.output_folder, model_name, "model.pkl"), "rb")
            )
            final_predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, final_predictions)
            rmse = np.sqrt(mse)
            mae = mean_squared_error(y_test, final_predictions)
            result[model_name]["MSE"] = np.round(mse, 2)
            result[model_name]["RMSE"] = np.round(rmse, 2)
            result[model_name]["MAE"] = np.round(mae, 2)
            mlflow.log_metric(key="rmse", value=result[model_name]["RMSE"])
            mlflow.log_metrics(
                {"mae": result[model_name]["MAE"], "mse": result[model_name]["MSE"]}
            )
        logger.debug("Evaluation Matrix Result: " + str(result))
        logger.info("Evaluation Matrix Result: " + str(result))
        return result

    def evaluate_mlfow(self, model):
        """
        Evaluate single model performance by calculating MSE, MAE, RMSE

        Parameters
        ----------
        opt: Namespace
            Containing the arguements passed from ArgeParser

        Return
        ----------
        result: dict
            Containing model values of rmse, mse, mae

        """
        logger.debug("Model to be evaluated: " + str(model))
        val_df = pd.read_csv(os.path.join(opt.data_folder, "processed", "val.csv"))
        X_test, y_test = utils.process_df(val_df)
        result = dict()
        final_predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, final_predictions)
        rmse = np.sqrt(mse)
        mae = mean_squared_error(y_test, final_predictions)
        result["MSE"] = np.round(mse, 2)
        result["RMSE"] = np.round(rmse, 2)
        result["MAE"] = np.round(mae, 2)
        mlflow.log_metric(key="rmse", value=result["RMSE"])
        mlflow.log_metrics({"mae": result["MAE"], "mse": result["MSE"]})
        logger.debug("Evaluation Matrix Result: " + str(result))
        logger.info("Evaluation Matrix Result: " + str(result))
        return result

    def main(self):
        logger.info(f"Logging: Evaluate - Start")
        result = self.evaluate()
        with open(os.path.join(opt.output_folder, "result.json"), "w") as outfile:
            json.dump(result, outfile)
        return result


if __name__ == "__main__":

    scr = Score()
    print(scr.main())

