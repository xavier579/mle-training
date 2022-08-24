"""
train.py

"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from my_package import utils

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

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


def get_feature_names_from_column_transformer(col_trans):
    """Get feature names from a sklearn column transformer.

    The `ColumnTransformer` class in `scikit-learn` supports taking in a
    `pd.DataFrame` object and specifying `Transformer` operations on columns.
    The output of the `ColumnTransformer` is a numpy array that can used and
    does not contain the column names from the original dataframe. The class
    provides a `get_feature_names` method for this purpose that returns the
    column names corr. to the output array. Unfortunately, not all
    `scikit-learn` classes provide this method (e.g. `Pipeline`) and still
    being actively worked upon.

	NOTE: This utility function is a temporary solution until the proper fix is
    available in the `scikit-learn` library.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder as skohe

    # SimpleImputer has `add_indicator` attribute that distinguishes it from other transformers
    # Encoder had `get_feature_names` attribute that distinguishes it from other transformers
    # The last transformer is ColumnTransformer's 'remainder'
    col_name = []
    for transformer_in_columns in col_trans.transformers_:
        is_pipeline = 0
        raw_col_name = list(transformer_in_columns[2])

        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
            is_pipeline = 1
        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, str):
                if transformer == "passthrough":
                    names = transformer._feature_names_in[raw_col_name].tolist()

                elif transformer == "drop":
                    names = []

                else:
                    raise RuntimeError(
                        f"Unexpected transformer action for unaccounted cols :"
                        f"{transformer} : {raw_col_name}"
                    )

            elif isinstance(transformer, skohe):
                names = list(transformer.get_feature_names(raw_col_name))

            elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [
                    raw_col_name[idx] + "_missing_flag" for idx in missing_indicator_indices
                ]

                names = raw_col_name + missing_indicators

            else:
                names = list(transformer.get_feature_names())

        except AttributeError as error:
            names = raw_col_name
        if is_pipeline:
            names = [f"{transformer_in_columns[0]}_{col_}" for col_ in names]
        col_name.extend(names)

    return col_name


def process_df(df):
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
        [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),]
    )

    housing_prepared_numpyarray = full_pipeline.fit_transform(df)

    column_names = utils.get_feature_names_from_column_transformer(full_pipeline)
    house_prep = (pd.DataFrame(housing_prepared_numpyarray[:, :8], columns=column_names[:8])).join(
        (pd.DataFrame(housing_prepared_numpyarray[:, 8:11], columns=cols)).join(
            pd.DataFrame(housing_prepared_numpyarray[:, 11:], columns=column_names[8:])
        )
    )

    for i in range(len(house_prep.columns)):
        if "num" in house_prep.columns[i]:
            house_prep.rename(
                columns={house_prep.columns[i]: re.sub("num_", "", house_prep.columns[i])},
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
