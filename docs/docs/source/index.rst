.. Housing Price Prediction documentation master file, created by
   sphinx-quickstart on Thu Aug 18 16:14:30 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Housing Price Prediction's documentation!
====================================================

Ingest Data
------------------
.. automodule:: my_package.ingest_data
    :members:
.. autoclass:: Ingest
   :members: fetch_housing_data, load_housing_data, split_train_test

Train
-----
.. automodule:: my_package.train
    :members:

Evaluate
--------
.. automodule:: my_package.score
    :members:
.. autoclass:: Score
    :members: evaluate

.. toctree::
   :maxdepth: 3
   :caption: Contents: