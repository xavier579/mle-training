import unittest
from pathlib import Path

class Test(unittest.TestCase):

    def test_data_downloaded(self):
        # ...
        path = Path("../../data/raw/housing.csv")
        self.assertEquals((str(path), path.is_file()), (str(path), True))

    def test_train_file(self):
        # ...
        path = Path("../../data/processed/train.csv")
        self.assertEquals((str(path), path.is_file()), (str(path), True))

    def test_val_file(self):
        # ...
        path = Path("../../data/processed/val.csv")
        self.assertEquals((str(path), path.is_file()), (str(path), True))

    def test_model_file(self):
        # ...
        path = Path("../../artifacts/RandomForest_grid/model.pkl")
        self.assertEquals((str(path), path.is_file()), (str(path), True))

if __name__ == "__main__":
    unittest.main(verbosity=2)
