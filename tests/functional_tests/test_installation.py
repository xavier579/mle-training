import unittest
import my_package

class Test(unittest.TestCase):

    def test_package(self):
        self.predict=Predict()
        res=my_package.__package__
        self.assertEqual(res, 'my_package')

if __name__ == "__main__":
    unittest.main(verbosity=2)
