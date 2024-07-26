import unittest
from wine_model.data import load_data
from wine_model.model import train_model

class TestWineModel(unittest.TestCase):
    
    def test_load_data(self):
        data = load_data()
        self.assertEqual(data.shape[0], 178)
        self.assertIn('target', data.columns)
    
    def test_train_model(self):
        data = load_data()
        model, accuracy = train_model(data)
        self.assertGreater(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()