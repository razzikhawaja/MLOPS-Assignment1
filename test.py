import unittest
import json
from app import app

class TestHousePricePredictionAPI(unittest.TestCase):

    def setUp(self):
        # Set up test client for Flask app
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        # Test if the homepage loads correctly
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)

    def test_prediction(self):
        # Test the predict endpoint with example data
        example_data = {
            'MedInc': 8.5,
            'HouseAge': 15.0,
            'AveRooms': 6.0,
            'AveBedrms': 1.0,
            'Population': 500.0,
            'AveOccup': 3.0,
            'Latitude': 37.0,
            'Longitude': -120.0
        }

        result = self.app.post('/predict', data=example_data)
        self.assertEqual(result.status_code, 200)

        # Check if the prediction returns a valid response
        self.assertIn(b'prediction', result.data)

if __name__ == '__main__':
    unittest.main()
