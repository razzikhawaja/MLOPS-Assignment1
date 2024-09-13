import unittest
from app import app
import joblib
import pandas as pd

class TestHousePricePredictionAPI(unittest.TestCase):

    # Set up the Flask test client
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test the / route (home page)
    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction', response.data)  # Check if the word "Prediction" is in the page

    # Test the /predict endpoint with valid input
    def test_predict_valid_input(self):
        # Sample valid form data
        valid_input = {
            'MedInc': '8.32',
            'HouseAge': '41.0',
            'AveRooms': '6.98',
            'AveBedrms': '1.02',
            'Population': '322.0',
            'AveOccup': '2.55',
            'Latitude': '37.88',
            'Longitude': '-122.23'
        }

        # Send POST request to /predict
        response = self.app.post('/predict', data=valid_input)
        
        # Check if the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains a prediction (rendered in HTML)
        self.assertIn(b'Prediction', response.data)

    # Test the /predict endpoint with invalid input (missing fields)
    def test_predict_invalid_input(self):
        # Input data with missing fields
        invalid_input = {
            'MedInc': '8.32',
            'HouseAge': '41.0',
            # 'AveRooms' is missing
            'AveBedrms': '1.02',
            'Population': '322.0',
            'AveOccup': '2.55',
            'Latitude': '37.88',
            'Longitude': '-122.23'
        }

        response = self.app.post('/predict', data=invalid_input)
        
        # Expecting a 400 Bad Request due to missing fields
        self.assertNotEqual(response.status_code, 200)
        self.assertIn(b'Bad Request', response.data)  # Check for an appropriate error message


    # Test the model prediction directly
    def test_model_prediction(self):
        # Load the trained model
        model = joblib.load('house_price_model.pkl')

        # Define valid input for the model
        input_data = pd.DataFrame([{
            'MedInc': 8.32,
            'HouseAge': 41.0,
            'AveRooms': 6.98,
            'AveBedrms': 1.02,
            'Population': 322.0,
            'AveOccup': 2.55,
            'Latitude': 37.88,
            'Longitude': -122.23
        }])

        # Get prediction from the model
        prediction = model.predict(input_data)

        # Check if prediction is a single float value
        self.assertEqual(len(prediction), 1)
        self.assertIsInstance(prediction[0], (int, float))


if __name__ == '__main__':
    unittest.main()
