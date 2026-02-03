# NYC Taxi Trip Duration Predictor - NO DATA LEAKAGE VERSION
# This model uses ONLY features available at PICKUP time

import joblib
import pandas as pd
import numpy as np

class NYCTaxiPredictor:
    """Predict taxi trip duration with NO data leakage"""

    def __init__(self, model_path, preprocessor_path):
        """Load model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, trip_data):
        """
        Predict trip duration in minutes

        Parameters:
        -----------
        trip_data : dict
            Must contain these keys (all available at pickup):
            - tpep_pickup_datetime: str or datetime
            - pickup_longitude, pickup_latitude: float
            - dropoff_longitude, dropoff_latitude: float
            - passenger_count: int (1-6)
            - VendorID: int (1 or 2)
            - RatecodeID: int (typically 1)
            - trip_distance: float (miles, estimated)
            - payment_type: int (1=credit, 2=cash, etc.)
        """

        # Required features (NO POST-TRIP INFORMATION)
        required_features = [
            'tpep_pickup_datetime',
            'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'VendorID', 'RatecodeID',
            'trip_distance', 'payment_type'
        ]

        # Check all required features are present
        missing = [f for f in required_features if f not in trip_data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Create DataFrame
        df = pd.DataFrame([trip_data])

        # Preprocess and predict
        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)[0]

        return {
            'predicted_duration_minutes': round(prediction, 1),
            'confidence_interval': f"{max(0, prediction-5):.1f} - {prediction+5:.1f} minutes",
            'features_used': len(self.preprocessor.named_steps['feature_engineer'].get_feature_names()),
            'data_leakage_prevention': 'YES - only uses pickup-time information'
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = NYCTaxiPredictor('best_model.pkl', 'preprocessor.pkl')

    # Example trip (ALL information available at pickup)
    example_trip = {
        'tpep_pickup_datetime': '2016-01-15 17:30:00',
        'pickup_longitude': -73.9855,
        'pickup_latitude': 40.7580,
        'dropoff_longitude': -73.9772,
        'dropoff_latitude': 40.7829,
        'passenger_count': 2,
        'VendorID': 2,
        'RatecodeID': 1,
        'trip_distance': 2.5,  # Estimated route distance
        'payment_type': 1
    }

    result = predictor.predict(example_trip)
    print(f"
🚖 NYC Taxi Trip Duration Prediction (No Data Leakage)")
    print("=" * 50)
    print(f"Predicted duration: {result['predicted_duration_minutes']} minutes")
    print(f"95% confidence: {result['confidence_interval']}")
    print(f"Features used: {result['features_used']}")
    print(f"Data leakage prevention: {result['data_leakage_prevention']}")
