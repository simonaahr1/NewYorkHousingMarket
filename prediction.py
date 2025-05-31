import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from scripts import extract_borough_from_state, extract_zip_code, calculate_distances, encode_categorical


def load_resources():
    """
        Load models, scalers, and encoders

        Returns: all models and scalers
    """
    nn_model = load_model('models/neural_network_model.keras')
    rf_model = joblib.load('models/random_forest_model.pkl')
    lr_model = joblib.load('models/linear_model.pkl')

    scaler_X = joblib.load('scalers/scaler_X.pkl')
    scaler_y = joblib.load('scalers/scaler_y.pkl')

    return nn_model, rf_model, lr_model, scaler_X, scaler_y


# Process input data
def process_input_lr_rf(input_data):
    """
       Preprocess the data so it could fit the model
       Returns: preprocessed data ready for the model
    """
    df = pd.DataFrame([input_data])

    # Apply feature transformations
    df = extract_borough_from_state(df)
    df = extract_zip_code(df)
    df = calculate_distances(df)
    df = encode_categorical(df)

    # Select columns used in training
    selected_columns = ['LATITUDE', 'LONGITUDE', 'DIST_TO_CENTER', 'DIST_TO_CENTRAL_PARK',
                        'BEDS', 'BATH', 'PROPERTYSQFT', 'BOROUGH_FROM_STATE_ENCODED', 'SUBLOCALITY_ENCODED',
                        'LOCALITY_ENCODED', 'ADMINISTRATIVE_AREA_LEVEL_2_ENCODED', 'TYPE_ENCODED',
                        'BROKERTITLE_ENCODED', 'ZIP_CODE_ENCODED', 'LONG_NAME_ENCODED', 'STATE_ENCODED',
                        'STREET_NAME_ENCODED']

    df = df[selected_columns]

    return df

def process_input_nn(input_data, scaler_X):
    """
       Preprocess the data so it could fit the model
       Returns: preprocessed data ready for the model
    """
    df = pd.DataFrame([input_data])

    # Apply feature transformations
    df = extract_borough_from_state(df)
    df = extract_zip_code(df)
    df = calculate_distances(df)
    df = encode_categorical(df)

    # Select columns used in training
    selected_columns = ['LATITUDE', 'LONGITUDE', 'DIST_TO_CENTER', 'DIST_TO_CENTRAL_PARK',
                        'BEDS', 'BATH', 'PROPERTYSQFT', 'BOROUGH_FROM_STATE_ENCODED', 'SUBLOCALITY_ENCODED',
                        'LOCALITY_ENCODED', 'ADMINISTRATIVE_AREA_LEVEL_2_ENCODED', 'TYPE_ENCODED',
                        'BROKERTITLE_ENCODED', 'ZIP_CODE_ENCODED', 'LONG_NAME_ENCODED', 'STATE_ENCODED',
                        'STREET_NAME_ENCODED']

    df = df[selected_columns]

    # Scale the features
    scaled_features = scaler_X.transform(df)

    return pd.DataFrame(scaled_features, columns=scaler_X.feature_names_in_)


def make_prediction(model, processed_input, scaler_y, is_nn=False):
    """
        The function use the model and make prediction on the preprocessed data
        Returns: the predicted prices by all the model
    """
    predictions_scaled = model.predict(processed_input)

    if is_nn:
        predictions_log = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()[0]
        predictions = np.expm1(predictions_log)
    else:
        predictions =  np.expm1(predictions_scaled[0])

    return predictions


def predict_property_price(input_data):
    """
    Main function
    Returns: the predicted prices by all the models
    """
    nn_model, rf_model, lr_model, scaler_X, scaler_y = load_resources()
    processed_input_lr_rf = process_input_lr_rf(input_data)
    processed_input_nn = process_input_nn(input_data, scaler_X)

    price_nn = make_prediction(nn_model, processed_input_nn, scaler_y, is_nn=True)
    price_rf = make_prediction(rf_model, processed_input_lr_rf, scaler_y)
    price_lr = make_prediction(lr_model, processed_input_lr_rf, scaler_y)

    return price_nn, price_rf, price_lr


# Example input actual price ---1,049,000
input_data = {
    'STATE': 'New York, NY',
    'MAIN_ADDRESS': '342 Manhattan St, Staten Island, NY 10307',
    'LATITUDE': 40.5016233,
    'LONGITUDE': -74.2417531,
    'BEDS': 4,
    'BATH': 4,
    'PROPERTYSQFT': 2554,
    'BOROUGH_FROM_STATE': 'Staten Island',
    'SUBLOCALITY': 'Manhattan Street',
    'LOCALITY': 'Staten Island',
    'ADMINISTRATIVE_AREA_LEVEL_2': 'Richmond County',
    'TYPE': 'Multi-family home',
    'BROKERTITLE': 'Brokered by Homes R Us Realty of NY, Inc.',
    'ZIP_CODE': '10307',
    'LONG_NAME': '342 Manhattan St',
    'STREET_NAME': 'Manhattan Street'
}

# Run prediction
price_nn, price_rf, price_lr = predict_property_price(input_data)

print(f"Predicted price using Neural Network: ${price_nn:,.2f}")
print(f"Predicted price using Random Forest: ${price_rf:,.2f}")
print(f"Predicted price using Linear Regression: ${price_lr:,.2f}")




