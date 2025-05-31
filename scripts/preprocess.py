import pandas as pd
import numpy as np
import re
import category_encoders as ce
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Coordinates for key locations, you can add more if you wish
city_center = (40.7128, -74.0060)
central_park = (40.7851, -73.9683)


def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Returns:
        The data frame with the new features
    """
    return pd.read_csv(filepath)


def extract_borough_from_state(df):
    """
    Extract the first abbreviation/name before the comma in the STATE column.

    Returns:
        The data frame with the new features
    """

    def extract_before_comma(state_value):
        if pd.notnull(state_value):
            return state_value.split(',')[0].strip()
        return None

    df['BOROUGH_FROM_STATE'] = df['STATE'].apply(extract_before_comma)
    return df


def extract_zip_code(df):
    """
    Extract the ZIP code from the MAIN_ADDRESS column.

    This function scans each address in the 'MAIN_ADDRESS' column using a regular
    expression to find a standard 5-digit US ZIP code.

    Returns:
        The data frame with the new features
    """
    df['ZIP_CODE'] = df['MAIN_ADDRESS'].apply(
        lambda x: re.search(r'\b\d{5}\b', x).group() if re.search(r'\b\d{5}\b', x) else None)
    return df


def encode_categorical(df, fit=False):
    """
    Encode categorical features using target encoding.
    We use Target Encoding which converts categorical variables into numeric values by replacing each category with
    the average value of the target variable.
    This function is used during training and prediction. It is working the following way:
    If fit=True, train encoders and save them.
    If fit=False, load existing encoders and transform.

    Returns:
        The data frame with the encoded columns and store the encoders for the future prediction
    """

    categorical_columns = ['BOROUGH_FROM_STATE', 'BROKERTITLE', 'LOCALITY', 'LONG_NAME', 'STATE', 'STREET_NAME',
                           'SUBLOCALITY', 'TYPE', 'ZIP_CODE', 'ADMINISTRATIVE_AREA_LEVEL_2']

    if fit: # use when we are training
        encoders = {}
        # create encoders for every column
        for col in categorical_columns:
            encoder = ce.TargetEncoder(cols=[col])
            df[f'{col}_ENCODED'] = encoder.fit_transform(df[col], df['LOG_PRICE'])
            encoders[col] = encoder

        # Save the trained encoders to a file
        joblib.dump(encoders, 'scalers/categorical_encoders.pkl')
        print("Encoders trained and saved.")

    else:
        # Load encoders from file
        encoders = joblib.load('scalers/categorical_encoders.pkl')
        print("Encoders loaded for prediction.")

        for col in categorical_columns:
            df[f'{col}_ENCODED'] = encoders[col].transform(df[col])

    return df


def calculate_distances(df):
    """
    Calculate distances from key locations and add as new features.

    Returns:
        The data frame with the new features
    """
    df['DIST_TO_CENTER'] = df.apply(lambda row: haversine(city_center, (row['LATITUDE'], row['LONGITUDE'])), axis=1)
    df['DIST_TO_CENTRAL_PARK'] = df.apply(lambda row: haversine(central_park, (row['LATITUDE'], row['LONGITUDE'])),
                                          axis=1)

    return df

def remove_outliers_iqr(df):
    """
    Removes outliers from the specified columns in a DataFrame using the IQR method.
    Returns a new DataFrame without outliers.
    """
    cols = ['LOG_PRICE', 'PROPERTYSQFT', 'BEDS', 'BATH']
    df_out = df.copy()
    for col in cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    return df_out


def preprocess_data(df):
    """
    Apply all preprocessing steps to the dataframe.
    1. we calculate the logarithmic value of the price using np.log1p(x) which represent the natural logarithm
    2. we find the borough of the property
    3. we extract the ZIP code of the property
    4. then we encode the categorical values
    5. calculate distance features to key landmarks (city center, Central Park).
    6. select relevant columns and split data into features and target.

    Returns:
         X_train, X_test, y_train, y_test for model input.
    """
    # Logarithmic transformation of PRICE
    df['LOG_PRICE'] = np.log1p(df['PRICE'])

    # Extract Borough from STATE column
    df = extract_borough_from_state(df)

    # Extract ZIP code from MAIN_ADDRESS column
    df = extract_zip_code(df)

    # Encode categorical variables
    df = encode_categorical(df, fit=True)

    # Calculate distances to key locations
    df = calculate_distances(df)

    print('before outliers removal')
    print(df.shape)
    df = remove_outliers_iqr(df)
    print('after outliers removal')
    print(df.shape)

    selected_columns = ['LOG_PRICE', 'LATITUDE', 'LONGITUDE', 'DIST_TO_CENTER', 'DIST_TO_CENTRAL_PARK', 'PRICE',
                        'BEDS', 'BATH', 'PROPERTYSQFT', 'BOROUGH_FROM_STATE_ENCODED', 'SUBLOCALITY_ENCODED',
                        'LOCALITY_ENCODED', 'ADMINISTRATIVE_AREA_LEVEL_2_ENCODED', 'TYPE_ENCODED',
                        'BROKERTITLE_ENCODED', 'ZIP_CODE_ENCODED', 'LONG_NAME_ENCODED', 'STATE_ENCODED',
                        'STREET_NAME_ENCODED']
    df= df[selected_columns]

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['PRICE', 'LOG_PRICE'])
    y = df['LOG_PRICE']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def preprocess_nn_data(df):
    """
        Preprocess data for neural network training.
        Apply the same preprocessing steps to the dataframe.
        Steps:
        1. Compute the natural log of price using np.log1p (to stabilize variance).
        2. Extract the borough from the state.
        3. Extract the ZIP code from the address.
        4. Encode all categorical features using Target Encoding.
        5. Calculate distances to central locations (e.g., city center, Central Park).
        6. Select relevant features for training.
        7. Split data into train, validation, and test sets.
        8. Scale numeric input features and the target variable (log price).
        9. Save scalers for later inference use.

        Returns:
        Scaled train/val/test sets and the fitted scalers.
    """

    df['LOG_PRICE'] = np.log1p(df['PRICE'])
    df = extract_borough_from_state(df)
    df = extract_zip_code(df)
    df = encode_categorical(df, fit=True)
    df = calculate_distances(df)
    df = remove_outliers_iqr(df)

    selected_columns = ['LOG_PRICE', 'LATITUDE', 'LONGITUDE', 'DIST_TO_CENTER', 'DIST_TO_CENTRAL_PARK', 'PRICE',
                        'BEDS', 'BATH', 'PROPERTYSQFT', 'BOROUGH_FROM_STATE_ENCODED', 'SUBLOCALITY_ENCODED',
                        'LOCALITY_ENCODED', 'ADMINISTRATIVE_AREA_LEVEL_2_ENCODED', 'TYPE_ENCODED',
                        'BROKERTITLE_ENCODED', 'ZIP_CODE_ENCODED', 'LONG_NAME_ENCODED', 'STATE_ENCODED',
                        'STREET_NAME_ENCODED']

    df = df[selected_columns]
    X = df.drop(columns=['PRICE', 'LOG_PRICE'])
    y = df['LOG_PRICE']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    joblib.dump(scaler_X, 'scalers\scaler_X.pkl')
    joblib.dump(scaler_y, 'scalers\scaler_y.pkl')

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y
