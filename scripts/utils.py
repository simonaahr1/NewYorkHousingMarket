import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def df_evaluate_lr_rf_model(model, X_test, y_test, return_df=False):
    '''
    This function is used in order to create dataframe with output predicted values and actual values
    It is build for Linear Regression and Random Fores
    '''
    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    y_actual = np.expm1(y_test)


    if return_df:
        df = pd.DataFrame({
            "Actual Price": y_actual,
            "Predicted Price": predictions,
            "Abs Error": np.abs(y_actual - predictions),
            "% Error": np.abs((y_actual - predictions) / y_actual) * 100
        })
        return df

    return 0

def df_evaluate_nn_model(model, X_test, y_test_scaled, scaler_y, y_test, return_df=False):
    predictions_scaled = model.predict(X_test)
    predictions_log = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test_log = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    predictions = np.expm1(predictions_log)
    y_actual = np.expm1(y_test_log)

    if return_df:
        df = pd.DataFrame({
            "Actual Price": y_actual,
            "Predicted Price": predictions,
            "Abs Error": np.abs(y_actual - predictions),
            "% Error": np.abs((y_actual - predictions) / y_actual) * 100
        })
        return df

    return 0