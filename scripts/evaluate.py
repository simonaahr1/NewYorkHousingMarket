from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .utils import df_evaluate_nn_model, df_evaluate_lr_rf_model

def evaluate_lr_rf_model(model, X_test, y_test):
    """
        Evaluation that could be used for Linear Regression and Random Forest

        Returns: mse, mae, r2 of the model
    """
    # Predict using the model
    predictions = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return f'mse: {mse}, mae: {mae}, r2: {r2}'


def evaluate_nn_model(model, X_test, y_test_scaled, scaler_y):
    """
        Evaluation of the neural network model

        Returns: mse, mae, r2 of the model
    """
    # Predict using the model
    predictions_scaled = model.predict(X_test)

    # Ensure predictions are reshaped properly for inverse scaling
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Reshape y_test_scaled to a 2D array before inverse transforming
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()


    # Calculate performance metrics
    mse = mean_squared_error(y_test_original, predictions)
    mae = mean_absolute_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    return f'mse: {mse}, mae: {mae}, r2: {r2}'

def df_evaluations(linear_model, random_forest_model, X_test, y_test, nn_model, X_nn_test, y_nn_test, scaler_y):
    """
        This function is used to create dataframe with predicted and actual values
        You can skip it
    """
    df_lr = df_evaluate_lr_rf_model(linear_model, X_test, y_test, return_df=True)
    df_rf = df_evaluate_lr_rf_model(random_forest_model, X_test, y_test, return_df=True)
    df_nn = df_evaluate_nn_model(nn_model, X_nn_test, y_nn_test, scaler_y, y_test, return_df=True)

    df_lr.to_csv('df_lr.csv')
    df_rf.to_csv('df_rf.csv')
    df_nn.to_csv('df_nn.csv')



