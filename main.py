from scripts import load_data, preprocess_data, preprocess_nn_data
from scripts import train_linear_regression
from scripts import train_random_forest
from scripts import create_neural_network, train_neural_network
from scripts import evaluate_lr_rf_model, evaluate_nn_model, df_evaluations
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os



if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)

    # Load and preprocess data
    df = load_data('data/NY-House-Dataset.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)

    X_nn_train, X_nn_val, X_nn_test, y_nn_train, y_nn_val, y_nn_test, scaler_X, scaler_y = preprocess_nn_data(df)

    # Train models
    train_linear_regression(X_train, y_train)
    train_random_forest(X_train, y_train)

    nn_model = create_neural_network(X_nn_train.shape[1])
    train_neural_network(nn_model, X_nn_train, y_nn_train, X_nn_val, y_nn_val)

    # Load models and evaluate
    linear_model = joblib.load('models/linear_model.pkl')
    random_forest_model = joblib.load('models/random_forest_model.pkl')
    nn_model = load_model('models/neural_network_model.keras', compile=False)

    print("Linear Regression Performance:", evaluate_lr_rf_model(linear_model, X_test, y_test))
    print("Random Forest Performance:", evaluate_lr_rf_model(random_forest_model, X_test, y_test))
    print("Neural Network Performance:", evaluate_nn_model(nn_model, X_nn_test, y_nn_test, scaler_y))

    # Addition of dataframe with predicted and actual values of the test set
    df_evaluations(linear_model, random_forest_model, X_test, y_test, nn_model, X_nn_test, y_nn_test, scaler_y)
