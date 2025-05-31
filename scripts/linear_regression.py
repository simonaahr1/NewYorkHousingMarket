from sklearn.linear_model import LinearRegression
import joblib

def train_linear_regression(X_train, y_train):
    """
        Create the Linear Regression, train it and save it
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/linear_model.pkl')
