from sklearn.ensemble import RandomForestRegressor
import joblib

def train_random_forest(X_train, y_train):
    """
        Create the Random Forest, train it and save it
    """
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/random_forest_model.pkl')
