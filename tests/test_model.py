import joblib

def test_model():
    model = joblib.load("model.pkl")
    pred = model.predict([[5]])
    
    assert round(pred[0]) == 10