import pandas as pd
from tensorflow import keras

from flask import Flask, request


app = Flask(__name__)
model = None


@app.post('/predict')
def get_prediction():
    body = request.get_json()['input']
    for row in body:
        row['character'] = ord(row['character'])
    df = pd.DataFrame(body)
    results = [int(pred[0] > 0.5) for pred in model.predict(df)]
    ones = results.count(1)
    zeros = results.count(0)
    return {
        'isThanush': ones > zeros
    }


if __name__ == '__main__':
    model = keras.models.load_model('model')
    app.run(host='127.0.0.1')
