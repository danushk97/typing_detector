import pickle
import pandas as pd

from flask import Flask, request


app = Flask(__name__)
model = None


@app.post('/predict')
def get_prediction():
    body = request.get_json()['input']
    for row in body:
        row['character'] = ord(row['character'])
    df = pd.DataFrame(body)
    results = list(model.predict(df))
    ones = results.count(1)
    zeros = results.count(0)
    import pdb; pdb.set_trace()
    return {
        'isThanush': ones > zeros
    }


if __name__ == '__main__':
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    app.run()
