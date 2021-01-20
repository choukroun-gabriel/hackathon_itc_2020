from flask import Flask, request
from pickle import load
import numpy as np
import pandas as pd
import json
import os
import sklearn

# Create flask application
app = Flask(__name__)


@app.route('/predict')
def predict_single():
    with open('gnb_heart.pkl', 'rb') as input_file:
        gnb = load(input_file)
    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    chol = request.args.get('chol')
    fbs = request.args.get('fbs')
    restecg = request.args.get('restecg')
    thalach = request.args.get('thalach')
    exang = request.args.get('exang')
    oldpeak = request.args.get('oldpeak')
    slope = request.args.get('slope')
    ca = request.args.get('ca')
    thal = request.args.get('thal')
    array = np.reshape([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal], (1, -1))
    result = gnb.predict(array.astype(float))
    return str(result)


if __name__ == '__main__':
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))