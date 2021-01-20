from flask import Flask, request
from pickle import load
import numpy as np
import pandas as pd
import json
import os
import sklearn

# Create flask application
app = Flask(__name__)


def min_max_scale(value, min, max):
    return (float(value) - min) / (max - min)


def dummy_gender(value):
    if value.lower() == 'female':
        return 0
    if value.lower() == 'male':
        return 1


def dummy_company(value):
    if value.lower() == 'product':
        return 0
    if value.lower() == 'service':
        return 1


def dummy_wfh(value):
    if value.lower() == 'no':
        return 0
    if value.lower() == 'yes':
        return 1


@app.route('/predict')
def predict():
    with open('stress_model.pkl', 'rb') as input_file:
        gnb = load(input_file)

    # Scaling
    fatigue = min_max_scale(request.args.get('fatigue'), min=0, max=10)
    designation = min_max_scale(request.args.get('designation'), min=0, max=5)
    hours_worked = min_max_scale(request.args.get('hours_worked'), min=1, max=10)

    # Dumming
    gender = dummy_gender(request.args.get('gender'))
    company_type = dummy_company(request.args.get('company_type'))
    wfh_setup = dummy_wfh(request.args.get('wfh_setup'))

    array = np.reshape([fatigue, designation, hours_worked, gender, company_type, wfh_setup], (1, -1))
    result = gnb.predict(array.astype(float))

    return str(result)


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run(host='0.0.0.0')
