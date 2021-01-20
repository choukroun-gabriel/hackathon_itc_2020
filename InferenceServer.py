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
    with open('stress_model.pkl', 'rb') as input_file:
        gnb = load(input_file)

    fatigue = request.args.get('fatigue')
    designation = request.args.get('designation')
    hours_worked = request.args.get('hours_worked')
    gender = request.args.get('gender')
    company_type = request.args.get('company_type')
    wfh_setup = request.args.get('wfh_setup')
    array = np.reshape([fatigue, designation, hours_worked, gender, company_type,
                        wfh_setup], (1, -1))

    result = gnb.predict(array.astype(float))
    return str(result)


if __name__ == '__main__':
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))