import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import joblib
import socket
import json
import numpy as np
import pandas as pd
from datetime import datetime as dt
import time



## import model specific functions and variables
from helper.fetchlib import fetch_ts, fetch_data, engineer_features

from helper.modeltools import model_train, model_load, model_predict

from helper.modeltools import models_load, models_train, _models_train

from helper.modeltools import _plot_learning_curve, _plot_feature_importance, _make_compare_plot


MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"
data_dir = os.path.join("data","cs-train")

app = Flask(__name__)
print('Starting API...')
print('Loading models...')
all_data, all_models = models_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
print('Models loading completed...')

@app.route('/retrain', methods=['GET'])
def retrain():
    run_start = time.time()
    try:
        models_train(data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION, test=False)
    except:
            return jsonify(msg='model_train error')
    try:
        tmp_all_data, tmp_all_models = models_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
    except:
            return jsonify(msg='model_load error')
    all_data = tmp_all_data
    all_models = tmp_all_models

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    return jsonify(modelTrain = 'OK', modelReload = 'OK',runningTime = "%d:%02d:%02d"%(h, m, s), modelsLoaded = ",".join(all_models.keys()))

@app.route('/reloadmodel', methods=['GET'])
def reloadmodel():
    run_start = time.time()
    
    try:
        tmp_all_data, tmp_all_models = models_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
    except:
            return jsonify(msg='model_load error')
    all_data = tmp_all_data
    all_models = tmp_all_models   

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    return jsonify(modelReload='OK',runningTime = "%d:%02d:%02d"%(h, m, s), modelsLoaded = ",".join(all_models.keys()))

@app.route('/predict', methods=['GET'])
def predict():
    """
    basic predict function for the API

    country='all'
    year='2019'
    month='04'
    day='06'
    2017-11-29    2019-05-31
    """
    run_start = time.time()

    country = request.args.get('country', default = 'all', type = str)
    date = request.args.get('date', default = '2019-05-06', type = str)
    startDate = dt.strptime("2017-11-29", "%Y-%m-%d")
    endDate = dt.strptime("2019-05-31", "%Y-%m-%d")
    try:
        predictDate = dt.strptime(date, "%Y-%m-%d")
    except:
        return jsonify(errMsg = 'Error Date, date should be in range 2017-11-29  -  2019-05-31')


    if predictDate >=startDate and predictDate < endDate:
        dataSplit = date.split('-')
        year = dataSplit[0]
        month = dataSplit[1]
        day = dataSplit[2]
        try:
            result = model_predict(country, year, month, day, all_models=all_models, all_data=all_data, data_dir=data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION)
        except:
            return jsonify(msg='model_predict error')
        #{'y_pred': array([134258.56976667]), 'y_proba': None}
        m, s = divmod(time.time()-run_start,60)
        h, m = divmod(m, 60)

        return  jsonify(status='OK', date = date, country = country, y_pred = result['y_pred'][0], runningTime = "%d:%02d:%02d"%(h, m, s),)
    else:
        return jsonify(errMsg = 'Error Date, date should be in range 2017-11-29  -  2019-05-31')

@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """

    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify(msg = "ERROR: API (log): file requested was not a log file: {}".format(filename))

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify(msg = "ERROR: API (log): cannot find log dir")

    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify(msg = "ERROR: API (log): file requested could not be found: {}".format(filename))
    
    return send_from_directory(log_dir, filename, as_attachment=True)    


if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)
        print('API runing on http://localhost:8080')

