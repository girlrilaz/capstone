#!/usr/bin/env python
import os
import sys
import re
import shutil
import time
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import csv,uuid,joblib

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

from collections import defaultdict
from datetime import datetime
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score



#from logger import update_predict_log, update_train_log
from helper.logger import update_predict_log, update_train_log

#from fetchlib  import fetch_ts, fetch_data
from helper.fetchlib import fetch_ts, fetch_data, engineer_features

from helper.modeltools import model_train, model_load, model_predict3

from helper.modeltools import models_load, models_train, _models_train

from helper.modeltools import _plot_learning_curve, _plot_feature_importance, _make_compare_plot

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

if __name__ == "__main__":
    
    run_start = time.time()

    ## train the model
    #print("\nTRAINING MODELS")
    data_dir = os.path.join("data","cs-train")
    #models_train(data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION, test=False)


    ## load the model
    print("\nLOADING MODELS")
    all_data, all_models = models_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    country='all'
    year = '2019'
    month = '4'
    day = '6'
    queries = [['2019', '1','2'], ['2019', '2','2'], ['2019', '3','2']]
    #y_pred = [predict(query, saved_model)[0] for query in queries]
    y_pred = [model_predict3( country, query[0], query[1], query[2], all_models=all_models, all_data=all_data, data_dir=data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION)[0] for query in queries]
    print(f1_score([1,2,0], y_pred, average='weighted'))
    
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("Running time:", "%d:%02d:%02d"%(h, m, s))
    