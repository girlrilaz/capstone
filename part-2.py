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

#from logger import update_predict_log, update_train_log
from helper.logger import update_predict_log, update_train_log

#from fetchlib  import fetch_ts, fetch_data
from helper.fetchlib import fetch_ts, fetch_data, engineer_features

from helper.modeltools import model_train, model_load, model_predict

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"


if __name__ == "__main__":
    run_start = time.time()

    ## train the model
    #print("TRAINING MODELS")
    data_dir = os.path.join("data","cs-train")
    model_train(data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION, test=False)


    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    country='united_kingdom'
    year='2019'
    month='04'
    day='06'
    result = model_predict(country,year,month,day, data_dir=data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION)
    print("...result {}".format(result))

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("Running time:", "%d:%02d:%02d"%(h, m, s))
    print("Done")
    
    