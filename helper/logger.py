#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import datetime
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    now = datetime.now()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", "train-{}-{}.log".format(now.year, now.month))
        
    ## write the data to a csv file    
    header = ['timestamp','unique_id','x_shape','eval_test','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)
        #Oct 11 22:14:15 
        timeStamp = now.strftime("%b %d %H:%M:%S.%f")
        to_write = map(str, [timeStamp, uuid.uuid4(), data_shape, eval_test,
                            MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(y_pred, y_proba, query, runtime, MODEL_VERSION, test=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    now = datetime.now()
    #Oct 11 22:14:15 
    timeStamp = now.strftime("%b %d %H:%M:%S.%f")
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(now.year, now.month))
        
    ## write the data to a csv file    
    header = ['timestamp','unique_id','y_pred','y_proba','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [timeStamp, uuid.uuid4(), y_pred, y_proba,query,
                            MODEL_VERSION, runtime])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    MODEL_VERSION = 0.1
    MODEL_VERSION_NOTE = 'Test'
    
    ## train logger
    update_train_log(str((100,10)),"{'rmse':0.5}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
    ## predict logger
    update_predict_log("[0]", "[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
                       "00:00:01", MODEL_VERSION, test=True)
    
        
