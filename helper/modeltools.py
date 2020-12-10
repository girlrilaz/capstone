#!/usr/bin/env python
import os
import sys
import re
import shutil
import time
import pickle
import csv,uuid,joblib

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

from helper.fetchlib import fetch_data, fetch_ts, engineer_features
from helper.logger import update_train_log, update_predict_log

def model_train(data_dir: str, MODEL_DIR, MODEL_VERSION, test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)
    ret = []
    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _ret = _model_train(df,country, MODEL_DIR, MODEL_VERSION, test=test)
        ret.append(_ret)
    return ret

def _model_train(df,tag, MODEL_DIR, MODEL_VERSION, test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 

    """


    ## start timer for runtime
    time_start = time.time()
    ret = []
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a random forest model
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])
    
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    ret = [eval_rmse, tag]
    
    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(grid,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log 
    # def update_train_log(data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):

    MODEL_VERSION_NOTE = ''
    update_train_log((str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    return ret
  
def model_load(prefix='sl',data_dir=None,training=True, MODEL_DIR='models'):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("cs-train")
    
    models = [f for f in os.listdir(MODEL_DIR) if re.search("sl",f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(MODEL_DIR,model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
        
    return(all_data, all_models)

def model_predict3(country,year,month,day,all_models=None, all_data=None,test=False, data_dir=None, MODEL_DIR='models', MODEL_VERSION=0.1):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        print('load models all_models and all_data')
        all_data,all_models = model_load(training=False, data_dir=data_dir, MODEL_DIR=MODEL_DIR)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    ## snity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None

    

    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == False:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    ##  update_predict_log(y_pred, y_proba, query, runtime, MODEL_VERSION, test=False):

    update_predict_log(y_pred,y_proba,target_date,runtime, MODEL_VERSION, test=test)
    
    return(y_pred)

def model_predict(country,year,month,day,all_models=None, all_data=None,test=False, data_dir=None, MODEL_DIR='models', MODEL_VERSION=0.1):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        print('load models all_models and all_data')
        all_data,all_models = model_load(training=False, data_dir=data_dir, MODEL_DIR=MODEL_DIR)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    ## snity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None

    

    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    ##  update_predict_log(y_pred, y_proba, query, runtime, MODEL_VERSION, test=False):

    update_predict_log(y_pred,y_proba,target_date,runtime, MODEL_VERSION, test=test)
    
    return({'y_pred':y_pred,'y_proba':y_proba})

def _plot_learning_curve(estimator, X, y, ax=None, cv=5):
    """
    an sklearn estimator 
    """

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    train_sizes=np.linspace(.1, 1.0, 6)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv, n_jobs=-1,
                                                            train_sizes=train_sizes,scoring="neg_mean_squared_error")
    train_scores_mean = np.mean(np.sqrt(-train_scores), axis=1)
    train_scores_std = np.std(np.sqrt(-train_scores), axis=1)
    test_scores_mean = np.mean(np.sqrt(-test_scores), axis=1)
    test_scores_std = np.std(np.sqrt(-test_scores), axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ## axes and lables
    buff = 0.05
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xbuff = buff * (xmax - xmin)
    ybuff = buff * (ymax - ymin)
    ax.set_xlim(xmin-xbuff,xmax+xbuff)
    ax.set_ylim(ymin-ybuff,ymax+ybuff)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("MSE Score")
    
    ax.legend(loc="best")

def _make_compare_plot(X, y, models, verbose=True, figname='figname'):
    """
    create learning curves for SGD, RF, GB and ADA
    """
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    if verbose:
        print("...creating learning curves")
    
    ## sgd
    reg1 = SGDRegressor(**models["SGD"][1])
    pipe1 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg1)])
    _plot_learning_curve(pipe1, X, y, ax=ax1)
    ax1.set_title("Stochastic Gradient Regressor")
    ax1.set_xlabel("")
    
    ## random forest
    reg2 = RandomForestRegressor(**models["RF"][1])
    pipe2 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg2)])
    _plot_learning_curve(pipe2, X, y, ax=ax2)
    ax2.set_title("Random Forest Regressor")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    
    ## gradient boosting
    reg3 = GradientBoostingRegressor(**models["GB"][1])
    pipe3 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg3)])
    _plot_learning_curve(pipe3, X, y, ax=ax3)
    ax3.set_title("Gradient Boosting Regressor")
    
    ## ada boosting
    reg4 = AdaBoostRegressor(**models["ADA"][1])
    pipe4 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg4)])
    _plot_learning_curve(pipe3, X, y, ax=ax4)
    ax4.set_title("Ada Boosting Regressor")
    ax4.set_ylabel("")
    
    ymin, ymax = [], []
    for ax in [ax1, ax2, ax3, ax4]:
        ymin, ymax = ax.get_ylim()
        
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim([ymin.min(), ymax.max()])
    fig.savefig(os.path.join("reports",figname))
        
def _plot_feature_importance(estimator, feature_names, verbose=True):
    """
    plot feature importance
    """
    
    if verbose:
        print("...plotting feature importance")
        
    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(111)
    
    # make importances relative to max importance
    feature_importance = estimator.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    ax.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, feature_names[sorted_idx])
    ax.set_xlabel('Relative Importance')
    ax.set_title('Variable Importance')

def models_train(data_dir: str, MODEL_DIR, MODEL_VERSION, test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    ret = []
    for country,df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _ret = _models_train(df,country, MODEL_DIR, MODEL_VERSION, test=test)
        ret.append(_ret)
    return ret

def _models_train(df,tag, MODEL_DIR, MODEL_VERSION, test=False):
    """
    funtion to train models
    """

    ## start timer for runtime
    time_start = time.time()
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Perform a train-test split
    rs=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    
    ## build models
    regressor_names = ["SGD","RF", "GB", "ADA"]
    
    regressors = (SGDRegressor(random_state=rs),
                  RandomForestRegressor(random_state=rs),
                  GradientBoostingRegressor(random_state=rs),
                  AdaBoostRegressor(random_state=rs))
    
    params = [
        {"reg__penalty":["l1","l2","elasticnet"],
         "reg__learning_rate":["constant","optimal","invscaling"]},
        {"reg__n_estimators":[10,30,50],
         "reg__max_features":[3,4,5],
         "reg__bootstrap":[True, False]},
        {"reg__n_estimators":[10,30,50],
         "reg__max_features":[3,4,5],
         "reg__learning_rate":[1, 0.1, 0.01, 0.001]},
        {"reg__n_estimators":[10,30,50],
         "reg__learning_rate":[1, 0.1, 0.01, 0.001]}]
    
    ## train models
    models = {}
    total = len(regressor_names)
    
    for iteration, (name,regressor,param) in enumerate(zip(regressor_names, regressors, params)):
        pipe = Pipeline(steps=[("scaler", StandardScaler()), 
                                ("reg", regressor)])
        
        grid = GridSearchCV(pipe, param_grid=param, 
                            scoring="neg_mean_squared_error",
                            cv=5, n_jobs=-1, return_train_score=True)

        grid.fit(X_train, y_train)
        models[name] = grid, grid.best_estimator_["reg"].get_params()

    ## plot learning curves
    if True:
        _make_compare_plot(X, y, models=models, verbose=True, figname="{}_learning_curves.pdf".format(tag))

    ## evaluation on the validation set
    val_scores = []
    for key, model in models.items():
        y_pred = model[0].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        val_scores.append(rmse)

    ## select best model
    bestmodel = regressor_names[np.argmin(val_scores)]
    opt_model, params = models[bestmodel]
    
    vocab = {"RF":"Random Forest",
             "SGD":"Stochastic Gradient",
             "GB":"Gradient Boosting",
             "ADA":"Ada Boosting"}
    ret = []
    if True:
        print("...best model:{}".format(vocab[bestmodel]), tag)
        ret = [vocab[bestmodel], tag]

    ## retrain best model on the the full dataset
    opt_model.fit(X, y)

    ## Check the data directory
    model_path=MODEL_DIR
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if test:
        saved_model = os.path.join(model_path,"test-{}-model-{}.joblib".format(tag,re.sub("\.","_",str(MODEL_VERSION))))
    else:
        saved_model = os.path.join(model_path,"prod-{}-model-{}.joblib".format(tag,re.sub("\.","_",str(MODEL_VERSION))))

    ## save the best model
    joblib.dump(opt_model, saved_model)
 

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log 
    # def update_train_log(data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):

    MODEL_VERSION_NOTE = ''
    update_train_log((str(dates[0]),str(dates[-1])),{'rmse':max(val_scores)},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    return ret
    

def models_load(prefix='sl',data_dir=None,training=True, MODEL_DIR='models'):
    """
     funtion to load model
    
    The prefix allows the loading of different models
    """
    if training:
        prefix = "prod"
    else:
        prefix = "test"

    if not data_dir:
        data_dir = os.path.join("cs-train")
    
    if not os.path.exists(MODEL_DIR):
        raise Exception("Erorr! Model dir does not exist")
    
    models = [f for f in os.listdir(MODEL_DIR) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(MODEL_DIR,model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}

    
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
    
        
    return(all_data, all_models)
