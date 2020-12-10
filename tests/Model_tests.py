#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from helper.logger import update_train_log, update_predict_log

from helper.modeltools import model_train, model_load, model_predict, models_train, models_load



class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
    
    def test_01_train(self):
        """
        models train

        """
        MODEL_DIR = "models"
        MODEL_VERSION = 0.1
        data_dir = os.path.join("data","cs-train")
        models = models_train(data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION, test=False)
        self.assertTrue(len(models) > 0)
    
    def test_02_train(self):
        """
        models train

        """
        MODEL_DIR = "models"
        MODEL_VERSION = 0.1
        data_dir = os.path.join("data","cs-train")
        model = model_train(data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION, test=False)
        self.assertTrue(len(model) > 0)

    def test_03_load(self):
        """
        train models
        """
        MODEL_DIR = "models"
        MODEL_VERSION = 0.1
        data_dir = os.path.join("data","cs-train")
        all_data, all_models = models_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
        

        self.assertEqual(len(all_models), 11)

    def test_04_load(self):
        """
        train models
        """
        MODEL_DIR = "models"
        MODEL_VERSION = 0.1
        data_dir = os.path.join("data","cs-train")
        all_data, all_models = model_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)
        
        self.assertEqual(len(all_models), 11)

    
    def test_05_predict(self):
        """
        predict value
        """
       
        MODEL_DIR = "models"
        MODEL_VERSION = 0.1
        data_dir = os.path.join("data","cs-train")

        ## load the model
        all_data, all_models = models_load(data_dir=data_dir, MODEL_DIR=MODEL_DIR)

        ## test predict
        country='all'
        year='2019'
        month='04'
        day='06'
        result = model_predict(country,year,month,day, all_models=all_models, all_data=all_data, data_dir=data_dir, MODEL_DIR=MODEL_DIR, MODEL_VERSION=MODEL_VERSION)

        self.assertEqual(round(result['y_pred'][0],3), round(134258.56976667,3))
    


### Run the tests
if __name__ == '__main__':
    unittest.main()
      
