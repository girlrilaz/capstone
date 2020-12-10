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

from helper.fetchlib import fetch_ts, fetch_data

if __name__ == "__main__":

    run_start = time.time() 
    data_dir = os.path.join(".","data","cs-train")
    print("...fetching data")

    ts_all = fetch_ts(data_dir,clean=False)

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print()
    print("information about fetched data")
    print("country name and number of records \n")

    for key,item in ts_all.items():
        print(key,item.shape)
  