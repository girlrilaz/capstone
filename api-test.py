import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np
import json

port = 8080

try:
    r = requests.get('http://127.0.0.1:{}/predict'.format(port))
    server_available = True
except:
    server_available = False

print(r)
print(r.text)
rj = json.loads(r.text)
print(rj['status'])
