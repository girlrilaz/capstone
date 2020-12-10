#!/usr/bin/env python
"""
api tests

"""

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
    requests.get('http://127.0.0.1:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """
      
        r = requests.get('http://localhost:8080/predict?country=united_kingdom&date=2019-05-05')
        check_complete = json.loads(r.text)['status']
        self.assertEqual(check_complete, 'OK')

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
