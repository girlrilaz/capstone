import unittest
import getopt
import sys
import os

## parse inputs
try:
    optlist, args = getopt.getopt(sys.argv[1:],'v')
except getopt.GetoptError:
    print(getopt.GetoptError)
    print(sys.argv[0] + "-v")
    print("... the verbose flag (-v) may be used")
    sys.exit()

VERBOSE = False
RUNALL = False

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

for o, a in optlist:
    if o == '-v':
        VERBOSE = True

## api tests
from api_tests import ApiTest
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(ApiTest)
# 
# 
# ## logger tests
from Logger_tests import LoggerTest
LoggerTestSuite = unittest.TestLoader().loadTestsFromTestCase(LoggerTest)

## model tests
from Model_tests import ModelTest
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)


#MainSuite = unittest.TestSuite([LoggerTestSuite,ModelTestSuite,ApiTestSuite])
MainSuite = unittest.TestSuite([LoggerTestSuite,ModelTestSuite,ApiTestSuite])
