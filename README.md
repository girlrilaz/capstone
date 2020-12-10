# Capstone Assignment


## Capston Part 0: Preparation
to run scripts and application you will need:

- setup virtual environment for python  (for example using conda)
% conda create -n test2 python=3.7 anaconda

- activate virtual environment
% conda activate test2

- clone repository with python scripts, flask application, DockerFiles and reports with outcomes
% git clone https://github.ibm.com/Tomasz-Rozmus/CapstonAssasment

- change working directory 
% cd to_dir_with_assigment

- install required python library
$ pip install -r requirements.txt



## Capstone project checklist 

### Are there unit tests for the API? 
- to test api (run api server before )
$ python -m unittest tests/api_tests.py

### Are there unit tests for the model? 
- to test models:
$ python -m unittest tests/Model_tests.py

### Are there unit tests for the logging? 
- to test logging capabilites run:
$ python -m unittest tests/Logger_tests.py

### Can all of the unit tests be run with a single script and do all of the unit tests pass? 
- or run all tests
$ python run_all_test.py

### Is there a mechanism to monitor performance? 

$ python part-2.py

### Was there an attempt to isolate the read/write unit tests from production models and logs?

### Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined? 
- start api by running:
$ python app.py

-  open following url in browser:

http://localhost:8080/predict?country=united_kingdom&date=2019-05-05

http://localhost:8080/predict?country=all&date=2019-05-05


### Does the data ingestion exists as a function or script to facilitate automation? 
helper/fetchlib.py
helper/modeltools.py


### Were multiple models compared? 

  time-series-notebooks

### Did the EDA investigation use visualizations? 

  Capstone Part1.ipynb

### Is everything containerized within a working Docker image? 

  Dockerfile 

### Did they use a visualization to compare their model to the baseline model? 

  time-series-notebooks


### Build the Docker image and run it
  
- build docker image
$ docker build -t apiapp .

 - Check that the image is there.
  
$ docker image ls

- run docker
$ docker run -d -p 8080:8080 apiapp

- check API open below url in a browser:

http://localhost:8080/predict?country=united_kingdom&date=2019-05-05

