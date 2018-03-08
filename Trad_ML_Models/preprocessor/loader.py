import csv
import os.path
import pandas as pd

data = []
curfilePath = os.path.abspath('__file__')
curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))
parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
dataPath = os.path.join(parentDir, 'data', 'preprocessedData.csv')


def loadData():
     return pd.read_csv(dataPath)

