import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


data  = pd.read_csv('Language Detection.csv')

df = pd.DataFrame(data)
print(df) 
