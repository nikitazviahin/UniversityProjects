import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import preprocessing
from matplotlib import pyplot as plt
import statistics

pd.options.display.max_columns = None

# fetch the training file
file_path_full_training_set = 'train.txt'

df = pd.read_csv(file_path_full_training_set, header=None)
df = df[(df[41] == 'back') | (df[41] == 'normal')]
df.head()