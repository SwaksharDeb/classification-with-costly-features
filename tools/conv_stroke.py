# %% Instructions:
# =============
# 1.Download the wine dataset and Load the wine dataset
# 2. Remove fist row and leading spaces
# 3. Run this script

# %% Import dependeces
import pandas as pd
import numpy as np
from ipdb import set_trace
from sklearn.utils import shuffle

# %%
SEED = 998822
#---
np.random.seed(SEED)

COL_COUNT = '_count'

# %% Import the wine dataset
data = pd.read_csv("../data/raw/stroke_rehabilitation.csv", header = None, engine = 'python')
data[COL_COUNT] = 1  # adding a coloum named _count
data = data.rename(columns={66:"_label"}) # rename the label column
data = data.sample(frac=1).reset_index(drop=True)

# %% train, validation, test split
TRAIN_SIZE = int(data.shape[0] * 0.5)  # 0.5%
VAL_SIZE   = int(data.shape[0] * 0.15)  # 0.15%
TEST_SIZE  = int(data.shape[0] * 0.35)   # 0.35 %

data_train = data.iloc[0:TRAIN_SIZE]
data_val   = data.iloc[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
data_test   = data.iloc[TRAIN_SIZE+VAL_SIZE:]

print("Total len:", data.shape[0])
print("----------")
print("Train len:", data_train.shape[0])
print("Val len:  ", data_val.shape[0])
print("Test len: ", data_test.shape[0])

# %% Storing the training, validation and test data
data_train.to_pickle("../data/stroke_rehabilitation-train")
data_val.to_pickle("../data/stroke_rehabilitation-val")
data_test.to_pickle("../data/stroke_rehabilitation-test")

# %% prepare meta data and store it
idx = data.columns[:-2]
meta = pd.DataFrame(index=idx, dtype='float32')

meta['avg'] = data_train.mean()
meta['std'] = data_train.std()
meta['cost'] = 1.

meta = meta.astype('float32')

meta.to_pickle("../data/stroke_rehabilitation-meta")
