# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:01:01 2019

@author: Ankit
"""

import glob
import os
import pandas as pd
path = ''                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df.to_csv("dataset_LSTM.csv", index=False)
# doesn't create a list, nor does it append to one