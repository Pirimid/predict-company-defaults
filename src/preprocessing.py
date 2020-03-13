import os

from tqdm import tqdm 
import numpy as np
import pandas as pd 

df = pd.read_csv('../data/z_score_data.csv')
columns = df.columns

def convert_string_to_float(x):
    return [float(str(x[col]).replace(",", "").replace(' -   ', str(0)).replace("%", "")) / 1000000 for col in columns]

df.replace(to_replace='#DIV/0!', value=0, inplace=True)

# def percentage_change(x):
#     for i,col in enumerate(columns):
#         try:
#             if i < len(columns) and i > 0 :
#                 if columns[i+1].find(col) >= 0:
#                     try:
#                         x[i+1] = (x[i+1] - x[i]) / x[i]
#                     except:
#                         pass
#         except:
#             pass
#     return x

df = df.apply(convert_string_to_float, axis=1)
# df = df.apply(percentage_change).values