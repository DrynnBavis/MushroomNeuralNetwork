import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']

df = pd.read_csv('HousingData.csv')
##some data processing here
df_std = df
for col in columns: ##standardizing data
    df_std[col] = (df[col] - df[col].mean()) / df[col].std()
df_std.to_csv('StdHousingData.csv')

df_norm = df
for col in columns: ##normailizing data
    df_norm[col] = (df[col] - df[col].min()) /(df[col].max() - df[col].min())
df_norm.to_csv('NormHousingData.csv')
