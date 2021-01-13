import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv")
#sicks = df[df['diagnosis'] == 'M']

# #for x in range(1,31) :
#     print (x)

# f_col_name = df.columns[0]
#sicks = df[df["diagnosis"] == 'M']
#
#print(type(df))
# for index in df.index:
#     print(df.loc[index]['diagnosis'])

#sorted = df.sort_values('radius_mean')
#print(df[df["diagnosis"] == 'M'].count()["diagnosis"])

one_columon = df["radius_mean"].sort_values()
print(one_columon)
print(one_columon[0])
#print(sicks)
# print(df.loc[0]['diagnosis'])




