import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv")
#sicks = df[df['diagnosis'] == 'M']

# #for x in range(1,31) :
#     print (x)

# f_col_name = df.columns[0]
# sicks = df[df[f_col_name] == 'M']
#
#print(type(df))
# for index in df.index:
#     print(df.loc[index]['diagnosis'])

sorted = df.sort_values('radius_mean')

print(sorted)
print(df)
# print(df.loc[0]['diagnosis'])




