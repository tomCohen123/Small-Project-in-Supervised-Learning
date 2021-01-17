import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
train_group_dict = df.to_dict(orient='list')
# features = df.columns
# features.pop(0)

# l = df.loc[0]

l =[0,1,2,3]

s = l[0:2]

print(s)

# vec1=[]
# vec2=[]
# for i in range(9):
#    vec1.append(1)
#    vec2.append(2)
# print(np.linalg.norm(np.array(vec1) - np.array(vec2)))

# examples_indices = [1,8,3]
# a=df.filter(examples_indices, axis=0)#["diagnosis"]
# index_list = a.index
# print(train_group_dict['diagnosis'][0])
# print(train_group_dict['diagnosis'][1])
# features = df.columns
#
#
# print(features)

# list = [14,2,4]
# list.sort()
# print(0.5*(list[0]+list[1]))
# print(list)

#df = np.genfromtxt(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv", delimiter=',',)
#print(df[0])
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

# one_columon = df["radius_mean"].sort_values()
# print(one_columon)
# print(one_columon[0])
#


#print(df)


#print(sicks)
# print(df.loc[0]['diagnosis'])




