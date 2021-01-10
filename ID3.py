import pandas as pd
import numpy as np


# from AbstractAlgorithm import AbstractAlgorithm
def selectFeatureAndDivideGroupByIt():
    pass



class ID3:
    def __init__(self):
        self.train_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv")
        self.test_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\test.csv")

        m = 0
        b = 0
        for row in self.test_group:
            if row[0] == 'M':
                m += 1
            else:
                b += 1
        self.default = 'M' if m > b else 'B'


    def fit(self, examples, features, default, selectFeature):
        pass

    def calculateEntropy(self,examples):
        examples_len = len(examples)
        m_counter = 0
        for e in examples:
            if e[0] == 'M':
                m_counter +=1
        b_counter = examples_len-m_counter
        return -1*(  (m_counter/examples_len)*np.log2(m_counter/examples_len)
                   + (b_counter/examples_len)*np.log2(b_counter/examples_len) )

    def ig(self,f,examples):
        #need to do it dynamically
        examples.sort(lambda e: e[f])

        # values = []
        # for e in examples:
        #     values.append(e[f])
        # values.sort()
        # values_len = len(values)
        # barriers = []
        # for i in range(values_len-1):
        #     barriers.append(0.5*(values[i]+values[i+1])
        # first_group = []
        # second_group = []



    def maxIg(self,features,examples):
        max_ig = 0
        f = features[0]
        for f in features:
            result = ig(f, examples)
            if result > max_ig:
                restult = max_ig
                f = f






    #def fit(self, train_group, test_group, ):



def main():
    id3 = ID3()
    # id3.fit()
    # id3.predict()

if __name__ == "__main__":
    main()

