from statistics import mode
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data(url):
    df = pd.read_csv(url)
    return df


def normalization(data):
    scaler = StandardScaler()
    data_aftre_normalization = scaler.fit_transform(data)
    return data_aftre_normalization, scaler


class knn:
    def __init__(self, k, type_problem='Classification'):
        self.k = k
        self.type_problem = type_problem

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        self.prediction = [self._predict(x) for x in X]

    def _predict(self, q):

        distance = np.array([np.sum((q-x)**2) for x in self.X])
        index_Knearest = np.argsort(distance)[:self.k]

        if self.type_problem == 'Classification':
            predict = mode(self.y[index_Knearest].values)
            return predict
        else:
            return np.mean(self.y[index_Knearest])

    def test(self, y):
        correct = self.prediction == y
        correct = correct.sum()
        return (correct/len(y))*100


df = get_data(
    r"D:\Data Science\course_ML_mostafa_saad\archive\KNNAlgorithmDataset.csv")
y = df['diagnosis']
X = df.drop(['diagnosis', 'Unnamed: 32', 'id'], axis=1)

model = knn(3)
model.fit(X.iloc[:500, :].to_numpy(), y.iloc[:500])
predict = model.predict(X.iloc[:500, :].to_numpy())
x = model.test(y.iloc[:500])
print(x)
