import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer

spam = pd.read_csv("spam.csv")
dummies = pd.get_dummies(spam.label)
spam = pd.concat([spam,dummies],axis="columns")
spam = spam.drop(["label","ham"],axis="columns")
print(spam.groupby("spam").describe())

# print(spam)
m_X_train, m_X_test, m_y_train, m_y_test = train_test_split(spam["text"], spam["spam"], test_size=0.7, random_state=0)
# v=CountVectorizer(analyzer='word',ngram_range=(2,2))
v=CountVectorizer()

m_X_train_T=v.fit_transform(m_X_train.values)
m_X_test_T=v.transform(m_X_test.values)

print(m_X_train_T.toarray()[:2])

m=Multinomial()
m.fit(m_X_train_T, m_y_train)