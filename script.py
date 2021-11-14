import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:\\Users\\sanje\\Desktop\\RG\\Machine-Learning-based-Password-Strength-Classification\\data.csv', usecols = ['password','strength'])

print(df.shape)
df.head()

def getTokens(inputString): #custom tokenizer. ours tokens are characters rather than full words
	tokens = []
	for i in inputString:
		tokens.append(i)
	return tokens

vectorizer = TfidfVectorizer(tokenizer=getTokens) #vectorizing
X = vectorizer.fit_transform(df['password'].values.astype('U'))
X_train, X_test, y_train, y_test = train_test_split(X, df['strength'], test_size=0.10, random_state=42)  #splitting
print("Train test")
# lgs = LogisticRegression(penalty='l2',multi_class='ovr')  #our logistic regression classifier
# lgs.fit(X_train, y_train) #training
# print(lgs.score(X_test, y_test))  #testing

# #more testing
# X_predict = ['faizanahmad','faizanahmad123','faizanahmad##','ajd1348#28t**','ffffffffff','kuiqwasdi','uiquiuiiuiuiuiuiuiuiuiuiui','mynameisfaizan','mynameis123faizan#','faizan','123456','abcdef']
# X_predict = vectorizer.transform(X_predict)
# y_Predict = lgs.predict(X_predict)
# print (y_Predict)

# our Decison Tree classifier 
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(max_depth=42, criterion="gini")
# clf = clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))

# Pickle
# filename = 'decisionTree_model.sav'
# pickle.dump(clf, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)

joblib.dump(vectorizer, 'vectorizer.pkl')

#our Random Forest classifier
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# clf = clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
