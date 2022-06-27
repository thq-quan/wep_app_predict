import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce

data = pd.read_csv("D:\streamlit\data_final_clean.csv")
X = data.drop(['is_paid'], axis=1)
y = data['is_paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
encoder = ce.OrdinalEncoder(cols=['region','language','package'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
print('lr')
print(y_lr)
print(accuracy_score(y_test, y_lr)*100)

from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(X_train,y_train)
y_clf = clf.predict(X_test)
print('preceptron')
print(y_clf)
print(accuracy_score(y_test, y_clf)*100)

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)
y_knn = KNN.predict(X_test)
print('KNN')
print(y_knn)
print(accuracy_score(y_test, y_knn)*100)

from sklearn.tree import DecisionTreeClassifier
id3 = DecisionTreeClassifier(criterion="entropy")
id3.fit(X_train,y_train)
y_id3 = id3.predict(X_test)
print('ID3')
print(y_id3)
print(accuracy_score(y_test, y_id3)*100)

from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)
y_svc = svc.predict(X_test)
print('svm')
print(y_svc)
print(accuracy_score(y_test, y_svc)*100)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
y_nb = nb.predict(X_test)
print('Naive Bayes')
print(y_nb)
print(accuracy_score(y_test, y_nb)*100)
