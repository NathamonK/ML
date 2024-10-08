import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset('iris')
df

x = df.iloc[:, :-1]
y = df['species']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

st.sidebar.title('Classifier')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'DT', 'NN', 'RF'))
k = st.sidebar.slider('K', 1, 120, 3)
if classifier =="KNN":
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier =="SVM":
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier =="DT":
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  y_pred = dt.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier =="NN":
  nn = MLPClassifier()
  nn.fit(x_train, y_train)
  y_pred = nn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier =="RF":
  rf = RandomForestClassifier()
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)


