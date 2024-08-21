import numpy as np

x = np.random.rand(100)
y = 2 * x + 1 + 0.2*np.random.randn(100)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


st.sidebar.title('Classifier')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'DT', 'NN', 'RF'))
k = st.sidebar.slider('K', 1, 120, 3)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.scatter(x, y_pred)
st.pyplot(fig)

if classifier =="KNN":
  knn = KNeighborsRegressor(n_neighbors=5)
  knn.fit(x.reshape(-1, 1), y)
  y_pred = knn.predict(x.reshape(-1, 1))
  plt.scatter(x, y)
  plt.scatter(x, y_pred)
  plt.show()
if classifier =="SVM":
  svm = SVR()
  svm.fit(x.reshape(-1, 1), y)
  y_pred = svm.predict(x.reshape(-1, 1))
  plt.scatter(x, y)
  plt.scatter(x, y_pred)
  plt.show()
  
if classifier =="DT":

if classifier =="NN":

if classifier =="RF":

