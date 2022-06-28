import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

# membaca data
iris = datasets.load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                  columns = iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
del df['target']
print("data awal".center(100,"="))
print(df)

# melihat statistik data
df['species'].unique()
print("deskripsi statistik data".center(100,"="))
print(df.describe(include='all'))

# melihat informasi data
print("informasi data".center(100, "="))
df.info()

# plotting untuk persebaran variabel
# sns.pairplot(df, hue="species")
# plt.show()

# pengecekan missing value
print("pengecekan missing value".center(100,"="))
df.isnull().sum()

# grouping yang dibagi menjadi dua
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
print("data variabel".center(100,"="))
print(X)
print("data kelas".center(100,"="))

# pelabelan string menjadi angka
print (y)
le = LabelEncoder()
y = le. fit_transform(y)

# pembagian training den testing
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("instance variabel data training".center(100, "="))
print(X_train)
print("instance kelas data training".center(100, "="))
print(y_train)
print("instance variabel data testing".center(100, "="))
print(X_test)
print("instance kelas data testing".center(100, "="))
print(y_test)

# pemodelan naive bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
accuracy_nb = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print("instance prediksi naive bayes:")
print(Y_pred)

# perhitungen confusion matrix
cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision = precision_score(y_test, Y_pred, average='micro')
recall = recall_score(y_test, Y_pred, average='micro')
f1 = f1_score(y_test,Y_pred, average='micro')
print('Confusion matrix for Naive Bayes\n', cm)
print('accuracy Naive Bayes: %.3f' %accuracy)
print('precision_Naive Bayes: %.3f' %precision)
print('recall_Naive Bayes: %.3f' %recall)
print('f1 score Naive Bayes: %.3f' %f1)