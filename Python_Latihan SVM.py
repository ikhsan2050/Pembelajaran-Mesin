import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data=pd.read_csv('Social_Network_Ads.csv')
df=pd.DataFrame(data)
print(df)

x=df.iloc[:,1:3]
y=df.iloc[:,4]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

model=SVC()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print("HASIL PREDIKSI SVM")
print(y_pred)
print("HASIL CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print("HASIL AKURASI PEMODELAN SVM:", accuracy_score(y_test, y_pred))