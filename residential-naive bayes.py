import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_excel("DataResidential.xlsx")
df = pd.DataFrame(data)
print("Data Residential".center(100,"="))
#print(df)
print(df.info())
#print(df.describe(include='all'))

df["Land Price"] = df["Land Price"].astype("category")
df["Distance from the City Center"] = df["Distance from the City Center"].astype("category")
df["Availability of Public Transportation"] = df["Availability of Public Transportation"].astype("category")
df["Selected for Housing"] = df["Selected for Housing"].astype("category")
print(df)
print(df.info())

X = df.iloc[:,1:4].values
y = df.iloc[:,4].values
print("data variabel".center(100,"="))
#print(X)
#print (y)
le = LabelEncoder()
y = le. fit_transform(y)

#X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1)
X_train = df.iloc[0:10, 1:4].values
print("instance variabel data training".center(100, "="))
print(X_train)

X_test = df.iloc[10, 1:4].values
print("instance variabel data testing".center(100, "="))
print(X_test)

y_train = df.iloc[0:10, 4].values
print("instance kelas data training".center(100, "="))
print(y_train)

y_test = df.iloc[10, 4]
print("instance kelas data testing".center(100, "="))
print(y_test)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
accuracy_nb = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print("instance prediksi naive bayes:")
print(Y_pred)
