import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_excel("DataResidential.xlsx")
df = pd.DataFrame(data)
df = df.iloc[:,1:5]
print("Data Residential".center(100,"="))
print(df)
#print(df.info())
print(df.describe(include='all'))

def count_plot(df, columns):
    plt.figure(figsize=(15, 10))
    for indx, var  in enumerate(columns):
        plt.subplot(2, 3, indx+1)
        g = sns.countplot(df[var], hue= df['class'])
    plt.tight_layout()


features = df.columns.tolist()
features.remove('Selected for Housing')
#print(features)

encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(df[features])
df_encoded = pd.DataFrame(data_encoded, columns=features)
#print(data_encoded)

encoder = LabelEncoder()
target_encoded = encoder.fit_transform(df['Selected for Housing'])
df_encoded['Selected for Housing'] = target_encoded
encoder.inverse_transform(target_encoded)
#print(target_encoded)
print(df_encoded)

X_train = df_encoded.iloc[0:10, 1:3].values
print("instance variabel data training".center(100, "="))
print(X_train)

X_test = df_encoded.iloc[10, 1:3].values
X_test = X_test.reshape(1, -1)
print("instance variabel data testing".center(100, "="))
print(X_test)

y_train = df_encoded.iloc[0:10, 3].values
print("instance kelas data training".center(100, "="))
print(y_train)

y_test = df_encoded.iloc[10, 3]
y_test = y_test.reshape(1, -1)
print("instance kelas data testing".center(100, "="))
print(y_test)

cnb = CategoricalNB()
cnb.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_cnb = cnb.predict(X_test)
y_prob_pred_cnb = cnb.predict_proba(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred_cnb).sum()
print("CategoricalNB".center(100, "="))
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred_cnb)
print('Accuracy: {:.2f}'.format(accuracy))

print("CONFUSION MATRIX".center(100, "="))
cm = confusion_matrix(y_test, y_pred_cnb)
print('Confusion matrix for Categorical Naive Bayes\n', cm)

#print("Recall score : ", recall_score(y_test, y_pred_cnb , average='micro'))
#print("Precision score : ",precision_score(y_test, y_pred_cnb , average='micro'))
#print("F1 score : ",f1_score(y_test, y_pred_cnb , average='micro'))
print(classification_report(y_test, y_pred_cnb))


y_pred_gnb = gnb.predict(X_test)
y_prob_pred_gnb = gnb.predict_proba(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred_gnb).sum()
print("GaussianNB".center(100, "="))
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred_gnb)
print('Accuracy: {:.2f}'.format(accuracy))

print("CONFUSION MATRIX".center(100, "="))
cm = confusion_matrix(y_test, y_pred_gnb)
print('Confusion matrix for Gaussian Naive Bayes\n', cm)

#print("Recall score : ", recall_score(y_test, y_pred_gnb , average='micro'))
#print("Precision score : ",precision_score(y_test, y_pred_gnb , average='micro'))
#print("F1 score : ",f1_score(y_test, y_pred_gnb , average='micro'))
print(classification_report(y_test, y_pred_gnb))