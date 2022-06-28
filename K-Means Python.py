import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# READ DATA
df = pd.read_csv('imdb_top_1000.csv')
#df.info()
#print(df_fix.head())

# TRANSFORM DATA
df['Runtime'] = df['Runtime'].map(lambda x: x.rstrip('min'))
df['Gross'] = df['Gross'].str.replace(',', '')
#print(df)
df['Runtime'] = df['Runtime'].astype(float)
df['Gross'] = df['Gross'].astype(float)
#df.info()

# SELECT DATA
df_new = df.iloc[:, [4, 6, 8, 14, 15]]
print(df_new)

# PRE-PROCESSING
print("\n", "MISSING VALUE".center(100, "="))
print(df_new.isna().sum())

print("\n", "HANDLING MISSING VALUE".center(100, "="))
df_fix = df_new.dropna(axis = 'index')
print(df_fix)
print("\n", df_fix.describe())

print("\n", "OUTLIER".center(100, "="))
outliers = []
def detect_outlier(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    
    for y in data:
        z_score = (y-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(y)
    return outliers

outlier_runtime = detect_outlier(df_fix['Runtime'])
print('Outlier Runtime = ', outlier_runtime)
outliers.clear()
outlier_imdb = detect_outlier(df_fix['IMDB_Rating'])
print('Outlier IMDB_Rating = ', outlier_imdb)
outliers.clear()
outlier_meta = detect_outlier(df_fix['Meta_score'])
print('Outlier Meta_score = ', outlier_meta)
outliers.clear()
outlier_votes = detect_outlier(df_fix['No_of_Votes'])
print('Outlier No_of_Votes = ', outlier_votes)
outliers.clear()
outlier_gross = detect_outlier(df_fix['Gross'])
print('Outlier Gross = ', outlier_gross)
outliers.clear()

#print("\n", "HANDLING OUTLIERS".center(100, "="))
#z_runtime = np.abs(stats.zscore(df_fix["Runtime"]))
#threshold = 3
#df_fix = df_fix[(z_runtime < 3)]
#print(df_fix)

#z_imdb = np.abs(stats.zscore(df_fix["IMDB_Rating"]))
#threshold = 3
#df_fix = df_fix[(z_imdb < 3)]
#print(df_fix)

#z_meta = np.abs(stats.zscore(df_fix["Meta_score"]))
#threshold = 3
#df_fix = df_fix[(z_meta < 3)]
#print(df_fix)

#z_votes = np.abs(stats.zscore(df_fix["No_of_Votes"]))
#threshold = 3
#df_fix = df_fix[(z_votes < 3)]
#print(df_fix)

#z_gross = np.abs(stats.zscore(df_fix["Gross"]))
#hreshold = 3
#df_fix = df_fix[(z_gross < 3)]
#print(df_fix)

# CHECK THE BEST K VALUE
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])
wcss = []
results = {}
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)    
    labels = kmeans.fit_predict(X)
    db_index = davies_bouldin_score(X, labels)
    results.update({i: db_index})
plt.plot(range(2, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
#plt.show() 

# COUNT THE DBI
print("\nthe value of DBI : ", results)
plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Boulding Index")
#plt.show()

# CLUSTERING
kmeans = KMeans(n_clusters = 4)
kmeans.fit(df_fix)

print("\n", "CLUSTERING ARRAY".center(100, "="))
print(kmeans.labels_)
df_fix['Cluster'] = kmeans.labels_
plt.hist(df_fix['Cluster'])
#plt.show()

print("\n", "CLUSTER DATA CENTER POINTS".center(100, "="))
print(kmeans.cluster_centers_)
cluster = kmeans.labels_
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster0')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'gray', label = 'Centroids')
plt.legend()
#plt.show() 

print("\n", "RESULT OF CLUSTERING".center(100, "="))
print(df_fix)

sns.pairplot(df_fix, kind = 'scatter', hue = 'Cluster', markers = 'o', palette = 'bright')
#plt.show()

# CONFUSION MATRIX
print("\n", "CONFUSION MATRIX".center(100, "="))
print(confusion_matrix(df_fix['Cluster'], cluster))
print(classification_report(df_fix['Cluster'], cluster))

# DATA AFTER CLUSTERING
clus0 = df_fix[df_fix["Cluster"] == 0]
des0 = clus0.describe()
#print(des0)

clus1 = df_fix[df_fix["Cluster"] == 1]
des1 = clus1.describe()
#print(des1)

clus2 = df_fix[df_fix["Cluster"] == 2]
des2 = clus2.describe()
#print(des2)

clus3 = df_fix[df_fix["Cluster"] == 3]
des3 = clus3.describe()
#print(des3)

# CONCLUSION
print("\n", "MEAN OF CLUSTER 0".center(50, "="))
Nilai_Clus0 = des0.iloc[1, 0:5]
print(Nilai_Clus0)

print("\n", "MEAN OF CLUSTER 1".center(50, "="))
Nilai_Clus1 = des1.iloc[1, 0:5]
print(Nilai_Clus1)

print("\n", "MEAN OF CLUSTER 2".center(50, "="))
Nilai_Clus2 = des2.iloc[1, 0:5]
print(Nilai_Clus2)

print("\n", "MEAN OF CLUSTER 3".center(50, "="))
Nilai_Clus3 = des3.iloc[1, 0:5]
print(Nilai_Clus3)