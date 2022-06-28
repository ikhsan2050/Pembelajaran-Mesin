import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.preprocessing import StandardScaler

# Generate Data / Read Data
df = pd.read_csv("foundation_food.csv")
df = df.iloc[:,0:2]
df.info()
print(df.head())

plt.figure(figsize=(6, 6))
plt.scatter(df['fdc_id'],df['NDB_number'] )
plt.xlabel('fdc_id')
plt.ylabel('NDB_number')

scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=['fdc_id','NDB_number'])
plt.show()


# Try with k=2
y_pred = KMeans(n_clusters=2).fit_predict(df)
plt.scatter(df['fdc_id'],df['NDB_number'], c=y_pred)
plt.title("Clustering Number of Blobs")
plt.show()

# Trying to find the best number of clusters
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
 km = KMeans(n_clusters=k)
 km = km.fit(df_scaled[['fdc_id','NDB_number']])
 Sum_of_squared_distances.append(km.inertia_)

# Elbow Method
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Clustering with the best k value
y_pred = KMeans(n_clusters=3).fit_predict(df)
plt.scatter(df['fdc_id'],df['NDB_number'], c=y_pred)
plt.show()

# Confusion Matrix
print("\n", "CONFUSION MATRIX".center(40, "="))
print(pair_confusion_matrix(df['NDB_number'],y_pred))