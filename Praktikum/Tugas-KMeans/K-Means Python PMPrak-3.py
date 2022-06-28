import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

df = pd.read_csv('food_calorie_conversion_factor.csv')

print("\n", "MISSING VALUE".center(100, "="))
print(df.isna().sum())

plt.scatter(df.protein_value, df.carbohydrate_value, s=10, c="c", marker="o", alpha=1)
plt.show()

# print("\n", "OUTLIER".center(100, "="))
# outliers = []
# def detect_outlier(data):
#     threshold = 3
#     mean = np.mean(data)
#     std = np.std(data)
    
#     for y in data:
#         z_score = (y-mean)/std
#         if np.abs(z_score)>threshold:
#             outliers.append(y)
#     return outliers

# outlier_id = detect_outlier(df['food_nutrient_conversion_factor_id'])
# print('Outlier ID = ', outlier_id)
# outliers.clear()
# outlier_value = detect_outlier(df['value'])
# print('Outlier Value = ', outlier_value)
# outliers.clear()

# import scipy.stats as stats
# print("\n", "HANDLING OUTLIERS".center(100, "="))
# z_runtime = np.abs(stats.zscore(df["value"]))
# threshold = 3
# df = df [(z_runtime < 3)]
# print(df.count())

X, y = make_blobs(n_samples=1500, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# trying to find the best number of clusters
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X,y)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# try with k=2
y_pred = KMeans(n_clusters=2).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

from sklearn.metrics.cluster import pair_confusion_matrix
pair_confusion_matrix(y,y_pred)