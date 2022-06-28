import imp
import pandas as pd
import numpy as np
from sklearn import preprocessing, tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

data = pd.read_csv("gender_classification_v7.csv")
df = pd.DataFrame(data)
#print(df)

gender = df['gender']
#print(gender)

miss = df.isna().sum()
#print(miss)

d = {"Male":1, "Female":0}
df['gender'] = df['gender'].map(d)
features = ["long_hair", "forehead_width_cm", "forehead_height_cm", "nose_wide", "nose_long", "lips_thin", "distance_nose_to_lip_long"]

X = df[features]
Y = df['gender']

print(X)
print(Y)

clf = DecisionTreeClassifier(max_depth = 4)
model = clf.fit(X, Y)
text_representation = tree.export_text(clf)
#print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names = features,
                   class_names = gender,
                   filled = True)
fig.savefig("decistion_tree.png")

#clf.fit(X, Y)
#tree.plot_tree(clf)
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
#plot_tree(clf, feature_names = features, filled = True)
#fig.savefig('imagename.png')


#dtree = DecisionTreeClassifier()
#dtree = dtree.fit(X, Y)
#final = tree.export_graphviz(dtree, out_file = None, feature_names=features)

#graph = pydotplus.graph_from_dot_data(final)
#graph.write_png('mydecisiontree.png')

#img = pltimg.imread('mydecisiontree.png')
#imgplot = plt.imshow(img)
#plt.show()
