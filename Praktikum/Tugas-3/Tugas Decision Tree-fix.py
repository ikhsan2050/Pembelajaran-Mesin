import pandas as pd
from sklearn import preprocessing, tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# READ DATA & CHECK THE VARIABLE
data = pd.read_csv("gender_classification_v7.csv")
df = pd.DataFrame(data)
df.info()
#print(df)

# TARGET VARIABLE
gender = df['gender']
#print(gender)

# CHECK MISSING VALUE
miss = df.isna().sum()
#print(miss)

# CONVERT THE TARGET VALUE INTO NUMERIC / BINARY
d = {"Male":1, "Female":0}
df['gender'] = df['gender'].map(d)

# MAKE THE CONDITION INTO FEATURES FOR THE TARGET VARIABLE
features = ["long_hair", "forehead_width_cm", "forehead_height_cm","nose_wide", "nose_long", "lips_thin", "distance_nose_to_lip_long"]

# MAKE THE TARGET AND CONDITION SIMPLER
X = df[features]
Y = df['gender']
#print(X)
#print(Y)

# CLASSIFICATE THE VARIABLE
clf = DecisionTreeClassifier(max_depth = 3)
model = clf.fit(X, Y)

# RESULT THE CLASSIFICATION IN TEXT
text_representation = tree.export_text(clf)
#print(text_representation)

# VISUALIZATING THE CLASSIFICATION
fig = plt.figure(figsize=(25,20), dpi=(1000))
plot = tree.plot_tree(clf, 
                   feature_names = features,
                   class_names = gender,
                   filled = True)

# SAVING THE IMAGE CLASSIFICATION RESULT
fig.savefig("gender_classification.png")