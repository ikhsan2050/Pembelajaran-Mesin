import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv(r"C:\Users\ACER\Downloads\Titanic.csv")
df.info()
d = {'First': 0, 'Second': 1, 'Crew': 2, 'Third':3}
df['Class'] = df['Class'].map(d)
d = {'Adult': 1, 'Child': 2}
df['Age'] = df['Age'].map(d)
d = {'Female': 1, 'Male': 2}
df['Sex'] = df['Sex'].map(d)
d = {'Yes': 1, 'No': 0}
df['Survived'] = df['Survived'].map(d)

features = ['Class', 'Age', 'Sex']

X = df[features]
y = df['Survived']

print(X)
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()






