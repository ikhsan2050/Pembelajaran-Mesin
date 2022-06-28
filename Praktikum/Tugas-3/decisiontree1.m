load fisheriris
ctree = fitctree(meas,species);
view(ctree)
view(ctree,'mode','graph')

load carsmall
X = [Horsepower Weight];
rtree = fitrtree(X,MPG,'MinParentSize',30);
view (rtree)
view (rtree,'mode','graph')

