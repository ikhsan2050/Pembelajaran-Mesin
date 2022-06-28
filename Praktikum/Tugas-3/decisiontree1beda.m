X=readtable('Titanic.csv');
disp(X)
ctree = fitctree(X,"Survived",'MinParentSize',30);
view(ctree)
view(ctree,'mode','graph')

Y=readtable('salarydataset.xlsx');
rtree = fitrtree(Y,'Salary~Level','MinParentSize',1);
view (rtree)
view (rtree,'mode','graph')







