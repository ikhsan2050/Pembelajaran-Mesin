X = readtable("Titanic.csv")
data = X(:,1:4);
kelas = X(:,4);
PohonKlasifikasi = fitctree(data,kelas)
view(PohonKlasifikasi,'Mode','graph')