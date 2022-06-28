datagolf = readtable("golf-dataset.csv")
data = datagolf(:,1:4);
kelas = datagolf(:,5);
PohonKlasifikasi = fitctree(data,kelas)
view(PohonKlasifikasi,'Mode','graph')