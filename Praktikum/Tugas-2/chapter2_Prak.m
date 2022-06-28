a1 = [1 4 17  48 10 7 13 2 3]
b1 = isoutlier(a1)

a2 = [57 59 65 70 59 58 57 58 350 61 62 60 62 58 57]
c2 = std(a2)
Outlier = 3*c2
b2 = filloutliers(a2,'nearest','mean')

a3 = [57 59 65 70 59 58 57 58 350 61 62 60 62 58 57]
[m3, n3] = rmoutliers(a3,'mean')
k3 = rmoutliers(a3,'mean')

a4 = [2 4 NaN 6 NaN NaN 9]
b4 = ismissing(a4)

Tanggal = datetime({'2015-12-18 08:03:05';'2015-12-18 10:03:17';'2015-12-18 12:03:13'});
Temperatur = [37.3;39.1;42.3];
Arah_Angin = categorical({'NW';'NW';'N'});
TT = timetable(Tanggal,Temperatur,Arah_Angin);
disp(TT)
TT.Tanggal(3) = missing;
TT.Temperatur(3) = missing;
TT.Arah_Angin(3) = missing;
disp(TT)

Temperatur = [37.3;NaN;42.3];
Arah_Angin = categorical({'NW';'N';'NW'});
TT = table(Temperatur,Arah_Angin);
disp(TT)
F = fillmissing(TT,'constant',0,'DataVariables',@isnumeric);
disp(F)

Temperatur = [37.3;NaN;42.3];
Arah_Angin = categorical({'NW';'';'N'});
TT = table(Temperatur,Arah_Angin);
disp(TT)
F = fillmissing(TT,'previous','DataVariables',{'Arah_Angin'})
G = fillmissing(F,'pchip','DataVariables',{'Temperatur'});
disp(G)

Data = readtable('Book1.xlsx')
Outlier = isoutlier(Data)
b5 = filloutliers(Data,0)
l5 = filloutliers(Data,'nearest','DataVariables',{'X2'})
k5 = rmoutliers(Data)
missing1 = ismissing(b5)
missing2 = ismissing(k5)
x = fillmissing(b5,'constant',0,'DataVariables',@isnumeric)
y = fillmissing(k5,'constant',0,'DataVariables',@isnumeric)
z = fillmissing(b5,'previous','DataVariables',{'X3'})
normalisasi = normalize(x,'zscore')


