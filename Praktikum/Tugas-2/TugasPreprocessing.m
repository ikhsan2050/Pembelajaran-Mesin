a = [2 1 29 28 100 38 29 2 13];
b = isoutlier(a);

a2 = [94 73 47 34 73 840 63 44 75 62 72 81 73];
c2 = std(a2);
Outlier = 3*c2;
b2 = filloutliers(a2,'nearest','mean');

a3 = [23 47 29 384 62 36 42 93 84 76 23 84];
[m3,n3] = rmoutliers(a3,'mean');
k3 = rmoutliers(a3,'mean');

a4 = [1 2 NaN 3 NaN NaN 8];
b4 = ismissing(a4);

Tanggal = datetime({'2022-03-16 08:03:05';'2022-03-16 10:03:17';'2022-03-16 12:03:13'});
Temperatur = [35.3;37.1;45.3];
Arah_Angin = categorical({'NW';'NW';'N'});
TT = timetable(Tanggal,Temperatur,Arah_Angin);
disp(TT)
TT.Tanggal(3) = missing;
TT.Temperatur(3) = missing;
TT.Arah_Angin(3) = missing;
disp(TT)

Temperatur = [35.3;NaN;41.3];
Arah_Angin = categorical({'NW';'N';'NW'});
TT = table(Temperatur,Arah_Angin);
disp(TT)
F = fillmissing(TT,'constant',0,'DataVariables',@isnumeric);
disp(F)

Temperatur = [35.3;NaN;41.3];
Arah_Angin = categorical({'NW';'';'N'});
TT = table(Temperatur,Arah_Angin);
disp(TT);
F = fillmissing(TT,'previous','DataVariables',{'Arah_Angin'});
G = fillmissing(F,'pchip','DataVariables',{'Temperatur'});
disp(G);

Data = readtable('BenderlyZwick.csv')
Outlier = isoutlier(Data)
b5 = filloutliers(Data,0)
l5 = filloutliers(Data,'nearest','DataVariables',{'growth'})
k5 = rmoutliers(Data)
missing1 = ismissing(b5)
missing2 = ismissing(k5)
x = fillmissing(b5,'constant',0,'DataVariables',@isnumeric)
y = fillmissing(k5,'constant',0,'DataVariables',@isnumeric)
z = fillmissing(b5,'previous','DataVariables',{'inflation'})
normalisasi = normalize(x,'zscore')