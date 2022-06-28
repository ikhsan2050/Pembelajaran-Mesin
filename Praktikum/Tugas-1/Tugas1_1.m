% Membaca Data
buka = readtable('Book1.xlsx')
% Mengambil nilai dari 3 variabel, yaitu X1, X2, dan X3
p = buka{:,1:3}
% Mengambil nilai dari variabel ke 2, yaitu X2
q = buka{:,2}
% Mengambil nilai dari variabel ke 1, yaitu X1
m = buka{:,1}
% Mengambil nilai dari variabel ke 3, yaitu X3
n = buka{:,3}