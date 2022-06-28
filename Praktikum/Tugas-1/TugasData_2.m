% Membaca Data Menggunakan readtable
data1 = readtable('data1.xlsx');
% Mengambil nilai dari 3 variabel, yaitu lokasi, jenis, dan omzet
var1 = data1(:,{'lokasi_omzet_penjualan', 'jenis_komoditi', 'omzet__rp_'})
% Mengambil nilai dari variabel ke 1, yaitu lokasi_omzet_penjualan
lokasi = data1{:,1};
% Mengambil nilai dari variabel ke 2, yaitu jenis_komoditi
jenis = data1{:,2};
% Mengambil nilai dari variabel ke 5, yaitu omzet__rp_
omzet = data1{:,5};