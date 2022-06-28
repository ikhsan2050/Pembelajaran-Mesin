%Membaca data dari excel
filename = 'gender_training.xlsx';
sheet = 1;
xlRange = 'A2:C76';
Data = xlsread (filename, sheet, xlRange);
data_latih = Data(:,1:2)'; 
target_latih = Data(:,3)';
% sheet = 2;
% xlRange = 'A2:H1001';
% Data = xlsread (filename, sheet, xlRange);
% data_latih = Data(:,[1,6])'; 
% target_latih = Data(:,8)';
[m, n] = size(data_latih);

% Membaca data dari excel
% Data = xlsxsread('mobilegender-train.xlsx');
% data_latih = Data(:,1:2)';
% target_latih = Data (:, 3)';
% [m, n] = size(data_latih);

% Pembuatan JST
% Arsitek jaringan yang dipakai adalah 3-3-1, artinya 3 neuron input
% X,Y,Z), 3 neuron hidden (karena Inputnya ada 3 neuron maka neuron hiddennya
% Fungsi Aktivasi di hidden layer menggunakan 'logsig', di output layer menggunakan 'purelin'
% Model JST yang digunakan gradien descent maka fungsi aktivasinya adalah traingdx
net = newff(minmax(data_latih),[2 1],{'logsig', 'purelin'}, 'traingdx');

% Memberikan nilai untuk mempengaruhi proses Training
net.performFcn= 'mse';
net.trainParam.goal = 0.0001; % Errornya (0 sampai 1)
net.trainParam.show = 20; % Boleh diganti
net.trainParam.epochs = 1500; % Banyaknya epoch / iterasi training
net.trainParam.mc = 0.95;
net.trainParam.lr = 1; % Nilai learning Rate (0 sampai 1)

% Proses training
[net_keluaran, tr, Y, E] = train(net, data_latih, target_latih);

% Hasil setelah pelatihan
bobot_hidden = net_keluaran.IW {1,1};
bobot_keluaran = net_keluaran.LW{2,1};
bias_hidden = net_keluaran.b{1,1};
bias_keluaran = net_keluaran.b{2,1};
jumlah_iterasi = tr.num_epochs;
nilai_keluaran = Y;
nilai_error = E;
error_MSE = (1/n)*sum(nilai_error.^2);

save ('C:\ikhsan\UNAIR\SEMESTER 4\PEMBELAJARAN MESIN (PRAKTIKUM)\Tugas SVM dan NN\gender_keluaran.mat')

% Hasil prediksi
hasil_latih = sim(net_keluaran, data_latih);

% Performansi hasil prediksi
target_latih_asli = target_latih;

figure,
plotregression(target_latih_asli, hasil_latih, 'Regression')
figure,
plotperform(tr)

% Gambar JST
figure,
plot(hasil_latih, 'bo-')
hold on
plot(target_latih_asli, 'ro-')
hold off
grid on
title(strcat(['Grafik Keluaran JST vs Target dengan nilai MSE = ', num2str(error_MSE)]))
xlabel('Pola ke-')
ylabel('MSE')
legend('Keluaran JST', 'Target', 'Location', 'Best')

% load jaringan yang sudah dibuat pada proses pelatihan
load('C:\ikhsan\UNAIR\SEMESTER 4\PEMBELAJARAN MESIN (PRAKTIKUM)\Tugas SVM dan NN\gender_keluaran.mat')

% Proses membaca data uji dari excel
filename = 'gender_test.xlsx';
sheet = 1;
xlRange = 'A2:C76';
Data = xlsread (filename, sheet, xlRange);
data_uji = Data(:,1:2)';
target_uji = Data (:,3)';
% sheet = 2
% xlRange = 'A1002:H1101';
% Data = xlsread (filename, sheet, xlRange);
% data_uji = Data(:,[1,6])';
% target_uji = Data (:,8)';
[m, n] = size (data_uji);

% Hasil prediksi
hasil_uji = sim(net_keluaran, data_uji);
nilai_error = abs(hasil_uji - target_uji)

% Performansi hasil prediksi
error = (1/n)*sum(nilai_error.^1);
Akurasi = (1-error)*100