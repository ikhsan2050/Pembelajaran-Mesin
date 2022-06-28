clc;clear
data = xlsread('Narcotics.xlsx')
datatraining = data(1:floor(0.7*length(data)),1:2);
kelastraining = data(1:floor(0.7*length(data)),3);
datatesting = data(floor(0.7*length(data))+1:end,1:2);
kelastesting = data(floor(0.7*length(data))+1:end,3);
a = templateSVM('Standardize',1,'KernelFunction','polynomial');
traini = fitcecoc(datatraining,kelastraining,'Learners',a);
hasil = predict(traini,datatesting);
cek = [hasil kelastesting]
