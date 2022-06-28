clc;clear
data = xlsread('datasbmbaru1.xlsx')
datatraining = data(1:floor(0.7*length(data)),1:4);
kelastraining = data(1:floor(0.7*length(data)),5);
datatesting = data(floor(0.7*length(data))+1:end,1:4);
kelastesting = data(floor(0.7*length(data))+1:end,5);
a = templateSVM('Standardize',1,'KernelFunction','polynomial');
traini = fitcecoc(datatraining,kelastraining,'Learners',a);
hasil = predict(traini,datatesting);
cek = [hasil kelastesting]
