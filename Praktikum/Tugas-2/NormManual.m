% Read Data
open = readtable('BenderlyZwick.csv');
data = open(:,2:6);
set = table2dataset(open);

% Min-Max Normalization
data1 = set.returns;
for i1 = 1:length(data1);
    norm(i1) = (data1(i1)-min(data1))/(max(data1)-min(data1));
end;
disp(norm);

% Z-Score Standarization
data1 = set.returns;
rata = mean(data1);
c = 0;
for i = 1:length(data1);
    d(i) = (data1(i)-rata)^2;
    c = c+d(i)
    sd = sqrt(c/length(data1));
end;
fprintf('Standar Deviasi = %.4f\n Data Baru = ', sd);
for i = 1:length(data1);
    X(i) = (data1(i)-rata)/sd;
end;
disp(X);
