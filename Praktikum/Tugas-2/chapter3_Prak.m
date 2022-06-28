v = [7 10 15 20 25]
for i = 1:length(v)
    nor(i) = (v(i)-min(v))/(max(v)-min(v))
end
disp(nor)

a = [7 10 15 20 25]
rata2 = mean(a)
c = 0

for i = 1:length(a)
    d(i) = (a(i)-rata2)^2
    c = c+d(i)
    sd = sqrt(c/length(a))
end
fprintf('Standar Deviasi = %.4f\n Data Baru = ')
for i = 1:length(a)
    X(i) = (a(i)-rata2)/sd
end
disp(X)

a2 = readtable('Book1.xlsx')
Normalisasi1 = normalize(a2)
Normalisasi2 = normalize(a2,'zscore')
Normalisasi3 = normalize(a2,'scale')
Normalisasi4 = normalize(a2,'range')




