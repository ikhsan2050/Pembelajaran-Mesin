% Read Data
open = readtable('BenderlyZwick.csv');
data = open(:,2:6);
x = fillmissing(data,'constant',0,'DataVariables',@isnumeric)

% Default Normalization
Norm1 = normalize(x)
% Normalization with ZScore
Norm2 = normalize(x,'zscore')
% Normalization with Scale
Norm3 = normalize(x,'scale')
% Normalization with Range
Norm4 = normalize(x,'range')