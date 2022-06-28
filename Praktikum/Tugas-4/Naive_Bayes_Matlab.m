load fisheriris % load the sample data 
y = species;
X = meas;
feat=X; label=y;
Dist = 'normal';   % Dist adalah nama distribusi yang digunakan
                   % 'normal' maka menggunakan distribusi Gaussian (Normal)   
rng('default');
% Divide data into k-folds
fold=cvpartition(label,'kfold',10); 
% Pre
pred2=[]; ytest2=[]; Afold=zeros(10,1); 
% Naive Bayes start
for i = 1:10
    % Call index of training & testing sets
    trainIdx=fold.training(i); testIdx=fold.test(i);
    % Call training & testing features and labels
    xtrain=feat(trainIdx,:); ytrain=label(trainIdx);
    xtest=feat(testIdx,:); ytest=label(testIdx);
    % Training the model
    Model=fitcnb(xtrain,ytrain,'Distribution',Dist);
    % Perform testing 
    Pred0 = predict(Model,xtest); 
    pred2=[pred2(1:end);Pred0]; ytest2=[ytest2(1:end);ytest];
end
%confusion chart for Data Testing
fig = figure;
confchart=confusionchart(ytest,Pred0');
confchart.Title = 'Confusion Matrix Using Naive Bayes for Data Testing';
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position;
 
% Overall confusion matrix for Data Training and Testing
confmat=confusionmat(ytest2,pred2); 
%confusion chart for Overall for Data Training and Testing
fig = figure;
confchartAll=confusionchart(ytest2,pred2');
confchartAll.Title = 'Overall Confusion Matrix Using Naive Bayes for Data Training and Testing';
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position; 