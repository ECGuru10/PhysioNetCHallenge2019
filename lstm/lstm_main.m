clc;clear all;close all force; 

addpath('../../train_test_data2');
addpath('../utils');

load('data.mat');
load('train_test_ind.mat');



drawnow;


XTrain=data(r_train);
YTrain=labels(r_train);
is_septic=contain_sepsis(r_train);


XTest=data(r_test);
YTest=labels(r_test);

w=numel(cat(1,YTrain{:}))/sum(cat(1,YTrain{:}));




maxv= max(cat(1,XTrain{:}),[],1);
minv= min(cat(1,XTrain{:}),[],1);
XTrain=cellfun(@(x)  normalize015(x,minv,maxv),XTrain,'UniformOutput',false);
XTest=cellfun(@(x) normalize015(x,minv,maxv),XTest,'UniformOutput',false);



XTrain=cellfun(@(x) nany_na_nuly(x) ,XTrain,'UniformOutput',false);
XTest=cellfun(@(x) nany_na_nuly(x) ,XTest,'UniformOutput',false);


drawnow;

XTrain=cellfun(@(x) x' ,XTrain,'UniformOutput',false);
YTrain=cellfun(@(x) x' ,YTrain,'UniformOutput',false);
XTest=cellfun(@(x) x' ,XTest,'UniformOutput',false);
YTest=cellfun(@(x) x' ,YTest,'UniformOutput',false);



YTrain_c=cellfun(@(x) categorical(x) ,YTrain,'UniformOutput',false);
YTest_c=cellfun(@(x) categorical(x) ,YTest,'UniformOutput',false);

numResponses = 2;
featureDimension = size(XTrain{1},1);

 
numHiddenUnits = 20;
layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     LayerNorm()
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     LayerNorm()
    fullyConnectedLayer(100)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(50)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(100)
    fullyConnectedLayer(numResponses)
    softmaxLayer
    diceClassificationLayer('out')];


save_name=['cpt'];
mkdir(save_name)

batch=64;
sp=0;
options = trainingOptions('adam', ...
    'GradientThreshold',1,...
    'L2Regularization', 1e-8, ...
    'InitialLearnRate',1e-4,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',15, ...
    'LearnRateDropFactor',0.1, ...
    'ValidationData',{XTest,YTest_c}, ...
    'ValidationFrequency',1000,...
    'MaxEpochs', 35, ...
    'MiniBatchSize', batch, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress',...
    'SequencePaddingValue',sp,...
    'CheckpointPath', save_name);






net = trainNetwork(XTrain,YTrain_c,layers,options);

save(['model.mat'],'net')
load(['model.mat'],'net')



vys=cell(size(XTest));
for k=1:length(XTest)
    k
    vyss=predict(net,XTest{k},'MiniBatchSize',1,'SequencePaddingValue',sp);
    vys{k}=vyss;
end

vys=cellfun(@(x) x(2,:),vys,'UniformOutput',false);



x0=[0.5];
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0];
ub=[1];
nonlcon=[];
x = ga(@(x) pred(x,YTest,vys),1,A,b,Aeq,beq,lb,ub,nonlcon,optimoptions('ga','Display','iter','MaxGenerations',25));


normalized_observed_utility=-pred(x,YTest,vys)


save('x.mat','x')




