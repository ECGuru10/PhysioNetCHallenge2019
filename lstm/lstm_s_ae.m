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


load('../model_auto2.mat')
% for k =1:length(XTrain)
%     x=XTrain{k};
%     xx=predict(net2,x,'MiniBatchSize',1,'SequencePaddingValue',0);
%     XTrain{k}=cat(1,x,xx);
%     
% end
% for k =1:length(XTest)
%     x=XTest{k};
%     xx=predict(net2,x,'MiniBatchSize',1,'SequencePaddingValue',0);
%     XTest{k}=cat(1,x,xx);
% end

lgraph = layerGraph(net2);



YTrain_c=cellfun(@(x) categorical(x) ,YTrain,'UniformOutput',false);
YTest_c=cellfun(@(x) categorical(x) ,YTest,'UniformOutput',false);

numResponses = 2;
featureDimension = size(XTrain{1},1);

% LayerNorm(['ln' num2str(k)])

numHiddenUnits = 300;
numFc1=200;
numFc2=100;
blocks=4;
layers = [];
for k=1:blocks
    layers = [...
        layers
        fullyConnectedLayer(numFc1,'Name',['qfc' num2str(k) '0'])
        lstmLayer(numHiddenUnits,'OutputMode','sequence','Name',['qlstm' num2str(k)])
        fullyConnectedLayer(numFc1,'Name',['qfc' num2str(k) '1'])
        reluLayer('Name',['qr' num2str(k) '1'])
        dropoutLayer(0.01,'Name',['qdo' num2str(k) '1'])
        fullyConnectedLayer(numFc2,'Name',['qfc' num2str(k) '2'])
        reluLayer('Name',['qr' num2str(k) '2'])
        dropoutLayer(0.01,'Name',['qdo' num2str(k) '2'])
        fullyConnectedLayer(numFc1,'Name',['qfc' num2str(k) '3'])
        concatenationLayer(1,3,'Name',['qcat' num2str(k) ''])];
    
end
layers = [...
    layers
    fullyConnectedLayer(numFc1,'Name','qfc1_final')
    reluLayer('Name','qr1_final')
    dropoutLayer(0.01,'Name','qdo1_final')
    fullyConnectedLayer(numFc2,'Name','qfc2_final')
    reluLayer('Name','qr2_final')
    dropoutLayer(0.01,'Name','qdo2_final')
    fullyConnectedLayer(numFc1,'Name','qfc3_final')
    
    fullyConnectedLayer(numFc1,'Name','qfc1_final2')
    reluLayer('Name','qr1_final2')
    dropoutLayer(0.01,'Name','qdo1_final2')
    fullyConnectedLayer(numFc2,'Name','qfc2_final2')
    reluLayer('Name','qr2_final2')
    dropoutLayer(0.01,'Name','qdo2_fina2')
    fullyConnectedLayer(numFc1,'Name','qfc3_final2')
    
    fullyConnectedLayer(numFc1,'Name','qfc1_final3')
    reluLayer('Name','qr1_final3')
    dropoutLayer(0.01,'Name','qdo1_final3')
    fullyConnectedLayer(numFc2,'Name','qfc2_final3')
    reluLayer('Name','qr2_final3')
    dropoutLayer(0.01,'Name','qdo2_fina3')
    fullyConnectedLayer(numFc1,'Name','qfc3_final3')
    
    fullyConnectedLayer(numResponses,'Name','qfcfinal_final')
    softmaxLayer('Name','qsm')
    diceClassificationLayer('qout')];

lgraph=removeLayers(lgraph,{'routput'});

layers=addLayers(lgraph,layers);
% layers=layerGraph(layers);
layers=connectLayers(layers,'fc2_centerx','qcat1/in2');
layers=connectLayers(layers,'input','qcat1/in3');
for k=1:blocks-1
    layers=connectLayers(layers,['qcat' num2str(k) ''],['qcat' num2str(k+1) '/in2']);
    layers=connectLayers(layers,'fc2_centerx',['qcat' num2str(k+1) '/in3']);
end
layers=connectLayers(layers,'fc2_centerx','qfc10');

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
    'LearnRateDropPeriod',10, ...
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
load(['model.mat'])



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


save('minv_maxv.mat','minv','maxv')




