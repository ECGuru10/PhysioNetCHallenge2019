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




featureDimension = size(XTrain{1},1);

numResponses = featureDimension;



numHiddenUnits = 200;
numFc1=200;
numFc2=100;
layers = [sequenceInputLayer(featureDimension,'Name','input')];

for k=1:3
    layers = [...
        layers
        fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '0'])
        lstmLayer(numHiddenUnits,'OutputMode','sequence','Name',['lstm' num2str(k)])
        fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '1'])
        reluLayer('Name',['r' num2str(k) '1'])
        dropoutLayer(0.5,'Name',['do' num2str(k) '1'])
        fullyConnectedLayer(numFc2,'Name',['fc' num2str(k) '2'])
        reluLayer('Name',['r' num2str(k) '2'])
        dropoutLayer(0.5,'Name',['do' num2str(k) '2'])
        fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '3'])
        concatenationLayer(1,3,'Name',['cat' num2str(k) ''])
        ];
    
end

layers = [...
    layers
    fullyConnectedLayer(numFc1,'Name','fc1_center')
    reluLayer('Name','r1_center')
    dropoutLayer(0.5,'Name','do1_center')
    fullyConnectedLayer(numFc2,'Name','fc2_center')
    reluLayer('Name','r2_center')
    dropoutLayer(0.5,'Name','do2_center')
    fullyConnectedLayer(numFc1,'Name','fc3_center')
    
    
    fullyConnectedLayer(numFc1,'Name','fc1_centerx')
    fullyConnectedLayer(5,'Name','fc2_centerx')
    fullyConnectedLayer(numFc1,'Name','fc3_centerx')


    fullyConnectedLayer(numFc1,'Name','fc1_center2')
    reluLayer('Name','r1_center2')
    dropoutLayer(0.5,'Name','do1_center2')
    fullyConnectedLayer(numFc2,'Name','fc2_center2')
    reluLayer('Name','r2_center2')
    dropoutLayer(0.5,'Name','do2_center2')
    fullyConnectedLayer(numFc1,'Name','fc3_center2')
    ];
    
    
 for k=4:6
    layers = [...
        layers
        fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '0'])
        lstmLayer(numHiddenUnits,'OutputMode','sequence','Name',['lstm' num2str(k)])
        fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '1'])
        reluLayer('Name',['r' num2str(k) '1'])
        dropoutLayer(0.5,'Name',['do' num2str(k) '1'])
        fullyConnectedLayer(numFc2,'Name',['fc' num2str(k) '2'])
        reluLayer('Name',['r' num2str(k) '2'])
        dropoutLayer(0.5,'Name',['do' num2str(k) '2'])
        fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '3'])
        concatenationLayer(1,3,'Name',['cat' num2str(k) ''])
        ];
    
end
    
    
    
 layers = [...
    layers
    fullyConnectedLayer(numFc1,'Name','fc1_final')
    reluLayer('Name','r1_final')
    dropoutLayer(0.5,'Name','do1_final')
    fullyConnectedLayer(numFc2,'Name','fc2_final')
    reluLayer('Name','r2_final')
    dropoutLayer(0.5,'Name','do2_final')
    fullyConnectedLayer(numFc1,'Name','fc3_final')
    
    fullyConnectedLayer(numFc1,'Name','fc1_final2')
    reluLayer('Name','r1_final2')
    dropoutLayer(0.5,'Name','do1_final2')
    fullyConnectedLayer(numFc2,'Name','fc2_final2')
    reluLayer('Name','r2_final2')
    dropoutLayer(0.5,'Name','do2_fina2')
    fullyConnectedLayer(numFc1,'Name','fc3_final2')
    
    fullyConnectedLayer(numFc1,'Name','fc1_final3')
    reluLayer('Name','r1_final3')
    dropoutLayer(0.5,'Name','do1_final3')
    fullyConnectedLayer(numFc2,'Name','fc2_final3')
    reluLayer('Name','r2_final3')
    dropoutLayer(0.5,'Name','do2_fina3')
    fullyConnectedLayer(numFc1,'Name','fc3_final3')
    
    fullyConnectedLayer(numResponses,'Name','fcfinal_final')
    regressionLayer('Name','routput')];


layers=layerGraph(layers);
layers=connectLayers(layers,'input','cat1/in2');
layers=connectLayers(layers,'input','cat1/in3');
for k=1:6-1
    layers=connectLayers(layers,['cat' num2str(k) ''],['cat' num2str(k+1) '/in2']);
    layers=connectLayers(layers,'input',['cat' num2str(k+1) '/in3']);
end



save_name=['cpt_auto'];
mkdir(save_name)

batch=64;
sp=0;
options = trainingOptions('adam', ...
    'GradientThreshold',1,...
    'L2Regularization', 1e-8, ...
    'InitialLearnRate',1e-3,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',0.1, ...
    'ValidationData',{XTest,XTest}, ...
    'ValidationFrequency',1000,...
    'MaxEpochs', 35, ...
    'MiniBatchSize', batch, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress',...
    'SequencePaddingValue',sp,...
    'CheckpointPath', save_name);






net = trainNetwork(XTrain,XTrain,layers,options);




lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,{'fc3_centerx','fc1_center2','r1_center2','do1_center2','fc2_center2','r2_center2','do2_center2','fc3_center2'});

 for k=4:6
    lgraph = removeLayers(lgraph,['fc' num2str(k) '0']);
    lgraph = removeLayers(lgraph,['lstm' num2str(k)]);
    lgraph = removeLayers(lgraph,['fc' num2str(k) '1']);
    lgraph = removeLayers(lgraph,['r' num2str(k) '1']);
    lgraph = removeLayers(lgraph,['do' num2str(k) '1']);
    lgraph = removeLayers(lgraph,['fc' num2str(k) '2']);
    lgraph = removeLayers(lgraph,['r' num2str(k) '2']);
    lgraph = removeLayers(lgraph,['do' num2str(k) '2']);
    lgraph = removeLayers(lgraph,['fc' num2str(k) '3']);
    lgraph = removeLayers(lgraph,['cat' num2str(k) '']);
     
 end

lgraph = removeLayers(lgraph,{'fc1_final','r1_final','do1_final','fc2_final','r2_final','do2_final','fc3_final'});
lgraph = removeLayers(lgraph,{'fc1_final2','r1_final2','do1_final2','fc2_final2','r2_final2','do2_fina2','fc3_final2'});
lgraph = removeLayers(lgraph,{'fc1_final3','r1_final3','do1_final3','fc2_final3','r2_final3','do2_fina3','fc3_final3','fcfinal_final'});


lgraph = connectLayers(lgraph,'fc2_centerx','routput');


net2 = assembleNetwork(lgraph);

save(['model_auto.mat'],'net')
save(['model_auto2.mat'],'net2')

load(['model_auto2.mat'])





vys=cell(size(XTest));
for k=1:length(XTest)
    k
    vyss=predict(net2,XTest{k},'MiniBatchSize',1,'SequencePaddingValue',sp);
    vys{k}=vyss;
end




