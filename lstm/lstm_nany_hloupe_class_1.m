clc;clear all;close all force; 

global YTest vys


addpath('../../train_test_data');


load('data.mat');
load('train_test_ind.mat');





% [data,nan_pos]=odstran_nany_hloupe(data);
% data=cellfun(@(x) x(:,1:7) ,data,'UniformOutput',false);

% for k=1:length(labels)
%     ll=labels{k};
%     t=find(ll,1);
%     if t==1
%         drawnow;
%     end
% end


drawnow;


XTrain=data(r_train);
YTrain=labels(r_train);
is_septic=contain_sepsis(r_train);

% 
% [data_train_aug,labels_train_aug]=dataPermutator(XTrain,YTrain);
% [data_train_aug,labels_train_aug]=dataAugmenter(XTrain,YTrain,is_septic);
% XTrain=[XTrain,data_train_aug];
% YTrain=[YTrain,labels_train_aug];

% YTrain=contain_sepsis(r_train);

XTest=data(r_test);
YTest=labels(r_test);
% YTest=contain_sepsis(r_test);

w=numel(cat(1,YTrain{:}))/sum(cat(1,YTrain{:}));

% 
% mu=nanmean(cat(1,XTrain{:}),1);
% sig=nanstd(cat(1,XTrain{:}),0,1);
% XTrain=cellfun(@(x) (x-repmat(mu,[size(x,1),1]))./repmat(sig,[size(x,1),1]) ,XTrain,'UniformOutput',false);
% XTest=cellfun(@(x) (x-repmat(mu,[size(x,1),1]))./repmat(sig,[size(x,1),1])  ,XTest,'UniformOutput',false);


maxv= max(cat(1,XTrain{:}),[],1);
minv= min(cat(1,XTrain{:}),[],1);
XTrain=cellfun(@(x)  normalize015(x,minv,maxv),XTrain,'UniformOutput',false);
XTest=cellfun(@(x) normalize015(x,minv,maxv),XTest,'UniformOutput',false);



% 
% for k=1:length(data)
%     data{k} =[data{k} nan_pos{k}*2-1] ;
% end


XTrain=cellfun(@(x) nany_na_nuly(x) ,XTrain,'UniformOutput',false);
XTest=cellfun(@(x) nany_na_nuly(x) ,XTest,'UniformOutput',false);



drawnow;

XTrain=cellfun(@(x) x' ,XTrain,'UniformOutput',false);
YTrain=cellfun(@(x) x' ,YTrain,'UniformOutput',false);
XTest=cellfun(@(x) x' ,XTest,'UniformOutput',false);
YTest=cellfun(@(x) x' ,YTest,'UniformOutput',false);

% XTrain0=XTrain;
% YTrain0=YTrain;
% for k=1:round(90)
%     XTrain=[XTrain,XTrain0(find(contain_sepsis(r_train)))];
%     YTrain=[YTrain,YTrain0(find(contain_sepsis(r_train)))];
% end

% XTrain=cellfun(@(x) fliplr(x) ,XTrain,'UniformOutput',false);
% YTrain=cellfun(@(x) fliplr(x) ,YTrain,'UniformOutput',false);
% XTest=cellfun(@(x) fliplr(x) ,XTest,'UniformOutput',false);
% YTest=cellfun(@(x) fliplr(x) ,YTest,'UniformOutput',false);

YTrain_c=cellfun(@(x) categorical(x) ,YTrain,'UniformOutput',false);
YTest_c=cellfun(@(x) categorical(x) ,YTest,'UniformOutput',false);
% YTrain_c=YTrain;
% YTest_c=YTest;

% numResponses = size(YTrain{1},1);
numResponses = 2;
% numResponses = 1;
featureDimension = size(XTrain{1},1);

%bilstmLayer
numHiddenUnits = 20;
layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
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
%     weightedClassificationLayer([1 w],'out')
%     sigLayer
%     uLoss];

save_name=['modely_015-3_dice'];
mkdir(save_name)

batch=64;
sp=0;
%0.0001 150 16b           'GradientThreshold',1,...
options = trainingOptions('adam', ...
    'GradientThreshold',1,...
    'L2Regularization', 1e-5, ...
    'InitialLearnRate',1e-4,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',70, ...
    'LearnRateDropFactor',0.1, ...
    'ValidationData',{XTest,YTest_c}, ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', batch, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress',...
    'SequencePaddingValue',sp,...
    'CheckpointPath', save_name);



lengths=cellfun(@(x) size(x,2),XTrain,'UniformOutput',true);


net = trainNetwork(XTrain,YTrain_c,layers,options);
% save([save_name '/model.mat'],'net')

% load([save_name '/model.mat'],'net')
% load('model015_dice_u0.75_t1.9302e-04 .mat')

vys=cell(size(XTest));
for k=1:length(XTest)
    k
    xx=XTest{k};
    tmp=randperm(length(lengths),batch);
    tmp=lengths(tmp);
    tmp=max(tmp);
    size0=size(xx,2);
    if tmp>size0
        pad=tmp-size0;
        xx=padarray(xx,[0 pad],sp,'post');
    end
    
    vyss=predict(net,{xx},'MiniBatchSize',1,'SequencePaddingValue',sp);
    vyss=vyss{1};
    vyss=vyss(:,1:size0);
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
% x = simulannealbnd(@pred,x0,lb,ub,optimset('Display','iter','MaxFunEvals',100))
x = ga(@pred,1,A,b,Aeq,beq,lb,ub,nonlcon,optimoptions('ga','Display','iter','MaxGenerations',25));
% x = fminsearch(@pred,x0);
% x = fminbnd(@pred,lb,ub,optimset('Display','iter','MaxFunEvals',1500,'TolX',1e-8));
% x = fmincon(@pred,x0,A,b,Aeq,beq,lb,ub); 

normalized_observed_utility=-pred(x)






% 
% aaa=cat(2,YTest_c{:});
% aa=grp2idx(aaa);


%prvnich 7 0.53
%vsechny nn 0.610  -bez normalizace lre-3  0.999  


% layers=net.Layers;

