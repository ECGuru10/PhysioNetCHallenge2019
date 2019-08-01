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







