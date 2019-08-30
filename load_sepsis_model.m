function model = load_sepsis_model()
    addpath('training')
    load('model.mat')
    load('x.mat')
    m=load('minv_maxv.mat');
    
    model = {net,x,m.minv,m.maxv};
end
