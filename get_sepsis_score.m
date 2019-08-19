function [score, label] = get_sepsis_score(data, model)
    addpath('lstm')
    addpath('utils')
    net=model{1};
    x=model{2};
    minq=model{3};
    maxq=model{4};
    
    
%     data=normalize015(data,minq,maxq);
%     data=nany_na_nuly(data);

    featureList = {'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS'};
    customScale = [0 1];
    q=-1;
    
    data=FeatureScaling( data, featureList, customScale );
    data=nany_na_x(data, q );
    
    
    sp=0;
    
    score=predict(net,data','MiniBatchSize',1,'SequencePaddingValue',sp);
    
    score=score';
    score=score(:,2);
    score=score(end);
    
    
    label = double(score > x);
end
