function [XTrain_peaces,YTrain_peaces]=divide_to_6_h_peaces(XTrain,YTrain)


lengths=cellfun(@(x) size(x,2),XTrain,'UniformOutput',true);

XTrain_peaces=zeros(sum(lengths-5),size(XTrain{1},1));

couter=0;
for k=1:length(XTrain)
    x=XTrain{k};
    y=YTrain{k};
    
    for kk=6:size(x,2)
        couter=couter+1;
        
        
    end
    
    
    
    
end



