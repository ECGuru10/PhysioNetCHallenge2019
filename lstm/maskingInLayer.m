classdef maskingInLayer < nnet.layer.Layer
    
%     properties (SetAccess = private)
%         InputNames = {'in'}
%         OutputNames = {'out','mask'}           
%     end
    
    
    
    
    methods
        function layer = maskingInLayer(name) 
            layer.NumInputs = 1;
            layer.NumOutputs = 2;
            
%             layer.InputNames = {'in'};
%             layer.OutputNames = {'out','mask'};
            layer.Name = name;
            layer.Description = "layer for N2V";
        end
        
        
        function [Z,mask, memory] = forward(layer,X)
            
            nonzeros=X~=0;
            
            mask=rand(size(nonzeros),'like',nonzeros)>0.9;
            
            
            X(mask&nonzeros)=0;
            
            Z = X;
%             memory=mask;
            memory=mask;
        end
        
                
        function [Z,mask] = predict(layer,X)            
            Z = X;
            mask=zeros(size(X),'like',X);
        end
        
        function [dLdX] = backward(layer,X,Z,mask,dLdZ,dLdmask,memory)
            dLdX = dLdZ;
        end
    end
end