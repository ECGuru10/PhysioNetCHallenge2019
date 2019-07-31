classdef maskingOutLayer < nnet.layer.Layer
    
%     properties (SetAccess = private)
%         InputNames = {'in','mask'}
%         OutputNames = {'out'}           
%     end
%     
    
    
    
    methods
        function layer = maskingOutLayer(name) 
            
%             layer.InputNames = {'in','mask'};
%             layer.OutputNames = {'out'} ;
            layer.NumInputs = 2;
            layer.NumOutputs = 1;
            layer.Name = name;
            layer.Description = "layer for N2V";
        end
        
        
        function [Z, memory] = forward(layer,X,mask)
            Z = X;
            memory=mask;
        end
        
                
        function Z = predict(layer,X,mask)            
            Z = X;
        end
        
        function [dLdX,dLdmask] = backward(layer,X,mask,Z,dLdZ,memory)
            nonzeros=X~=0;
            
            dLdX = dLdZ.*mask.*nonzeros;
            dLdmask=zeros(size(mask),'like',mask);
        end
    end
end