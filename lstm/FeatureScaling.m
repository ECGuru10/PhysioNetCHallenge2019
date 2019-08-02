function outputData = FeatureScaling( inputData, featureList, scale )
    
    numFeatures = numel( featureList );
    inputDataSize = size( inputData );
    
    scaleLowerBound = min( scale );
    scaleUpperBound = max( scale );
    scaleRange = abs( scaleUpperBound - scaleLowerBound );
    
    
    %Size check
    if all( inputDataSize ~= numFeatures )
        return
    end
    
    if inputDataSize(1) == numFeatures
        inputData = transpose( inputData );
        inputDataSize = fliplr( inputDataSize );
    end
        
    %Get feature statistical parameters
    featureStat = InitializeFeatureParams();
    
    %Feature check and parameter extraction
    [ ~, minVal, maxVal, isFeatureExist ] = GetFeatureParams( featureStat, featureList );
    
    %Remove feature that hasnt been recognized by the function
    inputData( :,  ~isFeatureExist ) = [];
    
    %Repeting feature params    
    minVal = repmat( minVal, inputDataSize(1), 1 );
    maxVal = repmat( maxVal, inputDataSize(1), 1 );
    
    %Data outside interval
    idxAboveMax = ( inputData > maxVal);
    idxBelowMin = ( inputData < minVal);
    
    %Thresholding    
    inputData( idxAboveMax ) = maxVal ( idxAboveMax );
    inputData( idxBelowMin) = minVal( idxBelowMin );
    
    %Scales data to custom range;
    outputData = ( inputData - minVal )./( maxVal - minVal );
    outputData = scaleRange.*outputData;
    outputData = outputData + scaleLowerBound;       
    
end

function [medVal, minVal, maxVal, isFeatureExist] = GetFeatureParams( featureStat, featureName )
    
    numFeatures = numel( featureName );
    isFeatureExist = false( 1, numFeatures );
    medVal = zeros( 1, numFeatures );
    minVal = zeros( 1, numFeatures );
    maxVal = zeros( 1, numFeatures );
    
    for idx = 1:numFeatures
        
        currentFeatureName = featureName{ idx };
        
        if isfield( featureStat, currentFeatureName )
            
            %Return median, lower and upper threshold
            medVal( idx )= featureStat.( currentFeatureName )( 1 );
            minVal( idx ) = featureStat.( currentFeatureName )( 2 );
            maxVal( idx ) = featureStat.( currentFeatureName )( 3 );
            
            %exist flag
            isFeatureExist( idx ) = true;
            
        else
            continue
        end                       
        
    end
    
    %Removing invalid feature column
    medVal( ~isFeatureExist ) = [];
    minVal( ~isFeatureExist ) = [];
    maxVal( ~isFeatureExist ) = [];   
    
end

function featureStat = InitializeFeatureParams()
    
    %Manualy adjusted feature parameteres
    featureStat.HR = [84 30 200];
    featureStat.O2Sat = [98 40 100];
    featureStat.Temp = [37 30 42];
    featureStat.SBP = [118 40 200];
    featureStat.MAP = [77 30 160];
    featureStat.DBP = [58 20 150];
    featureStat.Resp = [18 1 50];
    featureStat.BaseExcess = [0 -15 15];
    featureStat.HCO3 = [24 5 45];
    featureStat.FiO2 = [0.5 0 1];
    featureStat.pH = [7.38 6.5 8];
    featureStat.PaCO2 = [40 10 90];
    featureStat.SaO2 = [97 50 100];
    featureStat.BUN = [17 1 180];
    featureStat.Calcium = [8.3 1 25];
    featureStat.Chloride = [105 25 145];
    featureStat.Creatinine = [0.92 0.1 35];
    featureStat.Glucose = [126 11 750];
    featureStat.Lactate = [1.8 0.2 20];
    featureStat.Magnesium = [2 0.2 10];
    featureStat.Phosphate = [3.3 0.2 20];
    featureStat.Potassium = [4.1 1 10];
    featureStat.Bilirubin_total = [0.9 0.1 25.9];
    featureStat.Hct = [30.2 10 55];
    featureStat.Hgb = [10.3 2 22];
    featureStat.PTT = [32.4 10 150];
    featureStat.WBC = [10.3 0 32];
    featureStat.Platelets = [180 30 560];
    featureStat.Age = [50 1 100];   
    
end
