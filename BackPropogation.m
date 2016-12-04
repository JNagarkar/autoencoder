function [ deltaWMain,deltaBMain, arrayWeights,arrayBias ] = BackPropogation( arrayWeights,arrayBias,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix,deltaWMain,deltaBMain)
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here

alphaValue = 0.0001;
batchSize = 10.00;


currentTrainingSample = 0;
deltaValuesList = {};
%Calculate delta for output layer
currentTrainingSample = 1;

[deltaWMain, deltaBMain, deltaValuesList] = BackPropogationLastLayer(currentTrainingSample, batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix,deltaValuesList,deltaWMain,deltaBMain);
%disp(deltaWMain);
arrayWeights{currentLayer} = arrayWeights{currentLayer} - alphaValue*(deltaWMain)/batchSize;
arrayBias{currentLayer} = arrayBias{currentLayer} - alphaValue*(deltaBMain)/batchSize;
currentLayer = currentLayer - 1;

while currentLayer > 0
%    deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));
%    deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
    [deltaWMain, deltaBMain, deltaValuesList] = BackPropagationHiddenLayers(currentTrainingSample, batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix,deltaValuesList,deltaWMain,deltaBMain);
    arrayWeights{currentLayer} = arrayWeights{currentLayer} - alphaValue*(deltaWMain)/batchSize;
    arrayBias{currentLayer} = arrayBias{currentLayer} - alphaValue*(deltaBMain)/batchSize;
    currentLayer = currentLayer - 1;
end
end

