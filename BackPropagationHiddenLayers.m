function [ deltaWMain,deltaBMain,deltaValuesList ] = BackPropagationHiddenLayers(currentTrainingSample, batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix,deltaValuesList,deltaWMain,deltaBMain)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
deltaW = zeros(numberNeurons(currentLayer + 1),numberNeurons(currentLayer));
deltaB = zeros(numberNeurons(currentLayer + 1),batchSize);
  
weightMatrixNextLayer = double(arrayWeights{currentLayer + 1});
deltaNextNextLayer = deltaValuesList{currentLayer + 2};
activatedMatrixNextLayer = double(ActivatedArray{currentLayer + 1});
deltaNextLayer = (transpose(weightMatrixNextLayer) * deltaNextNextLayer) .* (activatedMatrixNextLayer .* (1 - activatedMatrixNextLayer));
deltaValuesList{currentLayer + 1} = deltaNextLayer;


activatedMatrixCurLayer = double(ActivatedArray{currentLayer});
partialDerivativeW = deltaNextLayer * transpose(double(activatedMatrixCurLayer));
deltaW = deltaW + partialDerivativeW;
deltaWMain = deltaW;
partialDerivativeB = deltaNextLayer;
deltaB = deltaB + partialDerivativeB;
deltaBMain = deltaB;


end

