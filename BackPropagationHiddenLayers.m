function [ deltaWMain,deltaValuesList ] = BackPropagationHiddenLayers(currentTrainingSample, arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix,deltaValuesList,deltaWMain)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
deltaW = zeros(numberNeurons(currentLayer + 1),numberNeurons(currentLayer));
  
weightMatrixNextLayer = double(arrayWeights{currentLayer + 1});
deltaNextNextLayer = deltaValuesList{currentLayer + 2};
activatedMatrixNextLayer = double(ActivatedArray{currentLayer + 1});
deltaNextLayer = (transpose(weightMatrixNextLayer) * deltaNextNextLayer) .* (activatedMatrixNextLayer .* (1 - activatedMatrixNextLayer));
deltaValuesList{currentLayer + 1} = deltaNextLayer;
activatedMatrixCurLayer = double(ActivatedArray{currentLayer}(:, currentTrainingSample));
partialDerivative = deltaNextLayer * transpose(double(inputMatrix));
deltaW = deltaW + partialDerivative;
deltaWMain = deltaW;


end

