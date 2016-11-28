function [ output_args ] = BackPropogation( arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix)
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here




currentTrainingSample = 0;

%Calculate delta for output layer
currentTrainingSample = 1;
BackPropogationLastLayer(currentTrainingSample, arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix);



end

