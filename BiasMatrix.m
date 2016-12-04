function [ bias ] = BiasMatrix(currentNeurons,NextNeurons,batchSize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

bias = rand(NextNeurons,batchSize);
bias = double(bias / 10000.0);

end

