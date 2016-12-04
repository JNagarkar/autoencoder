function [ bias ] = BiasMatrix(currentNeurons,NextNeurons)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

bias = rand(NextNeurons,10);
bias = double(bias / 1000.0);

end

