function [ neuralNet] = sigmoid(neuralNet)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

neuralNet = 1.0 ./ ( 1.0 + exp(-neuralNet) );

end

