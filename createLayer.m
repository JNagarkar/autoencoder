function [zMatrix] = createLayer(input,weight,bias)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

%original Wt*x matrix + add bias here
zMatrix = weight*double(input) + bias;
zMatrix = double(zMatrix);
%size(zMatrix)
%size(weight)
%size(input)
%size(zMatrix)
end

