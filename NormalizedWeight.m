function [weight] = NormalizedWeight(currentNeurons,NextNeurons)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

%weight = zeros(currentNeurons,NextNeurons);

weight = rand(NextNeurons,currentNeurons);
weight = double(weight / 100.0);
%disp(weight);

end

