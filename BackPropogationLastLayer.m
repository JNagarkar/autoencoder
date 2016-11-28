function [ output_args ] = BackPropogationLastLayer(currentTrainingSample, arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix)
%UNTITLED18 Summary of this function goes here
%   Detailed explanation goes here

%Calculate partial derivative of each J(W,b;x,y) for each Wij

%{
disp(ActivatedArray{currentLayer});
disp(numberNeurons(currentLayer+1))
disp(ActivatedArray{currentTrainingSample}(1))
disp(inputMatrix{currentTrainingSample}(1))
%}
%Summation of error weight for every pixel over a batch of 10
deltaW = zeros(numberNeurons(currentLayer + 1),numberNeurons(currentLayer));

%while currentTrainingSample <= size(inputMatrix,2)
    
    for nextLayerNeuron=1:numberNeurons(currentLayer+1)
        for currentLayerNeuron=1:numberNeurons(currentLayer)
            activatedMatrixCurLayer = ActivatedArray{currentLayer};
            %disp(currentLayerNeuron)
            ajCurLayer = double(activatedMatrixCurLayer(currentLayerNeuron,currentTrainingSample));
            %disp('ajCurLayer')
            %disp(ajCurLayer)
            targetPixel = double(inputMatrix(nextLayerNeuron,currentTrainingSample));
            
            activatedMatrixNextLayer = double(ActivatedArray{currentLayer + 1});
            %Calculate f'(Zi)
            aiNextLayer = double(activatedMatrixNextLayer(nextLayerNeuron,currentTrainingSample));
            %disp('aiNextLayer')
            %disp(aiNextLayer)
            derivativeZi = aiNextLayer * (1 - aiNextLayer);

            %Calculate partial derivative w.r.t. Wij
            partDrv = ajCurLayer *(aiNextLayer - targetPixel) * derivativeZi
            % disp('partDrv')
           % disp(partDrv)
            deltaW(nextLayerNeuron,currentLayerNeuron) = deltaW(nextLayerNeuron,currentLayerNeuron) + partDrv;
           % break
        end
        %break
    end
    %break
    %currentTrainingSample = currentTrainingSample + 1;
    disp(deltaW)
end


%end

