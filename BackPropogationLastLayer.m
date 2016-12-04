function [ deltaWMain,deltaBMain,deltaValuesList ] = BackPropogationLastLayer(currentTrainingSample, batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,inputMatrix,deltaValuesList,deltaWMain,deltaBMain)
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
deltaB = zeros(numberNeurons(currentLayer + 1),batchSize);

    %{
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
    %}
    
    %break
    
activatedMatrixNextLayer = double(ActivatedArray{currentLayer + 1});
%targetMatrix = double(inputMatrix(:, currentTrainingSample));
deltaNextLayer = -(double(inputMatrix) - activatedMatrixNextLayer) .* (activatedMatrixNextLayer .* (1 - activatedMatrixNextLayer));
deltaValuesList{currentLayer + 1} = deltaNextLayer;

activatedMatrixCurLayer = double(ActivatedArray{currentLayer});
partialDerivativeW = deltaNextLayer * transpose(activatedMatrixCurLayer);
deltaW = deltaW + partialDerivativeW;
partialDerivativeB = deltaNextLayer;
deltaB = deltaB + partialDerivativeB;
deltaWMain = deltaW;
deltaBMain = deltaB;
%disp(deltaW);
%currentTrainingSample = currentTrainingSample + 1;

%disp(deltaW);

end

