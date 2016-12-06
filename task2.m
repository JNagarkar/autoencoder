disp(datestr(now));
      
trainImages = dir('C:\\Users\\jayna\\Desktop\\FSLA\\TrainImages\\single\\*.pgm');
numberTrainFiles = length(trainImages);    % Number of files found
counter = 0;
numberOfEpochs = 5;
%create 1024*10 matrix
initial1 = [];
%Allow user to start create a neural network with custom size.
currentLayerNeurons = 1024;
numberNeurons =[1024,512,512,1024];

batchSize = 10;

if batchSize > numberTrainFiles
    batchSize = numberTrainFiles;
end
       
numHiddenLayer = length(numberNeurons)-2;
%Matrix arrays
arrayWeights={};

%bias matrix
arrayBias={};
ZArray = {};
ActivatedArray = {};

% 1000 examples , 512 * 1024
%weight=[]
%weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);

alphaValue = 0.01;
lambda = 0.01;

currentLayer = 1;
numTotalLayers = numHiddenLayer + 1;
% create a structure of the neural network   
for Layer=1:numTotalLayers
    currentLayerNeurons = numberNeurons(Layer);
    nextLayerNeurons = numberNeurons(Layer + 1);
    weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);
    bias   = BiasMatrix(currentLayerNeurons,nextLayerNeurons,batchSize);  
    arrayWeights{currentLayer} =  weight;
    arrayBias{currentLayer} = bias;
    currentLayer = currentLayer + 1;
end

currentLayer = 1;
%iterate over number of epochs
for epoch=1:numberOfEpochs    
    % Need new delta weight, delta bias values for each epoch. 
    arraydeltaW = {};
    arraydeltaB = {};
    counter = 0;
    
    for Layer=1:numTotalLayers
        deltaWTempBatch = zeros(numberNeurons(Layer+1), numberNeurons(Layer));
        deltaBTempBatch = zeros(numberNeurons(Layer+1), batchSize);
        arraydeltaW{Layer} = deltaWTempBatch;
        arraydeltaB{Layer} = deltaBTempBatch;
    end
        
    % iterate over total number of images
    for ii=1:numberTrainFiles
       currentfilename = [trainImages(ii).folder '/' trainImages(ii).name];
       currentimage = imread(currentfilename);
       
       B = reshape(currentimage,1024,1);
       B = double(B) ./ double(255);
       initial1 = [initial1 B];
       images{ii} = B;
       counter  =counter +1;

       if mod(counter,batchSize) == 0
           %Feed Forward pass
           input = initial1;
           % numTotalLayer does not include the first layer input.
           ActivatedArray{currentLayer} = initial1;
           for Layer=1:numTotalLayers
               currentLayerNeurons = numberNeurons(Layer);
               nextLayerNeurons = numberNeurons(Layer + 1);
               ZMatrix = createLayer(input,arrayWeights{currentLayer},arrayBias{currentLayer});
               %ZArrray is input array for a layer
               ZArray{currentLayer+1}=ZMatrix;
               ActivatedMatrix = ActivationFunction(ZMatrix);           
               ActivatedArray{currentLayer+1}= ActivatedMatrix;
               input = ActivatedMatrix;
               currentLayer = currentLayer+1;
           end
           currentLayer = currentLayer -1;
           %feed forward pass ends
           
           % Back propogation starts
           deltaValuesList = {};
           % Back propogation for last layer
           [deltaWTempBatch, deltaBTempBatch, deltaValuesList] = BackPropogationLastLayer(batchSize,ActivatedArray,currentLayer,numberNeurons,deltaValuesList,deltaWTempBatch,deltaBTempBatch); 
           arraydeltaW{currentLayer} = arraydeltaW{currentLayer} + deltaWTempBatch;
           arraydeltaB{currentLayer} = arraydeltaB{currentLayer} + deltaBTempBatch;
           currentLayer = currentLayer - 1;
                      
           while currentLayer > 0
                % Back propogation for last but one layer
                [deltaWTempBatch, deltaBTempBatch, deltaValuesList] = BackPropagationHiddenLayers(batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,deltaValuesList,deltaWTempBatch,deltaBTempBatch);
                arraydeltaW{currentLayer} = arraydeltaW{currentLayer} + deltaWTempBatch;
                arraydeltaB{currentLayer} = arraydeltaB{currentLayer} + deltaBTempBatch;
                currentLayer = currentLayer - 1;
           end
           
           for Layer=1:numTotalLayers
                arrayWeights{Layer} = arrayWeights{Layer} - alphaValue*((arraydeltaW{Layer})./batchSize + (lambda * arrayWeights{Layer}));
                arrayBias{Layer} = arrayBias{Layer} - alphaValue*(arraydeltaB{Layer})./batchSize;
           end
            % end of backpropogation. 
           initial1=[];
           currentLayer = 1;
       end
    end
    disp(strcat('iteration',int2str(epoch)));
end
displayOutput(arrayWeights,arrayBias,numTotalLayers);
