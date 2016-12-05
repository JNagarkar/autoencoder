disp(datestr(now));
%imagefiles = dir('E:\\ASU\\Fall 2016\\FSL\\Project\\TrainImages\\*.pgm');      
trainImages = dir('C:\\Users\\jayna\\Desktop\\FSLA\\TrainImages\\TrainImages\\*.pgm');
numberTrainFiles = length(trainImages);    % Number of files found
numberTrainFiles = 100;
counter = 0;
%create 1024*10 matrix
initial1 = [];

currentLayerNeurons = 1024;
nextLayerNeurons = 512;

numHiddenLayer = 1;
numberNeurons =[1024,512,1024];
batchSize = 10;


%Matrix arrays
arrayWeights={};

%bias matrix
arrayBias={};


ZArray = {};
ActivatedArray = {};

% 1000 examples , 512 * 1024
%weight=[]
%weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);

alphaValue = 0.001;
lambda = 0.001;


currentLayer = 1;
%arrayWeights{currentLayer} =weight;

numTotalLayers = numHiddenLayer + 1;
%deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));
%deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
   
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

for epoch=1:1000

    
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
           %Feed Forward
           input = initial1;
           % numTotalLayer does not include the first layer input.
           ActivatedArray{currentLayer} = initial1;
           for Layer=1:numTotalLayers
               currentLayerNeurons = numberNeurons(Layer);
               nextLayerNeurons = numberNeurons(Layer + 1);
               %weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);
               %bias   = BiasMatrix(currentLayerNeurons,nextLayerNeurons);  
               %arrayWeights{currentLayer} =  weight;
               %arrayBias{currentLayer} = bias;
               
               %disp(strcat('CurrentLayer',int2str(currentLayer)));
               %disp(strcat('Layer',int2str(Layer)));
               
               
               
               %disp(strcat('Layer',int2str(Layer)));
               %disp(strcat('currentLayer',int2str(currentLayer)));
               ZMatrix = createLayer(input,arrayWeights{currentLayer},arrayBias{currentLayer});
               %ZArrray is input array for a layer
               ZArray{currentLayer+1}=ZMatrix;
               ActivatedMatrix = ActivationFunction(ZMatrix);           
               ActivatedArray{currentLayer+1}= ActivatedMatrix;
               input = ActivatedMatrix;
               currentLayer = currentLayer+1;
           end

           %disp(strcat('CurrentLayer after it',int2str(currentLayer)));
           currentLayer = currentLayer -1;

           %disp('CL')
           %disp(currentLayer)

           % Back propogation
           %[deltaWMain,deltaBMain,arrayWeights,arrayBias] = BackPropogation(arrayWeights,arrayBias,ZArray,ActivatedArray,currentLayer,numberNeurons,initial,deltaWMain,deltaBMain);       
           

           
           deltaValuesList = {};
           %deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));
           %deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
           [deltaWTempBatch, deltaBTempBatch, deltaValuesList] = BackPropogationLastLayer(batchSize,ActivatedArray,currentLayer,numberNeurons,deltaValuesList,deltaWTempBatch,deltaBTempBatch); 
           arraydeltaW{currentLayer} = arraydeltaW{currentLayer} + deltaWTempBatch;
           arraydeltaB{currentLayer} = arraydeltaB{currentLayer} + deltaBTempBatch;
           currentLayer = currentLayer - 1;
           
           %disp(strcat('CurrentLayer after last',int2str(currentLayer)));
               
           
           while currentLayer > 0
                %deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));

                %deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
                [deltaWTempBatch, deltaBTempBatch, deltaValuesList] = BackPropagationHiddenLayers(batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,deltaValuesList,deltaWTempBatch,deltaBTempBatch);
                arraydeltaW{currentLayer} = arraydeltaW{currentLayer} + deltaWTempBatch;
                arraydeltaB{currentLayer} = arraydeltaB{currentLayer} + deltaBTempBatch;
                currentLayer = currentLayer - 1;
           end
           
           for Layer=1:numTotalLayers
                %disp(strcat('counter',int2str(counter)));
                arrayWeights{Layer} = arrayWeights{Layer} - alphaValue*((arraydeltaW{Layer})./batchSize + (lambda * arrayWeights{Layer}));
                arrayBias{Layer} = arrayBias{Layer} - alphaValue*(arraydeltaB{Layer})./batchSize;
           end

           %disp(strcat('CurrentLayer after hidden',int2str(currentLayer)));

           initial1=[];
           currentLayer = 1;
       end
    end
    %{
    for Layer=1:numTotalLayers
        %disp(strcat('counter',int2str(counter)));
        arrayWeights{Layer} = arrayWeights{Layer} - alphaValue*(arraydeltaW{Layer})./counter;
        arrayBias{Layer} = arrayBias{Layer} - alphaValue*(arraydeltaB{Layer})./counter;
    end
    %}
    disp(strcat('iteration',int2str(epoch)));
end
displayOutput(arrayWeights,arrayBias,numTotalLayers);
