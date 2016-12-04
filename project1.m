
%imagefiles = dir('E:\\ASU\\Fall 2016\\FSL\\Project\\TrainImages\\*.pgm');      
imagefiles = dir('C:\\Users\\jayna\\Desktop\\FSLA\\TrainImages\\train\\*.pgm');
nfiles = length(imagefiles);    % Number of files found

counter = 0;
%create 1024*10 matrix
initial = [];

currentLayerNeurons = 1024;
nextLayerNeurons = 512;

currentLayer=0;
numHiddenLayer = 1;

numberNeurons =[1024,512,1024];


%Matrix arrays
arrayWeights={};

%bias matrix
arrayBias={};

layersArrayCounter = 2;
ZArray = {};
ActivatedArray = {};

% 1000 examples , 512 * 1024
%weight=[]
%weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);



currentLayer = currentLayer+1;
%arrayWeights{currentLayer} =weight;

batchSize = 10.00;
numTotalLayers = numHiddenLayer + 1;
%deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));
%deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
   
for Layer=1:numTotalLayers
    currentLayerNeurons = numberNeurons(Layer);
    nextLayerNeurons = numberNeurons(Layer + 1);
    weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);
    bias   = BiasMatrix(currentLayerNeurons,nextLayerNeurons);  
    arrayWeights{currentLayer} =  weight;
    arrayBias{currentLayer} = bias;
    currentLayer = currentLayer + 1;
end

currentLayer = 1;


for epoch=1:1000

    
    % Need new delta weight, delta bias values for each epoch. 
    arraydeltaW = {};
    arraydeltaB = {};
    for Layer=1:numTotalLayers
        deltaWMain = zeros(numberNeurons(Layer+1), numberNeurons(Layer));
        deltaBMain = zeros(numberNeurons(Layer+1), batchSize);
        arraydeltaW{Layer} = deltaWMain;
        arraydeltaB{Layer} = deltaBMain;
    end
    
    
    % iterate over total number of images
    for ii=1:nfiles
       currentfilename = [imagefiles(ii).folder '/' imagefiles(ii).name];
       currentimage = imread(currentfilename);
       B = reshape(currentimage,1024,1);
       B = double(B) / double(255);
       initial = [initial B];
       images{ii} = B;
       counter  =counter +1;

       if mod(counter,10) == 0
           %Feed Forward
           input = initial;
           % numTotalLayer does not include the first layer input.
           ActivatedArray{currentLayer} = initial;
           for Layer=1:numTotalLayers
               images{ii} = currentimage;
               currentLayerNeurons = numberNeurons(Layer);
               nextLayerNeurons = numberNeurons(Layer + 1);
               %weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);
               %bias   = BiasMatrix(currentLayerNeurons,nextLayerNeurons);  
               %arrayWeights{currentLayer} =  weight;
               %arrayBias{currentLayer} = bias;
               ZMatrix = createLayer(input,arrayWeights{currentLayer},arrayBias{currentLayer});
               ZArray{currentLayer+1}=ZMatrix;
               ActivatedMatrix = ActivationFunction(ZMatrix);           
               ActivatedArray{currentLayer+1}= ActivatedMatrix;
               input = ActivatedMatrix;
               currentLayer = currentLayer+1;
           end

           currentLayer = currentLayer -1;

           %disp('CL')
           %disp(currentLayer)

           % Back propogation
           %[deltaWMain,deltaBMain,arrayWeights,arrayBias] = BackPropogation(arrayWeights,arrayBias,ZArray,ActivatedArray,currentLayer,numberNeurons,initial,deltaWMain,deltaBMain);       
           
           deltaValuesList = {};
           %deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));
           %deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
           deltaWMain = arraydeltaW{currentLayer};
           deltaBMain = arraydeltaB{currentLayer};
           [deltaWMain, deltaBMain, deltaValuesList] = BackPropogationLastLayer(batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,deltaValuesList,deltaWMain,deltaBMain); 
           arraydeltaW{currentLayer} = arraydeltaW{currentLayer} + deltaWMain;
           arraydeltaB{currentLayer} = arraydeltaB{currentLayer} + deltaBMain;
           currentLayer = currentLayer - 1;
           
           while currentLayer > 0
                %deltaWMain = zeros(numberNeurons(currentLayer+1), numberNeurons(currentLayer));
                %deltaBMain = zeros(numberNeurons(currentLayer+1), batchSize);
                deltaWMain = arraydeltaW{currentLayer};
                deltaBMain = arraydeltaB{currentLayer};
                [deltaWMain, deltaBMain, deltaValuesList] = BackPropagationHiddenLayers(batchSize,arrayWeights,ZArray,ActivatedArray,currentLayer,numberNeurons,deltaValuesList,deltaWMain,deltaBMain);
                arraydeltaW{currentLayer} = arraydeltaW{currentLayer} + deltaWMain;
                arraydeltaB{currentLayer} = arraydeltaB{currentLayer} + deltaBMain;
                currentLayer = currentLayer - 1;
           end
           
           initial=[];
           currentLayer = 1;
       end
    end
    
    alphaValue = 0.0001;
    
    for Layer=1:numTotalLayers
        arrayWeights{Layer} = arrayWeights{Layer} - alphaValue*(arraydeltaW{Layer})/counter;
        arrayBias{Layer} = arrayBias{Layer} - alphaValue*(arraydeltaB{Layer})/counter;
    end
    disp(strcat('iteration',int2str(epoch)));
end

currentLayer = 1;
input = images{1};
numTotalLayers = numHiddenLayer + 1;
for Layer=1:numTotalLayers
       currentLayerNeurons = numberNeurons(Layer);
       nextLayerNeurons = numberNeurons(Layer + 1);
       weight = arrayWeights{currentLayer};
       bias = arrayBias{currentLayer}(:,1);
       ZMatrix = createLayer(input,weight,bias);
       ZArray{currentLayer+1}=ZMatrix;
       ActivatedMatrix = ActivationFunction(ZMatrix);           
       ActivatedArray{currentLayer+1}= ActivatedMatrix;
       input = ActivatedMatrix;
       currentLayer = currentLayer+1;
end
output = input;
RMSE = sqrt(sum((output-double(images{1})).^2));
disp(RMSE);

output = output*255;
imshow(reshape(output,32,32));
