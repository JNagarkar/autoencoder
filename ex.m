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



for Layer=1:numTotalLayers
    deltaWMain = zeros(numberNeurons(Layer+1), numberNeurons(Layer));
    deltaBMain = zeros(numberNeurons(Layer+1), batchSize);
    arraydeltaW{Layer} = deltaWMain;
    arraydeltaB{Layer} = deltaBMain;
end
    arraydeltaW = {};
    arraydeltaB = {};