
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



for epoch=1:100
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
           numTotalLayers = numHiddenLayer + 1;
           ActivatedArray{currentLayer} = initial;
           for Layer=1:numTotalLayers
               images{ii} = currentimage;
               currentLayerNeurons = numberNeurons(Layer);
               nextLayerNeurons = numberNeurons(Layer + 1);
               weight = NormalizedWeight(currentLayerNeurons,nextLayerNeurons);
               bias   = BiasMatrix(currentLayerNeurons,nextLayerNeurons);  
               arrayWeights{currentLayer} =  weight;
               arrayBias{currentLayer} = bias;
               ZMatrix = createLayer(input,weight,bias);
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
           [arrayWeights,arrayBias] = BackPropogation(arrayWeights,arrayBias,ZArray,ActivatedArray,currentLayer,numberNeurons,initial);       
            initial=[];
            currentLayer = 1;
       end
    end
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


imshow(reshape(output,32,32));
