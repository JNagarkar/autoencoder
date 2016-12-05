function [ output_args ] = displayOutput( arrayWeights,arrayBias,numTotalLayers)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

outputDirectory = 'C:\Users\jayna\Desktop\FSLA\TrainImages\output\';

testImages = dir('C:\Users\jayna\Desktop\FSLA\TestImages\TestImages\*.pgm');
numberTestFiles = length(testImages);


counterTest = 0;
currentLayer  = 1;
initialtest = [];

MSE = 0;

for ii=1:numberTestFiles
       currentfilenameTest = [testImages(ii).folder '/' testImages(ii).name];
       currentimageTest = imread(currentfilenameTest);
       
       currentImageName = testImages(ii).name;
       
       BTest = reshape(currentimageTest,1024,1);
       BTest = double(BTest) ./ double(255);
       initialtest = [initialtest BTest];
       imagesTest{ii} = BTest;
       counterTest  =counterTest +1;
       inputTest = imagesTest{ii};
       for Layer=1:numTotalLayers
           weightTest = arrayWeights{Layer};
           biasTest = arrayBias{Layer}(:,1);


           ZMatrixTest = createLayer(inputTest,weightTest,biasTest);
           ActivatedMatrixTest = ActivationFunction(ZMatrixTest);           
           inputTest = ActivatedMatrixTest;           
       end
       outputTest = inputTest;
       
       imwrite(reshape(outputTest,[32,32]),strcat(outputDirectory,currentImageName),'pgm');
       
    %RMSE = sqrt(sum((output-double(images{1})).^2));
    %disp(RMSE);

    %outputTest - double(inputTest)
    MSE = MSE + immse(outputTest, double(imagesTest{ii}));                            
    
end


MSE = MSE ./ double(numberTestFiles);
RMSE = sqrt(MSE);
disp(RMSE);
disp(datestr(now));
end

