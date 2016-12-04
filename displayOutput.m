function [ output_args ] = displayOutput( arrayWeights,arrayBias,numTotalLayers)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

outputDirectory = 'C:\Users\jayna\Desktop\FSLA\TrainImages\output\';

testImages = dir('C:\Users\jayna\Desktop\FSLA\TestImages\TestImages\*.pgm');
numberTestFiles = length(testImages);


counterTest = 0;
currentLayer  = 1;
initialtest = [];
for ii=1:numberTestFiles
       currentfilenameTest = [testImages(ii).folder '/' testImages(ii).name];
       currentimageTest = imread(currentfilenameTest);
       
       currentImageName = testImages(ii).name;
       
       B = reshape(currentimageTest,1024,1);
       B = double(B) ./ double(255);
       initialtest = [initialtest B];
       imagesTest{ii} = B;
       counterTest  =counterTest +1;
       input = imagesTest{ii};
       for Layer=1:numTotalLayers
           weight = arrayWeights{Layer};
           bias = arrayBias{Layer}(:,1);


           ZMatrix = createLayer(input,weight,bias);
           ActivatedMatrix = ActivationFunction(ZMatrix);           
           input = ActivatedMatrix;           
       end
       output = input;
       
       imwrite(mat2gray(reshape(output,[32,32])),strcat(outputDirectory,currentImageName),'pgm');
       
    %RMSE = sqrt(sum((output-double(images{1})).^2));
    %disp(RMSE);
end


end

disp(datestr(now));