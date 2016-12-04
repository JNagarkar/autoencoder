%{
arrayWeights = {};
arrayBias = {};
arrayWeights{1} = W1;
arrayWeights{2} = W2;
arrayBias{1} = b1;
arrayBias{2} = b2;
initial1 = [];
currentimage = imread('C:\\Users\\jayna\\Desktop\\FSLA\\TestImages\\TestImages\\Adolfo_Rodriguez_Saa_0002.pgm');
B = reshape(currentimage,1024,1);
B = double(B) ./ double(255);
initial1 = [initial1 B];
images{1} = B;
input = initial1;
for Layer=1:2
    ZMatrix = createLayer(input,arrayWeights{Layer},arrayBias{Layer});
    ZArray{Layer}=ZMatrix;
    ActivatedMatrix = ActivationFunction(ZMatrix);           
ActivatedArray{Layer}= ActivatedMatrix;
    input = ActivatedMatrix;

end
    imshow(reshape(ActivatedMatrix,32,32));
%}

outputDirectory = 'C:\Users\jayna\Desktop\FSLA\TrainImages\output';

testImages = dir('C:\Users\jayna\Desktop\FSLA\TestImages\TestImages\*.pgm');
numberTestFiles = length(testImages);




outputDirectory = 'C:\Users\jayna\Desktop\FSLA\TrainImages\output\';
currentimageTest = imread('C:\Users\jayna\Desktop\FSLA\TrainImages\train\Adrien_Brody_0003.pgm');
imwrite(reshape(currentimageTest,[32,32]),strcat(outputDirectory,'abc.pgm'),'pgm');