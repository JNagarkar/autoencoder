
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