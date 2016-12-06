#Autoencoder with configurable hidden layers.

#Goal:

Above auto-encoder consumes in images of size 32 * 32 and produces satisfactory outputs (similar to input images) on test dataset.


# Instructions for running the autoencoder:

1. Matlab needs to be installed on the system
2. The 'task1.m', 'task2.m', 'task3.m' files are to be run for the Autoencoder to execute. 
3. The path for the training images is to be stated in 'trainImages' variable in 'task*.m' files
4. The value of learning rate (variable 'alphaValue') and decay parameter ('lambda') can be set.   


## Implementation details and instructions for running the tests.
The number of hidden layers to be used is to be mentioned in the 'numHiddenLayer' variable.
According to the value of 'numHiddenLayer', the variable 'numberNeurons' is to be set.
numberNeurons = [Number of neurons in input layer, Number of neurons in hidden layer(s), Number of neurons in output layer]
eg. numHiddenLayer=1 ; numberNeurons=[1024,512,1024]
	numHiddenLayer=2 ; numberNeurons=[1024,512,512,1024]

## Assumptions about the dataset:

1. OutputDirectory in displayoutput is set by default to in the same directory.  However, if admin permissions are not present, you may be required to explicitly set the output directory.
2. For task 1:TrainImages input path: Has to be set.
3. For task 2:TrainImages input path: Has to be set.
4. For task 3:TrainImages input path: Has to be set.
5. In displayOutput.m Function:  input path for test images has to be set. output path: for reconstructed images has to be set.

## Reference material

We have used the following reference material.

1. [University of Florida Tutorial on Autoencoder](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
2. [Stanford University Course CS 294 by Andrew NG](http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)























