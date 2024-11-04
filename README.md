#########
Image Recoloration using diffusion imaging
##########

The Network learns to do a reconstruction step and can therefore recreate the base image to some accuracy from a completly noised input.
This is one of the basic principles in image creation using AI. In this case the used dataset is expansive (CIFAR or smth if i am correct) so the size of the train/test data is not applicable to your problem. 

When Tim Alder and I talked about it, I was mentioning the possibility of 3D cell reconstruction or better said predicting a higher resolution image of known 3D structures.

For this process the idea at the time was to take manually traced astrocytes (through painstaking effort a high resolution can be obtained) and from one cell select a vartiety of sample branches.
Then the the data of these branches is noised and the objective of the network is reconstruct the branch through the noise. The idea beeing that the algorithm learns the most high resolution structures based on the underlying data and is able to reconstruct them from the noise.

Now for this to work only a few cells are needed but due to the complex morphology the idea is taht since only ever one branch is fed into the network enough traning/test data can be extracted. 

Now as I do not know the specifics of your problem it is hard to see if this idea is genereally feasible also in your case. (segmenting small pathches from your available data to train a network) 
But at least this is a workaround in the case of expensive to obtain datasets.

To see how diffusion imaging works in general i alse added the lecture notes. Please consider whatever copyright claims may apply to those documents.

Best regards, 
Alexander
