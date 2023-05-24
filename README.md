# Image_Classification_Multiclass
The aim of this project is to build an image classifier that categorize images of airplanes,motorbikes and schooners into their respective classes. 

The dataset is obtained from kaggle https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners

The dataset consists of 3 folders representing
1. Motorcycles
2. Schooners
3. Airplanes

PROJECT DESIGN

The project is divided into 7 phases namely, 
Libraries and Variables
Data
Pre-processing
Neural Network Architecture - VGG16
Training and Saving the best model
Visualizing the obtained results in terms of Loss and Accuracy
Testing

Initial Step - Load the data into the runtime. For that, we first need to mount the drive where we are using the google colab. We use opendatasets module of python to directly load the data from the Kaggle by providing the kaggle dataset link and authorizing with our username and key.
The data is downloaded into the runtime, hence we can start working on different phases.

Libraries and Variables

First, we shall import some modules that we will be using for processing the images and building the neural network architecture. 
Here, we are using modules like os, numpy, tensorflow, matplotlib, keras [3]etc

Now, we define variables for storing the data, labels and classes along with the location of the images in the drive.
Data

The class_counter function counts the number of classes to just make sure only 3 classes are created in the furthur process. 
We then loop over the images in the folder to append them into data, labels, imagepaths.In the above code, we calculate the number of images in each class and notice that Schooner class has 63 images and Airplanes have 800 wheras Motorbikes have 798.

We see that 63 schooners are not enough to perform the training and obtain best results, hence we need to artificially expand the dataset by augmenting the existing data. Hence we should perform scaling and rotation to increase images to 800. We also augment motorbikes so that all the 3 classes have equal number of images.

Scaling - used to resize the images, so we can create multiple copies of the same       images with different size.

Rotation - We rotate the images to save multiple copies of the same image in different angles and hence we can expand the dataset.

The below code performs augmentation. Augmentation[6] is done on the existing images. For loop is applied on every image and we scale and rotate every image until the count is equal to the maximum images of a class. We augment motorbikes and schooner classes to make the images equal to the number of images in airplanes class.After the classes are augmented, we see that there are 800 images in each class.

Pre-Processing

First, we have to normalize the data i.e., change the range from [0,255] to [0,1] We then convert labels and imagepaths into numpy arrays and later, we convert class labels into encoding.

We then split the data into train and test data. The whole dataset is divided into 95% training and 5% testing. And then unpack the split variable into other variables like train and test images,labels and paths.

We save the names of the test images in a text file to perform testing on the NN later.

Neural Network Architecture

In this project, VGG16[2] neural network is used. Basically, VGG16 is a 16 layer deep neural network. VGG16[5] algorithm is very much efficient even upto classifying the images in 1000 classes. 
We freeze all the layers of VGG to prevent from training and we flatten the max-pooling layer which is the output of the vgg. We use softmax activation function to classify the images. 

We then have to add this output to the model.

Define hyperparameters like learning rate, number of epochs, and batch size.
After that, to set the loss method, we must define the dictionary and also for target training and testing output.

Training and Saving the best modelHere, we must save the best model from all the epochs and also we compile the model and obtain the model summary.

We then perform training on the model.The following is the result obtained - 
https://static.wixstatic.com/media/5a03fc_6cb3be1955fd4cda96181fe0612608f3~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/9f92b633-ed72-4a3e-b0bd-ea920c55f45f)

We see that the best accuracy is obtained from 8th epoch and it is 93.33 percent.

Visualizing the obtained results in terms of Loss and Accuracy

We plot the graph between training loss and validation loss and obtain the following https://static.wixstatic.com/media/5a03fc_52dfdef03ab8412db0cec9bb76c5c941~mv2.jpg/v1/fill/w_1480,h_902,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_52dfdef03ab8412db0cec9bb76c5c941~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/7e9f65e5-52e3-487c-b654-6cd19328b0c5)

The graph between Accuracy and validation accuracy is as follows https://static.wixstatic.com/media/5a03fc_0ff26b682e6e409e9f78be24c855a4ee~mv2.jpg/v1/fill/w_1480,h_902,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_0ff26b682e6e409e9f78be24c855a4ee~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/a814c10e-74c6-4d38-8c1f-5c4aa3c27afa)

Testing

We load the best model obtained for testing and then predict the labels of test images.We predict the class label of the 10 testing images and observe that all the values are predicted correctly and hence the model performs pretty decent.

Output
https://static.wixstatic.com/media/5a03fc_6a8720c1dddb4b9b8162b1cf6186ded2~mv2.jpg/v1/fill/w_1480,h_790,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_6a8720c1dddb4b9b8162b1cf6186ded2~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/85a1bc22-6b58-4600-88bb-542ee74303f3)

https://static.wixstatic.com/media/5a03fc_a364bfa2896c496cbae14d2e43864061~mv2.jpg/v1/fill/w_1480,h_702,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_a364bfa2896c496cbae14d2e43864061~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/18b83f99-e8ca-4d3d-b7fb-64e2a558489a)

CONTRIBUTION

The above model performs pretty well but in the process of increasing the accuracy, I tried experimenting with different hyperparameters and different number of layers.

Experiment 1 : Increasing the number of epochs - 

Previously, the number of epochs were 9 I tried increasing them to 40 and hence got better accuracy.
https://static.wixstatic.com/media/5a03fc_196923baff024c0380e5b4fd740df9a7~mv2.jpg/v1/fill/w_1480,h_702,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_196923baff024c0380e5b4fd740df9a7~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/69fb0f15-5e7b-43e4-b604-80c07b20f3cd)

We see that previously, the accuracy was 93.33 percent, but after increasing the epochs, the accuracy got increased to 98.61 and hence the model is performing even better.

We also analysed the loss and accuracy by plotting the performance.
https://static.wixstatic.com/media/5a03fc_ef58e500c5964afbb51f8cb3e9a9bb38~mv2.jpg/v1/fill/w_1480,h_894,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_ef58e500c5964afbb51f8cb3e9a9bb38~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/5d76470f-63bd-406e-867a-049ac734ef7a)

https://static.wixstatic.com/media/5a03fc_ae06db9ad65b456b858fafc3b63a852d~mv2.jpg/v1/fill/w_1480,h_894,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5a03fc_ae06db9ad65b456b858fafc3b63a852d~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/f4bd51a0-964e-4874-a94f-820003058180)

Experiment 2  : Number of Layers

Previously, there were only 2 layers and I tried increasing it to 3 and hence the accuracy is also increased from 93.33 to 97.22https://static.wixstatic.com/media/5a03fc_de6879d6758a475c9ac84c2e7d63a6af~mv2.jpg![image](https://github.com/shreya-malraju/Image_Classification_Multiclass/assets/132793649/86222931-9006-4642-ae25-658a3de25291)

From the above two experiments, the highest accuracy is obtained from increasing the number of epochs and hence the performance of the model is increased.

REFERENCES


[1] https://www.kaggle.com/code/maricinnamon/multiclass-classification-caltech101-tensorflow

[2]https://www.mathworks.com/help/deeplearning/ref/vgg16.html;jsessionid=c9b7423621bb9e3cf9c8d63d2cec 

[3]https://keras.io/guides/sequential_model/ 

[4]https://www.tensorflow.org/learn#build-models 

[5]https://www.mathworks.com/help/deeplearning/ref/vgg16.html 

[6]https://towardsdatascience.com/non-linear-augmentations-for-deep-leaning-4ba99baaaaca 

