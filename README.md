## Introduction: 
Handwriting is the most common way of transmitting and organizing information. Mechanical and digital technology have gradually begun to replace conventional handwriting methods for the sake of efficiency and time. Despite the obvious advantages of these modern technologies, the importance of handwritten communication cannot be underestimated. It is not fair for the world to retreat to pen and paper in the face of the current data deluge and perpetual sense of urgency that exists in practically every major industry. Handwriting recognition (HWR) technology on tablets and other devices, on the other hand, allows you to keep the cognitive benefits of handwriting without going back to the days of overflowing filing cabinets. In this project we strive to solve this problem by extracting text from handwritten image files using deep convoluted neural networks and computer vision.


## Project Dataset: 
The model was trained using the Kaggle Alphabet Dataset: A_Z Handwritten Data.csv Dataset. There are 3,72,450 photos in the collection, with 26 labels for all the English alphabets. It was also utilized to create a program that recognizes handwritten text on paper. The Dataset contains 785 columns and is of 699 MB. Each letter in the image is center fitted to a 20 X 20-pixel box, and the dataset comprises of 26 labels (A-Z) containing handwritten graphics of size 28 X 28 pixels. Each image is stored as Gray level. The images are taken from NIST(https://www.nist.gov/srd/nist-special-database-19) and NMIST large dataset and few other sources which were then formatted as mentioned above.


## Project Scope: 
Keeping the timeline in mind the current scope of the project is to recognize the alphabets in English language having 26 characters. The input images will be grayscale with limited size, resolution and orientation. Lower case alphabets are written in a number of different ways by different writers, whereas capital letters are written in a similar manner by all writers that’s why for this project, we will consider only uppercase letters. Individual handwriting detection is easier to detect than cursive which is connected and poses more issues with evaluation. It is difficult to interpret handwriting with no distinct separation between characters. Since we are trying to design a semi-trained model of a Neural Network, the computation time and power will also be very high. 

## Dataflow Diagram

<img width="293" alt="Screenshot 2023-01-08 at 4 52 00 PM" src="https://user-images.githubusercontent.com/11815663/211223056-6538c9ce-8407-41f9-8161-aaf26b26a350.png">

Figure 1: Data Flow for Handwriting Recognition System (Methodology)

## Methodology
Initially we load the dataset into a NumPy array.
As we have reported earlier the dimensions of the dataset are (372450). A list of the English alphabets is created.
We extract the numeric labels of the images through the first column and map them to their corresponding alphabets using the alphabet list created above.
The data of each image is represented in a single row in the dataset where each feature represents an
individual pixel intensity of the 784 image pixels (28 X 28).

A. Pre – processing
We reshape the dataset to a 3D format. We arrange the pixels in 28 X 28 format and label each image according to the label class. Basically, we are constructing 372,450,785 2D matrices, each representing an image from the dataset.
We count the occurrence of each alphabet image in the dataset and plot the count against the alphabet. It is observed that the most occurring alphabet in the dataset is ‘o’ followed by ‘s’. The least count is observed in letters ‘f’ and ‘I’.
To get a better understanding of the handwritten images with which we will train our model, we plot them. Having reshaped the original dataset from 372450 X 784 to 372450 X 28 X28, we plot 400 out of the 372450 to get a visual intuition of the data.

B. Splitting the Dataset
Next, we split the data into training and testing data. As we will be dealing with convoluted neural networks, we need a lot of training data for better model performance. For this reason, we have chosen a 90% training and 10% testing split. For the ease of convenience, we add another dimension to the data to make it linear and easier to split.

C. Building the Convolutional Neural Network Model
After pre-processing and splitting our dataset, we begin to implement the neural network. We are building out CNN models using Sequential API. Sequential API allows us to create a model’s layer by layer in a step-by-step fashion. A 4 convolutional layer with 1 max pooling layer after every 2 convolutional layers. As our input layer should be two dimensional, a flattening layer needs to be added between them. At the end of the fully connected layers is a softmax layer.
 
Figure 2: Stages of a Deep Convoluted Neural Network
Figure 2: Kernel Function is convoluted with pixel values of images in the convolution layer.
Figure 3: Kernel Function is translated throughout the image. Pixel values of the image are transformed according the kernel function used

Pooling:
Pooling is generally performed to reduce the dimensionality. Therefore, it gives us the bandwidth to reduce the number of parameters, that reduces the training time and chances of overfitting. The most common type of pooling is max pooling that takes the maximum value in the pooling space. Pooling slides a window over its input and simply takes maximum value in the window. Thus, the largest element from the feature map is selected.

Max-pooling:
Max pooling is a way to reduce the dimensionality of an image. It takes the maximum pixel value of a grid. This technique makes the model more generic and helps reduce overfitting.

Figure 4: Max pooling reduces the number of features by replacing a set of features with a value which represents the group of features closely. Eg. maximum value in a group of pixels.

D. Training The Network
A low probability results in minimal effect and a high probability results in under learning by the network. When dropout is used on a larger network, we are likely to get better performance as the model learns more independent representations. Using dropout on visible as well as hidden units that shows good results.

E. Dense Layer and Flatten Layer
In a neural network, a dense layer is simply a regular layer of neurons. Each neuron in the previous layer receives information from all the neurons in the layer above it, making it tightly linked. The reason we do this is that we're going to need to insert this data into an artificial neural network later on.

## Results:
Training on the full dataset (over 3 million examples) can take many hours or days even with the fastest GPU.
Constrained by the computation resources we have access to, we ran our major experiments (for model comparison and visualization) only with one epoch.
We can increase the number of epochs to train our model more efficiently which perhaps will increase our accuracy even further.
With one epoch, the accuracy we get is, around 95% and the validation accuracy we get is, around 98%.
