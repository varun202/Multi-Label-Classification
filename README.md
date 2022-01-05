# Multi-Label-Classification
A neural network that can classify the image among the 228 classes and also can detect the object present in the image
GitHub Repo link for code:- 

Data Preparation
Notebook link:-
To train the model we need to have a train and validation dataset along with their respective images and their labels.
To load the images we can directly pass the path to the directory in which train and validation images.
If we load the labels directly into the model there will be 15000 different labels for the 15000 images of the training dataset, and 3000 labels for the 3000 images in the validation dataset.
So we need to understand sending the raw labels could not be helpful to train the model
We process the data and if we look clearly at each image has a list of labels given to them which are basically classes that they belong to. Thus we need to create a data frame that can resemble all the labels in the dataset. It means classes from [1 - 228].

So we implement some pre-processing of train and validation labels files and create respective data frames 
These dataframes can represent a cell value with 1 if the respective column value is present in that particular label of the image, else it is marked as 0
Next, we create dataset preparation using imagedatagenerator() and flow_from_dataframe() methods. We chose the flow_from_dataframe method because have the labels in dataframes, thus the model can know for a respective image it can look to the respective cell values that we have passed.
We have set the target size to be (128, 128) to minimize the load on the model.
Similarly, we create the test dataset

Build and Compile the Neural Network.

We create the model network by using the Sequential layer as the base layer and adding a few convolutional layers on top of it.
For every convolutional layer, we set the kernel size as (5, 5) with the activation function as ‘relu’
After a set of convolution layers, we need to convert the outputs returned by the last layer into a vector which is known as the flattening layer. This flattens layer determines the number of input nodes to the neural network.
We need to have a certain number of nodes in the dense layer such that it could handle the data received from the previous layer. We calculate it by using the average of the number of nodes in flatten layer and the number of nodes in the output layer.
Flatten has 1024 nodes and we need 228 nodes in the output layer because we have 228 classes. So [1024 + 228]/2  -> 626. Thus we add a certain number of nodes close to this average value.
Then we train the model using the training dataset and validation dataset for 10 epochs.

Model Evaluation

We can plot the accuracy and loss function for train and validation dataset by the model using matplot library
To predict the results we pass the test dataset to model.predict()
We normalize the predicted values to [0, 1] using lambda expressions and display the predictions with associated images name in a dataframe
Finally, we process the dataframe to store the names of the images with a list of column names of cell values that are having value =1

Export these predictions dataframe to the result.csv file

Object Detection

For object detection, we use a VGG-19 pre-trained network that can classify up to 1000 objects. 

To perform detection on 6000 test images might take up quite a bit of time so in order to ease the process I have performed detections on randomly chosen 1000 images from the test dataset

To load the image we use load_image from keras.preprocessing.image that accepts the path of the image file. So we create a list of 1000 image file names along with their path as file names.



The VGG-19 model accepts images in the form of a NumPy array that has 4-dimensions. So we transform the pixels of the image into a NumPy array with each cell having a value ranging between [0 to 256] and add one more dimension to this array.



We can predict the objects in the image now and we iterate through each file name present in the list that we have created previously.



Finally, we can save the predictions to the csv file.


Hopefully, I have mentioned everything clearly about the approach for the Multi-label classification and object detection from the dataset. Please do revert back to me if there are any doubts/clarifications regarding my code.
