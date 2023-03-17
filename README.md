# cse-455-final-project
Simran Malhi, Ash Luty, Amrutha Srikanth, Nik Smith

main.py = for training the model and getting results (used for running experiments)

model.py = model implementation

data_parsing.py = for loading data, processing data, and splitting it into training and testing

---

## Abstract

In this project, we analyze a dataset provided by Kaggle in order to create a American Sign Language (ASL) Alphabet Translator, which can then be used to predict an ASL letter from a provided image. In order to analyze the provided dataset, we train a Convolutional Neural Network (CNN) model to predict the correct label for the input image (whether a letter from the English Alphabet, a space character, the delete character, or an empty character). We chose an CNN model because it automatically learns and detects the important features and parts of input on its own, which is desired for our ASL image classification system. 

Our model was pretty successful in its predictions for the training data, but was overfitting, since the testing
accuracy was much lower than the training accuracy. With a model structure of 5 layers training for 5 epochs with a
learning rate of 0.002, momentum of 0.9, and a weight decay of 0.005, the model's final training accuracy was about 98%
and its final testing accuracy was about 18%. We decided to train on 5 epochs for the sake of the training time, but we suspect that with more epochs the testing accuracy could have been higher. Since the model is overfitting, our future work 
is to try training a model with less layers or with batch normalization to reduce the overfitting.

---

## Problem

American sign language (ASL) is used by around 500,000 deaf people in the US and Canada, primarily by the deaf or hard of hearing (1). What makes ASL unique from other languages is that it is conveyed through video in the form of sequences of hand gestures, different from auditory or handwritten forms of communication. This offers a unique accessibility challenge for forms of media like broadcasted video, which may include captions based on speech detection for spoken language, but may not support captions for sign language.
	
Creating an ASL alphabet translator would help improve the accessibility of videos which display sign language, for audiences who do not understand sign language, bringing communities together through thoughtful technological innovation. From the detected transcription of signed gestures, we then use existing technologies to generate audio tracks or caption tracks to help the audience understand what is being signed. 

Stakeholders for our project include both those who use ASL, primarily deaf or hard of hearing individuals, since they would be able to communicate with a wider audience. For the Americans who can’t understand ASL, a significant majority, our project would help them connect with and understand a small but significant population of individuals who previously may have been unheard.

---

### Our Chosen Dataset: The Kaggle ASL Alphabet Dataset (featuting 29 different ASL Alphabet Characters)

There are various public datasets that we could use in order to train and test our model. The main dataset we used is a collection of images of alphabets from American Sign Language, which was separated into 29 folders, each representing a character class. [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) 

In the dataset, there are 29 classes, of which 26 are for the letters A-Z and 3 classes for “space”, “delete” and “nothing”. While there are 29 images in this dataset for testing, we also tested with our own real-life images so we can have a better view of the usability of our model. We also ensured that the dataset split is 80% training data and 20% testing data per
character class.

---

## Methodology

### Overview
We approached this problem using the following steps:

By running `data_parsing.py`:

1. Retrieving ASL Alphabet data and creating the training and testing dataset to train the model on.

By running `main.py`:

2. Initializing the parameters (train and test batch sizes, epochs, learning rate, momentum, weight decay, and test accuracy/loss printing interval) and the Stochastic Gradient Descent (SGD) optimizer.

3. Defining and training a model to learn the features of the 29 ASL character classes and predict the label of the test image through invoking `model.py` in `main.py`.

### Retrieving ASL Alphabet Data

In order to retrieve the appropriate data from Kaggle, we first downloaded the Kaggle dataset into the same folder as our code since our code was written based on the assumption that the ASL Alphabet Kaggle data is already downloaded. We then manipulated the initial dataset so it would be more ideal for our training and testing purposes. 

The initial dataset is split into a training and testing dataset. The initial training dataset contains 87,000 images with the resolution of 200x200 pixels, split into 29 character classes. The initial testing dataset contains 29 images, one for each character class. Because of the extremely small nature of the initial testing dataset, we modified the training and testing dataset so that the proportions of the training and testing datasets were more evenly distributed (80% training data and 20% testing data). The modified training dataset we created had 2400 train images per class with 69600 images overall. The modified testing dataset we created had 601 test images per class with 17428 images overall. We created a shell script to auto-generate .csv files containing the file names of each image and their corresponding label number.

### Pre-Processing Data

Processing the ASL Alphabet data for training and testing our model was handled by the `data_parsing.py` script. 

This involves two major pieces, the `get_ASL_data` method, which creates the training and testing dataloaders, and the custom ASL Dataset we created for this model:
   
   1. `get_ASL_data` gets the ASL Alphabet data from the downloaded data folders and creates the two datasets and two dataloaders for use in training and testing of the model. It also creates a list of character classes for use in labeling and categorizing the input images. The method then returns the training dataloader, the testing dataloader, and the character class list for use in the main model training program.
   
   2. `ASLDataset` is a custom dataset we created in order to accomadate for our more unique dataloading method. Within this custom dataset, we included an init method, a length method, and a getitem method. Because our data was in the form of an image and a corresponding character label, we had to customize these three methods to save and return images and their corresponding labels.

### Model Definition

We trained a Convolutional Neural Network (CNN) model on the dataset in order to predict the ASL letter from the input image. We chose a CNN because it automatically learns and detects the important features and parts of input on its own, which is desired for our ASL image classification system. This network and the train and test methods are defined in `model.py`.

Our network has a kernel size of 5 and five convolutional layers with ReLU acrivation functions to learn the features of the daraset. The convolutional layers have a starting in_channels of 3 (using rgb channel) and an final out_channels of 64. The model then has a final linear layer to generate the prediction, which has in_features of the resulting features from the convolutional layers and has 29 out_features (since there are 29 different possible image labels from the dataset).

For our loss function, we chose to use the Cross Entropy loss function because our problem is for classification.

For our optimizer, we used a SGD optimizer because it typically generalizes better in comparison to the Adam optimizer.

After the model has been trained and evaluated with the training and testing data, we plot out four graphs based on training loss, training accuracy, testing loss, and testing accuracy of model. We then compared the testing accuracies and training accuracies of multiple experiments with varying hyperparameters to find the best results.

---

## Experiments/Evaluation


### Experiments

For our experiments, we experimented with changing our epochs, learning rate, momentum, and weight decay to improve our test accuracies. As a reminder, we gathered the user data for our experiments by downloading the ASL Alphabet Kaggle Dataset and running `data_parsing.py` to organize the datasets into training and testing dataloaders. We then ran our experiments by running `main.py`, which:

   1. Loaded and processed the training and testing data from the dataloaders provided by `data_parsing.py`.

   2. Created the Convolutional Neural Network Model by calling upon `model.py`
   
   3. Initialized the parameters and the SGD optimizer
   
   3. Trained the model through invoking methods from `model.py`, including `model.train`, `model.test`, and `model.train_acc`. When training, per epoch it prints out the training loss for each batch along with the average test loss and test accuracy.

   4. Plotted/Graphed four graphs based on training loss, training accuracy, testing loss, and testing accuracy of model

Our worst model trained for 5 epochs and had a learning rate of 0.002, a momentum of 0.9, and a weight decay of 0.1.
This had a final training accuracy of 85% and a final testing accuracy of 9.2%.
![img1](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/worst-test-acc.png)
![img2](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/worst-test-loss.png)
![img3](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/worst-train-acc.png)
![img4](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/worst-train-loss.png)

Our best model trained for 5 epochs and had a learning rate of 0.002, a momentum of 0.9, and a weight decay of 0.005.
This had a final training accuracy of 98% and a final testing accuracy of 18%.
![img5](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/best-test-acc.png)
![img6](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/best-test-loss.png)
![img7](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/best-train-acc.png)
![img8](https://github.com/simranmalhi/cse-455-final-project/blob/main/results/best-train-loss.png)


### Evaluation

To evaluate the accuracy of the image label predictions, we calculated the test accuracies for each epoch by comparing the expected image label with the predicted image label and calculating the percentage of predicted values that were correct. We did a similar calculation for the training accuracy. We determined that a model with final training and testing accuracies greater than 60% each would be acceptable.

With this baseline, our model doesn't meet the evaluation criteria. Our best model had a final training accuracy of 98% and a final testing accuracy of 18%. Although our training accuracy met the evaluation criteria, the testing accuracy was too low.
This is because our model is overfitting the training data. Although we ran experiments and were able to improve the training and testing accuracies (as shown above), we did not raise the testing accuracy enough.

We would have experimented more with different network architectures to find a model with less overfitting to improve the testing accuracy, however, we didn't have enough time and will leave that for future work. Our ideas for reducing the overfitting so that both the training and testing accuracies meet the evaluation criteria is: training for more epochs, reducing the number of layers, and using batch normalization between the convolutional layers.

---

## Example Outputs (TODO)


---

## Video (TODO)

Demo video can be found [here](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/cse490g1-final-project-video.mp4).

---

## Code

Code for this project can be found [here](https://github.com/simranmalhi/cse-455-final-project).
