# cse-455-final-project
Simran Malhi, Ash Luty, Amrutha Srikanth, Nik Smith

main.py = for running experiments

model.py = model implementation

data_parsing.py = for loading data, processing data, and splitting it into training and testing

---

## Abstract

In this project, we analyze a dataset provided by Kaggle in order to create a American Sign Language (ASL) Alphabet Translator, which can then be used to predict an ASL letter from a provided image. In order to analyze the provided dataset, we train a Convolutional Neural Network (CNN) model to predict the correct label for the input image, whether a letter from the English Alphabet, a space character, the delete character, or an empty character. We chose an CNN model because it automatically learns and detects the important features and parts of input on its own, which is desired for our ASL image classification system. 

####Model Results### TODO
However, our model was unsuccessful in its predictions: the predicted track scores converged at one value despite the wide variety in the actual scores, which resulted in the same ten songs getting recommended for each user, regardless of their listening habits and actual audio feature preferences. We suspect that this is because the model underfits our data due to the layers and parameters used. In the future, we plan to experiment with different numbers and kinds of hidden layers, different loss functions, and different learning rates and weight decays to see if we can improve our model's ability to learn the user and track features.

---

## Problem

American sign language (ASL) is used by around 500,000 deaf people in the US and Canada, primarily by the deaf or hard of hearing (1). What makes ASL unique from other languages is that it is conveyed through video in the form of sequences of hand gestures, different from auditory or handwritten forms of communication. This offers a unique accessibility challenge for forms of media like broadcasted video, which may include captions based on speech detection for spoken language, but may not support captions for sign language.
	
Creating an ASL alphabet translator would help improve the accessibility of videos which display sign language, for audiences who do not understand sign language, bringing communities together through thoughtful technological innovation. From the detected transcription of signed gestures, we then use existing technologies to generate audio tracks or caption tracks to help the audience understand what is being signed. 

Stakeholders for our project include both those who use ASL, primarily deaf or hard of hearing individuals, since they would be able to communicate with a wider audience. For the Americans who can’t understand ASL, a significant majority, our project would help them connect with and understand a small but significant population of individuals who previously may have been unheard.

---

### Our Chosen Dataset: The Kaggle ASL Alphabet Dataset (featuting 29 different ASL Alphabet Characters)

There are various public datasets that we could use in order to train and test our model. The main dataset we used is a collection of images of alphabets from American Sign Language, which was separated into 29 folders, each representing a character class. [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) In the dataset, there are 29 classes, of which 26 are for the letters A-Z and 3 classes for “space”, “delete” and “nothing”. While there are 29 images in this dataset for testing, we will also be testing with our own real-life images so we can have a better view of the usability of our model. We will ensure that the dataset split is 80% training data and 20% testing data.

The initial dataset is split into a training and testing dataset. The initial training dataset contains 87,000 images with the resolution of 200x200 pixels, split into 29 character classes. The initial testing dataset contains 29 images, one for each character class. Because of the extremely small nature of the initial testing dataset, we modified the training and testing dataset so that the proportions of the training and testing datasets were more evenly distributed (80% training data and 20% testing data). The modified training dataset we created had 2400 train images per class with 69600 images overall. The modified testing dataset we created had 601 test images per class with 17428 images overall.

---

## Methodology

### Overview
We approached this problem using the following steps:

By running `data_parsing.py`:

1. Retrieving ASL Alphabet data and creating the training and testing dataset to train the model on.

By running `main.py`:

2. Initializing the parameters (train and test batch sizes, epochs, learning rate, momentum, weight decay, and test accuracy/loss printing interval) and the Stochastic Gradient Descent (SGD) optimizer.

3. Defining and training a model to learn the features of the 29 ASL character classes and predict the label of the test image through invoking `model.py`.

### Retrieving ASL Alphabet Data

Retrieving data from Spotify was handled by the `create_spotify_dataset.py` script. 

This involves three major steps:

1. Create a Spotify API client, prompting the current user to sign in

2. Retrieve the current user's user id, then iterate through playlists, saved tracks, recently played tracks, and followed artists to add tracks to the dataset

3. Write the dataset to a file (`spotify_dataset.csv`), appending it to the end of any existing data (not overwriting)


### Pre-Processing Data

Processing Spotify data for training and testing our model was handled by the `process_data.py` script. 

This involves five major steps:
   
   1. `process_user_data` reads from the overall Spotify dataset and calculates the user's preferred feature values by taking a weighted average of the features of each track in the dataset
        - Reads the user_id and the 9 track audio features for each track that the user has listened to
        - For each user, calculate the weighted average of each feature
            - feature_weighted_avg = sum(feature_value(i) * feature_weight(i) for i in range(len(num_tracks))) / num_tracks`
        - Returns a mapping of: {user_id: [feature_1_weighted_avg,...,feature_9_weighted_avg]}
   
   2. `process_track_data` reads from the overall Spotify dataset and stores the feature values for each track 
        - Reads the track_id and 9 track audio features
        - Returns a mapping of: {track_id: [feature_1_value,...,feature_9_value]}

   3. `item_to_onehot` converts the user ids and track ids (stored as alphanumeric strings) into an array representation of 1s and 0s (a onehot encoding). This is done because neural networks can't process categorical data. Both user ids and track ids are padded to 30 characters during encoding, since this is the maximum allowed length for a Spotify user id. `onehot_to_item` does the reverse conversion at the very end of the program when we want to print out the string track_ids for the user to see the track recommendations
        - `item_to_onehot`
            - takes in a string user_id or track_id
            - converts each character of the user_id into an array representation where the length of the array is the number of possible characters with a 1 at the index of the appropriate character and 0s elsewhere
                - ex. when possible chars = [a, b, c], the array for 'a' = [1, 0, 0]
            - merges the characters arrays into a 1d array
                - ex. when possible chars = [a, b, c], the return arr for string "abc" is [1, 0, 0, 0, 1, 0, 0, 0, 1]
            - pads the id to have 30 characters to ensure all encodings are of the same length by adding (30 - num_characters) arrays of zeros to the end of the 1d array before returning it
                - ex. when the string is "abc" the padded array is [1, 0, 0, 0, 1, 0, 0, 0, 1, 0........0]
        - `onehot_to_item`
            - does the reverse of one_hot encoding: takes in the 1d array of 1s and 0s and returns the string id
   
   4. `get_scores` takes user feature preferences (output of `process_user_data`) and track features (output of `process_track_data`) and calculates a compatibility score for each user/track combination using a dot product of user_feature and track_feature. These scores are normalized between 0 - 1 and get added to a dataset along with the one-hot encoded user ids and track ids (outputs of `item_to_onehot`)
   
   5. `get_train_test_data` takes the dataset of scores (output of `get_scores`) and separates it into training and test data. The training data is generated by randomly selecting 80% of rows (user_encoding, track_encoding, score) for each user. The test data is the remaining 20% of the rows in the dataset.


### Neural Network Model Definition

We trained a Neural Collaborative Filtering model on the dataset in order to generate track scores for the users. We chose an NCF model because it learns and predict user-item interactions based on past interactions. The Spotify dataset is user-song interaction based, which is implicit feedback, which makes it easier to gather lots of data. NCFs are able to utilize this implicit feedback in the data in order to learn user song audio preferences in order to predict the likelihood of the user liking a particular track. Thus, this model seemed like a good deep learning solution for a Spotify song recommendation system. This network and the train and test methods are defined in `network.py`.

The network first generates 2 embedded layers (1 for users and 1 for tracks) to represent their traits in a lower dimensional space in order to learn the features of the users and tracks. We had 9 features for the embedded layers, since each user and track has 9 traits.

The network then concatenate the user and track embeddings into one vector and passes it through a series of Linear layers with ReLU activation functions to map the embeddings to score predictions. Our final model had 4 Linear + ReLU passes followed by a final Linear layer. The first linear layer takes in 18 features because the concatenated vector has 18 features (9 user features + 9 track features). The final linear layer had a output of 1 because we want one score prediction for each (user, track) pairing.

Finally, we then pass the predicted scores through a sigmoid function to ensure that they would be between 0 and 1, since the calculated scores were normalized between 0 and 1.

For our loss function, we chose to use the MSE loss function because our problem is a regression problem and we are trying to minimize the difference between the predicted and actual scores.

### Song Predictions
After the model has been trained and evaluated with the training and testing data, we print out the top ten song predictions per user using the following steps:

1. Convert the user onehot and track onehot encodings back to their string id representations using `onehot_to_item`.

2. Get the track name and artist for each track_id

3. For each user:
    - Get the model score predictions for all of the tracks in the entire dataset

    - Sort the (track_id, scores) from highest to lowest scores

    - Print out the track name and artist for the first ten (track_id, score) items in the sorted list

---

## Experiments/Evaluation


### Experiments

For our experiments, we experimented with changing our batch sizes, epochs, learning rate, and weight decay to improve our test accuracies. As a reminder, we gathered the user data for our experiments through running `create_spotify_dataset.py` for different Spotify users. We then ran our experiments by running `main.py`, which:

   1. Loaded and processed the training and testing data from `spotify_dataset.csv` through invoking `process_data.py`.

   2. Initialized the parameters and the Adam optimizer
   
   3. Created and trained the model through invoking `network.py`. When training, per epoch it prints out the test loss for each batch along with the average test loss and test accuracy.

   4. Generated the top ten song recommendations per user by selecting the top ten songs with the highest scores predicted by the model

Our worst model had batch sizes of 512, 20 epochs, a learning rate of 0.1, and a weight decay of 0.0005. This had test accuracies that fluctuated dramatically between epochs, bouncing between test accuracies from around 26% to 73%.
![img1](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/results/worst-test-acc.png)
![img2](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/results/worst-test-loss.png)
![img3](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/results/worst-train-loss.png)

Our best model had batch sizes of 50, 50 epochs, a learning rate of 0.001, and a weight decay of 0.0005. Since this model had a final epoch test accuracy of 58% and had accuracies consistently between 40% - 60%, we stuck with it.
![img4](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/results/best-test-acc.png)
![img5](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/results/best-test-loss.png)
![img6](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/results/best-train-loss.png)

We would have experimented more with different network architectures to find the most optimal one, however, as mentioned in the Results section, we weren't able to create a model that fulfilled all of our base evaluation criteria.


### Evaluation

To evaluate the accuracy of the score predictions, we calculated the test accuracies for each epoch by subtracting the difference of each predicted and actual scores and counting the number of predicted values that had a difference in value less than 0.00000001. We determined that a final test accuracy greater than 30% would be acceptable.

To evaluate the accuracy of the recommender system as a whole, we manually reviewed each of the ten song recommendations per user. We determined that the evaluation would be a success if each user had reasonably different song recommendations that align with the songs that they listen to and contain at least one song that they have listened to before. 

---

## Results

Our results met the test accuracy criteria of being over 30%, as our final epoch had a test accuracy of 58%. However, this only meant that the predicted float values weren't very off from the actual values, which was easy to achieve since both the predicted and actual values were between 0 and 1. 

Unfortunately, our model didn't accurately recommend songs for each of the users. Our model generated the same predicted score values for every user, track pairing. Although each user had very distinct and diverse track lists, the model wasn't able to accurately learn the user and track features in order to generate many different score values within the same iteration. For example, the predicted scores for one user was ```[0.5373, 0.5373, 0.5373, 0.5373, 0.5185, 0.5373, 0.5373, 0.5373, 0.5373]``` when the actual scores were ```[0.6605, 0.4063, 0.3994, 0.4503, 0.4685, 0.4467, 0.6143, 0.3738, 0.4584]```. This lead to the same set of songs being recommended for every user. 

We have several possible ideas for why this model didn't work. However, because of time constraints, we weren't able to create a better model on time.

1. **We did not pick the right activation function.** 
    
    It is possible that the ReLU activation function caused the predicted values to converge too quickly, which would explain why almost all of the predicted scores were the same value. If we found a better activation function, it is possible that the model would have predicted a more diverse range of values for each track and have had more accurate song recommendations per user.

2. **We did not have enough hidden layers.** 
    
    It is possible that we didn't have enough layers for the model to find patterns between the input and output vectors and learn the features of the data with our given parameters (epoch, learning rate, weight decay, batch size).
    
3. **We did not have good model parameters, specifically the learning rate and weight decay.** 
    
    It is possible that our learning rate was still too large, which caused our model to keep overshooting. Our weight decay may have been too large, which caused our model to severely underfit the data and only create one score prediction at a time for each batch of users and tracks.

4. **We did not use the correct loss function.**
    
    We chose to use the MSE loss function we want to minimize the difference between the predicted and actual scores. However, it is possible that our loss function prevented our model from being able to properly fit the data and that there is a better loss function that we can use.
---

## Example Outputs

### Output for worst model
generating song recommendations...
Top 10 song recommendations for Sylvi
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for cruella
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for Josh Seitz
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for OpalApple
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for rohan
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar
done

### Output for best model
generating song recommendations...
Top 10 song recommendations for Sylvi
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for cruella
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for Josh Seitz
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for OpalApple
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar

Top 10 song recommendations for rohan
1. Aap Jaisa Koi - Qurbani / Soundtrack Version by Nazia Hassan
2. Aao Chalein by Taba Chake
3. Papa Kahte Hain by Udit Narayan
4. Suhani Raat Dhal Chuki by Mohammed Rafi
5. Patakha Guddi by Sultana
6. Nadiyon Paar (Let the Music Play Again) (From "Roohi") by Sachin-Jigar
7. Aaj Se Pehle Aaj Se Jyada by K. J. Yesudas
8. Tujhse Naraz Nahin Zindagi - Male Vocals by Anup Ghoshal
9. Pukarta Chala Hoon Main (From "Mere Sanam") by Mohammed Rafi
10. Mere Sapnon Ki Rani (From "Aradhana") by Kishore Kumar
done
---

## Video

Demo video can be found [here](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/cse490g1-final-project-video.mp4).

---

## Code

Code for this project can be found [here](https://github.com/simranmalhi/cse-455-final-project).
