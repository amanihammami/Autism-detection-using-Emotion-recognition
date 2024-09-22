# Autism-detection-using-Emotion-recognition

![autist](https://github.com/user-attachments/assets/0a8c8ab6-11cb-4ef1-a11b-cd27a31388b4)


**This study explores the use of emotion recognition to detect autism spectrum disorder (ASD) in children, employing the ResNet 50 model to analyze seven key emotions that may signify autism-related traits.**

The project aims to create a video, captured in a kindergarten, to evaluate a child's emotions and determine signs of autism. If the child displays an emotional level significantly higher than that of a typical child, an alarm will be triggered to alert healthcare professionals, indicating the need for further evaluation.
To determine if a child is autistic, detecting their emotions is a key factor. To this end, several algorithms have been designed using convolutional architectures. These algorithms leverage specific facial expression features to identify and classify emotions, providing a potentially valuable tool for assisting in the early diagnosis of autistic disorders.
I implemented a model based on ResNet50 and incorporated additional layers to classify 7 emotions from facial expressions.
## Architecture of the model ##
 1)BaseModel: This layer is composed of several convolutional blocks followed by pooling layers, residual blocks (a series of residual blocks that form skip connections, enabling deeper training), and a global average pooling layer that reduces the output dimensionality to a feature vector.
 
2)Dropout (rate 0.5): A regularization layer that randomly deactivates a fraction of neurons during training to reduce overfitting.

3)Flatten: This layer transforms the data into a one-dimensional vector to be used by the following Dense layers.

4)BatchNormalization: A layer that normalizes the activations of the previous layer to speed up training and reduce overfitting.

5)3 Dense Layers: These layers are fully connected layers that learn from the features extracted by the previous layers. Each layer is followed by a batch normalization layer, a ReLU activation layer, and a dropout layer.

6)Final Dense Layer with Softmax Activation: The last dense layer produces the output probabilities for the classes in our problem, with a softmax activation to obtain a probability distribution over the classes.
## Hyperparameters ##
- Epochs:60
- Batch Size: 64,128,256
- optimizer='Adam'
## Data Augmentation ##
- To artificially increase the size of a dataset by creating variations of the existing data.
- they include rotation, width_shift, height_shift,shear,zoom,horizontal_flip,vertical_flip etc
- Was applied only on training set.
## Accuracy ##
the training accuracy reaches 85.71%. Test set was evaluated and accuracy achieved was 83%
## Dataset ##
Fer2013 contains approximately 30,000 RGB facial images of different expressions, limited to a size of 48×48 pixels, and its main labels can be divided into 7 types:
0 = Anger, 1 = Disgust, 2 = Fear, 3 = Joy, 4 = Sadness, 5 = Surprise, 6 = Neutral.
The expression 'Disgust' has the minimum number of images - 600, while the other labels have nearly 5,000 samples each. Here are some examples for each emotion
## Tools used ##
- Python
- TensorFlow
- OpenCV
- Numpy
## Detector ##
-Uses Haar Cascade Classifiers in OpenCV to detect faces.

-Preprocesses the detected face and resizes the face into 48x48 sized image

-Passes the resized image into the model for emotion prediction

-Predicted emotion is then displayed along with the bounding boxes on the face.


[Open the code in Google Colab](https://colab.research.google.com/drive/1HNjN3NUGkVYNPqV1PDuZYRmiS0iU4OAG)
