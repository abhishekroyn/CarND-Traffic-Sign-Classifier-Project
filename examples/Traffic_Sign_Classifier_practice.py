
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[2]:


# Load pickled data
import pickle
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.utils import shuffle

# TODO: Fill this in based on where you saved the training and testing data

training_file = '../../traffic-signs-data/train.p'
validation_file= '../../traffic-signs-data/valid.p'
testing_file = '../../traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[17]:

if False:
    ### Replace each question mark with the appropriate value. 
    ### Use python, pandas or numpy methods rather than hard coding the results

    # TODO: Number of training examples
    n_train = X_train.shape[0]

    # TODO: Number of validation examples
    n_validation = X_valid.shape[0]

    # TODO: Number of testing examples.
    n_test = X_test.shape[0]

    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train.shape[1], X_train.shape[2], X_train.shape[3]

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(list(set(y_train)))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[ ]:

if False:
    ### Data exploration visualization code goes here.
    ### Feel free to use as many code cells as needed.
    # Visualizations will be shown in the notebook.
    # get_ipython().run_line_magic('matplotlib', 'inline')

    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()

    plt.figure(figsize=(1,1))
    plt.imshow(image, cmap="gray")
    plt.show()
    print(y_train[index])

if False:
    data = pd.read_csv('../signnames.csv')
      
    num_of_samples=[]

    cols = 5
    num_classes = 43

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
    fig.tight_layout()

    for i in range(cols):
        for j, row in data.iterrows():
          x_selected = X_train[y_train == j]
          axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
          axs[j][i].axis("off")
          if i == 2:
            axs[j][i].set_title(str(j) + " - " + row["SignName"])
            num_of_samples.append(len(x_selected))
    plt.show()

if False:
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the train dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    assert(X_train.shape[0] == np.sum(num_of_samples)), "The total number of training images is not equal to the sum total of number of training images for each of the labels."

# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[36]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# convert image into grayscale
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

# convert the RGB image to YUV format
def rgb_to_yuv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

# histogram equalization to improve contrast of image
def equalize(img):
    # equalize the histogram of the Y channel
#    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.equalizeHist(img)
    return img

# convert the YUV image back to RGB format
def yuv_to_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img

# normalize data
def normalize(img):
#    img = (img - 128)/ 128
    img = img/255
    return img

def preprocess(img):
    img = grayscale(img)
#    img = rgb_to_yuv(img)
    img = equalize(img)
#    img = yuv_to_rgb(img)
    img = normalize(img)
    return img

# preprocess training, testing and validation data
X_train = np.array(list(map(preprocess, X_train)))
X_test = np.array(list(map(preprocess, X_test)))
X_valid = np.array(list(map(preprocess, X_valid)))


# reshape data
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 32, 32, 1)

# ### Model Architecture

# In[ ]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

EPOCHS = 10          # 10
BATCH_SIZE = 50

from tensorflow.contrib.layers import flatten

#def modified_model(x):    
#    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
#    mu = 0
#    sigma = 0.1
#    
#    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
#    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
#    conv1_b = tf.Variable(tf.zeros(6))
#    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

#    # SOLUTION: Activation.
#    conv1 = tf.nn.relu(conv1)

#    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
#    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

#    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
#    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
#    conv2_b = tf.Variable(tf.zeros(16))
#    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#    
#    # SOLUTION: Activation.
#    conv2 = tf.nn.relu(conv2)

#    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
#    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

#    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
#    fc0   = flatten(conv2)
#    
#    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
#    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
#    fc1_b = tf.Variable(tf.zeros(120))
#    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
#    
#    # SOLUTION: Activation.
#    fc1    = tf.nn.relu(fc1)

#    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
#    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
#    fc2_b  = tf.Variable(tf.zeros(84))
#    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
#    
#    # SOLUTION: Activation.
#    fc2    = tf.nn.relu(fc2)

#    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
#    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
#    fc3_b  = tf.Variable(tf.zeros(43))
#    logits = tf.matmul(fc2, fc3_W) + fc3_b
#    
#    return logits

def modified_model(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x60.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 60), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(60))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Layer 1: Convolutional. Input = 28x28x60. Output = 24x24x60.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 60, 60), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(60))
    conv1   = tf.nn.conv2d(conv1, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 24x24x60. Output = 12x12x60.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Input = 12x12x60. Output = 10x10x30.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 60, 30), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(30))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Layer 2: Convolutional. Input = 10x10x30. Output = 8x8x30.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 30, 30), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(30))
    conv2   = tf.nn.conv2d(conv2, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 8x8x30. Output = 4x4x30.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 4x4x30. Output = 480.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 480. Output = 500.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(480, 500), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(500))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # add dropout
    fc1 = tf.nn.dropout(fc1, 0.5)

    # SOLUTION: Layer 4: Fully Connected. Input = 500. Output = 43.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(500, 43), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc1, fc2_W) + fc2_b
    
    return logits

## Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

## Training Pipeline
rate = 0.001

logits = modified_model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

## Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[ ]:

if False:
    ### Train your model here.
    ### Calculate and report the accuracy on the training and validation set.
    ### Once a final model architecture is selected, 
    ### the accuracy on the test set should be calculated and reported as well.
    ### Feel free to use as many code cells as needed.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            
        saver.save(sess, './modified_model')
        print("Model saved")

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
#predict internet number
#import requests
#from PIL import Image
#url = 'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg'
#r = requests.get(url, stream=True)
#img = Image.open(r.raw)

#img = cv2.imread('../test_images/GK10NJ.jpg') 
#cv2.imshow('test image', img)
#cv2.waitKey(0)

# Make a list of test images and save the results

def evaluate_image_top_index(img):

    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    #plt.imshow(img, cmap = plt.get_cmap('gray'))
    #plt.show()
    #print(img.shape)
    img = img.reshape(1, 32, 32, 1)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        # top argmax index
        prediction_max = sess.run(tf.argmax(tf.nn.softmax(logits), axis=1), feed_dict={x: img})
        print("\nBest predicted sign index: "+ str(prediction_max))
        print("\n")
    return prediction_max

def evaluate_image_top_k_index(img):

    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    #plt.imshow(img, cmap = plt.get_cmap('gray'))
    #plt.show()
    #print(img.shape)
    img = img.reshape(1, 32, 32, 1)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        # top k argmax indices
        k = 5                               # required count of top results  
        prediction_max_k = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k = 5), feed_dict={x: img})
        print("\nTop {0} predicted sign indices       : {1}".format(k, prediction_max_k.indices))
        print("\nTop {0} predicted sign probabilities : {1}".format(k, prediction_max_k.values))
        print("\n")

import glob
images = sorted(glob.glob('../test_images/test*.jpg'))
images_actual_id =  np.array([[14, 34, 38, 9, 27]])     # test images label id
images_predicted_id = []

for idx, img in enumerate(images):

    print("Image name : {}".format(img))
        
    # Read in image
    img = cv2.imread(img)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    print("Label ID for the image : {}".format(images_actual_id[0][idx]))
    print("\n")

    # best predicted sign index
    prediction_max = evaluate_image_top_index(img)

    # check accuracy
    images_predicted_id.append(prediction_max)

    # best predicted sign index
    evaluate_image_top_k_index(img)

    #img = mpimg.imread('../test_images/STOP_sign.jpg')
    #plt.imshow(img)
    #plt.show()

# check accuracy
images_predicted_id = np.transpose(np.array(images_predicted_id))

print("Images actual id    : {}".format((images_actual_id)))
print("images predicted id : {}".format((images_predicted_id)))

assert(images_predicted_id.shape == images_actual_id.shape), "The total count of test images evaluated is not equal to the total count of test images assumed to have provided as input."

correct_prediction_count = np.sum(images_predicted_id == images_actual_id)
accuracy_percentage = (correct_prediction_count / images_actual_id.shape[1]) * 100

print("Accuracy percentage {:.2f}".format(accuracy_percentage))

# ### Predict the Sign Type for Each Image

# In[ ]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

#def evaluate_image_top_index(img):
#    img = np.asarray(img)
#    img = cv2.resize(img, (32, 32))
#    img = preprocess(img)
#    #plt.imshow(img, cmap = plt.get_cmap('gray'))
#    #plt.show()
#    #print(img.shape)
#    img = img.reshape(1, 32, 32, 1)

#    with tf.Session() as sess:
#        saver.restore(sess, tf.train.latest_checkpoint('.'))

#        # top argmax index
#        prediction_max = sess.run(tf.argmax(tf.nn.softmax(logits), axis=1), feed_dict={x: img})
#        print("\nbest predicted sign index: "+ str(prediction_max))
#        print("\n")

# ### Analyze Performance

# In[ ]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


# In[ ]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

#def evaluate_image_top_k_index(img):
#    with tf.Session() as sess:
#        saver.restore(sess, tf.train.latest_checkpoint('.'))

#        # top k argmax indices
#        k = 5                               # required count of top results  
#        prediction_max_k = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k = 5), feed_dict={x: img})
#        print("\ntop {} predicted sign indices : ".format(k)+ str(prediction_max_k.indices))
#        print("\ntop {} predicted sign probabilities : ".format(k)+ str(prediction_max_k.values))
#        print("\n")

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

