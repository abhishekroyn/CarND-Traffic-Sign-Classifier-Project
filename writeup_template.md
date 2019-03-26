# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/single_image_sample.jpg "Single image sample.jpg"
[image2]: ./output_images/multiple_images_sample.jpg "Multiple images sample.jpg"
[image3]: ./output_images/train_dataset_distribution.jpg "Train dataset distribution.jpg"
[image4]: ./output_images/test_image_preprocessed_0.jpg "Traffic Sign preprocessed 1"
[image5]: ./output_images/test_image_preprocessed_1.jpg "Traffic Sign preprocessed 2"
[image6]: ./output_images/test_image_preprocessed_2.jpg "Traffic Sign preprocessed 3"
[image7]: ./output_images/test_image_preprocessed_3.jpg "Traffic Sign preprocessed 4"
[image8]: ./output_images/test_image_preprocessed_4.jpg "Traffic Sign preprocessed 5"
[image9]: ./test_images/test_1.jpg "Traffic Sign 1"
[image10]: ./test_images/test_2.jpg "Traffic Sign 2"
[image11]: ./test_images/test_3.jpg "Traffic Sign 3"
[image12]: ./test_images/test_4.jpg "Traffic Sign 4"
[image13]: ./test_images/test_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a single sample image below, 

![alt text][image1]

followed by 5 samples from each of the 43 different classes.

![alt text][image2]

Also, an exploratory visualization of the data set is provided below. It is a bar chart showing distribution of the training dataset.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color information doesn't help us to identify important edges as much as grayscale do. Also, having just single channel instead of three channels, it also helps in faster processing of image too.

Next, I applied two types of normalization one after another. The former being histogram equalization because it helps in even distribution of intensity of an image, thereby resulting in better contrast in the resulting image. While the latter being normalization of the image data which brings the pixel values within 0 and 1 values. The former will attempt to produce a histogram with equal amounts of pixels in each intensity level and uses probability distribution approach. The latter preserves relative levels, and thus together they assist in developing better contrast in the image.
 
Here are examples of original image and preprocessed images for 5 test images from the web.

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Initial image shape      : (32, 32, 3)
Preprocessed image shape : (32, 32, 1)
The preprocessed image is then fed to the model.

My final model consisted of the following layers:

| Layer         		|     Description	        				             	    | 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 32x32x1 GRAYSCALE image   							        | 
| Convolution 5x5     	| 1x1 stride, VALID padding, 60 filters, outputs 28x28x60    	|
| RELU					|												                |
| Convolution 5x5     	| 1x1 stride, VALID padding, 60 filters, outputs 24x24x60 	    |
| RELU					|												                |
| Max pooling	      	| 2x2 stride, VALID padding, 2x2 window_size, outputs 12x12x60 	|
| Convolution 3x3     	| 1x1 stride, VALID padding, 30 filters, outputs 10x10x30    	|
| RELU					|												                |
| Convolution 3x3     	| 1x1 stride, VALID padding, 30 filters, outputs 8x8x30 	    |
| RELU					|												                |
| Max pooling	      	| 2x2 stride, VALID padding, 2x2 window_size, outputs 4x4x30 	|
| Flatten       	    | outputs 480     									            |
| Fully connected		| outputs 500        									        |
| RELU					|	                                                            |
| Dropout				| keep_probabilty 0.5                           		    	|
| Fully connected		| outputs 43       									            |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I,
* converted the labels into one-hot-encoder format
* created a computation pipeline with feeding training data to the model
* calculated softmax and cross entropy on the obtained logits above
* used `tf.reduce_mean` to calculate loss
* used `Adam` optimizer to optimize the learning process by minimizing errors
* compared obtained final values from softmax function and ground-truth label values to find accuracy of the prediction
* evaluated entire training dataset using below parameter values in every epoch
* evaluated entire validation dataset using below parameter values and model values obtained after 10 epochs

| Parameter         	     |     Value	         | 
|:--------------------------:|:---------------------:| 
| EPOCHS         		     | 10   			   	 | 
| BATCH_SIZE         	     | 50   			     |
| rate (learning rate)       | 0.001 				 | 
| mean               	     | 0   				     | 
| sigma (deviation in data)  | 0.1 				     |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 96.7%
* test set accuracy of 94.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
>> I started with LeNet architecture. It is a relatively simple architecture, well proven on image-classification with good results, and it provides plenty of room to make smaller improvements on it to improvise the result.

* What were some problems with the initial architecture?
>> Number of filters were too less and layers were only a few as well. Traffic signs are complex images as they have many noise factors around them.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
>> The architecture was adjusted to add more convolutional filters and extra convolutional layers as well. Padding was kept as valid and strides were kept the same. Even the max pooling layers were kept the same. But the additional layers of filters and extra convolutional layers helped to ensure that the model could detect more features from the images at each stage. Also addition of dropout ensured partial weights get vanished and thus more of the weights get to participate in the active learning process and thus have their values updated to cover various relevant features in the images. 

* Which parameters were tuned? How were they adjusted and why?
>> Iniitially I trained with 3 epochs only to observe the model behavior and results of the changes more quickly, later I trained it for 20 epochs, and finally settled down on 10 epochs as the results were reasonably good and learning was saturared around that value.
I tried with batch size of 128, then compared with batch size of 50 and seems to get similar results without much difference, so settled on batch size of 50 itself. In future with augmented images, there will be more varying training data, and then batch size of 50 (lower value) could be more reasonable to learn from more variations.
I tried with learning rate of higher value as well. The computation was faster but performance degraded, so settled on 0.001 value. As it was already giving reasonably well results, thus I did not try with much lower values like 0.00001 as it will make the computation very slow without adding too much benefit.
I tried dropout value of 0.75 but the learning wasn't improving enough, so lowering dropout value more to 0.5 helped half of the weights to be dropped and rest half to learn aggresivly the details in each of the epochs.
I tried converting rgb to yuv images for preprocessing but histogram equalization results were relatively poor, so I choose to stick to grayscale conversion and preprocessing only.
I tried normalizing data as (pixel - 128)/ 128 but results were poor compared to (pixel/255) normalization, thus I choose latter. This might be because I was normalizing the images after histogram equalization, which is also another type of normalization technique mostly for intenstiy normalization.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
>> Most of the images have very specific well-defined patterns as edges and curves. Those once isolated from noises and identified as various individual edges can help to identify the sign in the image and distinguish one quite well. Convolution layer helps to achieve the same, as long as it is tuned well to separate noises from those edges. More complex layers may detect specific patterns which could distinguish one sign easily from another, thus convolution layer seems preferred apporach.
Dropout ensured partial weights get vanished and thus more of the weights get to participate in the active learning process and thus have their values updated to cover various relevant features in the images. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
>> Final model's accuracy was obtained to be more than 93% for test set and much higher for validation set. These results were obtained without any data-augmentation but simple preprocessing and modifying LeNet model to have additional layers and filters. Thus with improved results the model seems to be working reasonably well. I am hoping precision and recall values for each of the classes could be further helpful to verify the same.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

The third image might be difficult to classify because it has extra stripe on sign-stand in the image, which could be confusing to the model as the model may see it as a potetial traffic-sign itself.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop      		    | Stop 									        | 
| Turn left ahead       | Speed limit (80km/h)							|
| Keep right		    | Yield											|
| No passing	      	| Roundabout mandatory					 	    |
| Pedestrians			| Priority road      							|

The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares unfavorably to the accuracy on the test set of ..... Several factors might have led to this vast difference between test output results. Images from the web also seems to have varying noise and orientation for which the model might not have been trained enough due to lack of augmented training dataset. Also, based on distribution of training dataset displayed in bar-graph few of the classes have very few training datasets while others have relatively higher. Thus, the classes with less number of training data may have been trained poorly thus, precion and recall values are expected to be unsatisfactory, which also leads to difficulty in testing unseen traffic sign images from the web.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th code-cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the second image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the third image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the fourth image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the fifth image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


### Future Work
* Data augmentation types - width shift, height shift, zoom in and out, rotation, translation, image-flip, color perturbation
* Data augmentation strategies - individually or combined
* calculate the precision and recall for each traffic sign type from the test set and then compare performance on these five new images
* Looking at performance of individual sign types can help guide how to better augment the data set or how to fine tune the model
* For each of the five new images, create a graphic visualization of the soft-max probabilities. Bar charts might work well.
* Play around more preprocessing techniques
* Balance out number of examples per label by generating fake-data or data-augmentation
* tune the hyperparameters further

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


