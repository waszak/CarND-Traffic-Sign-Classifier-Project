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

[image1]: ./examples/visualization1.png "Visualization 1"
[image9]: ./examples/visualization2.png "Visualization 2"
[image10]: ./examples/visualization3.png "Visualization 3"

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./trafic_signs/11.jpg "Traffic Sign 1"
[image5]: ./trafic_signs/12.jpg "Traffic Sign 2"
[image6]: ./trafic_signs/17.jpg "Traffic Sign 3"
[image7]: ./trafic_signs/26.jpg "Traffic Sign 4"
[image8]: ./trafic_signs/27.jpg "Traffic Sign 5"

[image12]: ./trafic_signs/14.jpg "Traffic Sign 6"
[image11]: ./examples/predictions_2.png "Predicitions"

[image13]: ./examples/network.jpg "Network"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/waszak/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


* The size of training set is 34799 samples 
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram of representation of classes in trainining, validation and test dataset. The distribution is the same which is good. 

![alt text][image1]
![alt text][image9]
![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I decided to add more data because some classes had less than 200 examples. I used rotation, brigthness, color_agumentation and blur. For every image i aplied all this operations. With elements of randomness just to be safe.
I used VGG-16 normalization because it worked the best.






#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2x1 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2x1 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64  	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64  	|
| RELU					|												|
| Max pooling	      	| 2x2x1 stride,  outputs 4x4x64   				|
| Flatten				| ouputs 1x1024									|
| Dropout       	    | keep probabilty 50%      						|
| Fully connected		| 120       									|
| RELU					|												|
| Fully connected		| 42        									|
| RELU					|												|
| Sotmax				| 42 (one-hot= number of classes			    |
 
![alt text][image13]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with learning rate 0.0001, I had 20 epochs and batch size was 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Train Accuracy = 0.935
* Validation Accuracy = 0.931
* Test Accuracy = 0.930
I actually manage to get even more 95+ but it had issues with Stop sign and Flipped images so it got like 67% of new images right. With addition of them it managed to get to 93-94 (its harder to get that accuracy) but it had no issue with Stop Sign.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LaNet because we used it class
* What were some problems with the initial architecture?
    - It wasn't very good on its own. It had low accuracy. Image agumentation helped, and using normalization also improved performance. I also didn't like that it used grayscale. Changing this network to use 3 channels was not hard but network seems to small to learn well.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    + I used Vgg-16 but i adjusted it slightly for 32x32x3 images and removed a lot of layers.
* Which parameters were tuned? How were they adjusted and why?
    + Batch size, Learning rate was adjusted to improve learning, also drop rate. Learning rate was adjusted to train sligthly  slower but more accurate. Batch size to maximize my hardware for training and get good accuracy. Drop rate to regularize more network to prevent overffiting. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    + Dropout layer helped to reguralize my network

If a well known architecture was chosen:
* What architecture was chosen?
    + My architecture was slimmed version of VGG-16 with few changes. I tried LaNet but it didn't learn well on my agumented images. It was trained from 0. So 
* Why did you believe it would be relevant to the traffic sign application?
    + It was newer and bigger network.I could simplify it for my needs.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

First image is dark and its upside down.
![alt text][image4] 

Second image is pixelated.
![alt text][image5]

Third image is blured.
![alt text][image6] 

Forth image is too bright and some details are not visible
![alt text][image7] 

Part of fith image is hidden
![alt text][image8]

Sixth image also has part of image hidden
![alt text][image12]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection  | Right-of-way at the next intersection		| 
| Priority road     			         | Priority road     							|
| No entry					             | No entry		    							|
| Traffic signals	      		         | raffic signals				 				|
| Road narrows on the right		         | Road narrows on the right					|
| Stop sign		         | Stop sign						|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.
W

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22th cell of the Ipython notebook.
My predictions are 100% certain. Before adding rotated/flipped images it was 83.33% because it had issues with stop sign.
![alt text][image11] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
