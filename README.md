Download Link: https://assignmentchef.com/product/solved-ee569-homework-5-cnn-training-on-lenet-5
<br>
<h1>Problem 1: CNN Training on LeNet-5</h1>

In this problem, you will learn to train a simple convolutional neural network (CNN) called the LeNet-5, introduced by LeCun et al. [1], and apply it to the CIFAR-10 dataset [2]. LeNet-5 is designed for handwritten and machine-printed character recognition. Its architecture is shown in Fig. 1. This network has two <em>conv</em> layers, and three <em>fc</em> layers. Each conv layer is followed by a <em>max pooling</em> layer. Both <em>conv</em> layers accept an input receptive field of spatial size <em>5×5</em>. The filter numbers of the first and the second <em>conv</em> layers are 6 and 16 respectively. The stride parameter is 1 and no padding is used. The two <em>max pooling</em> layers take an input window size of <em>2×2</em>, reduce the window size to <em>1×1</em> by choosing the maximum value of the four responses. The first two <em>fc</em> layers have 120 and 84 filters, respectively. The last <em>fc</em> layer, the output layer, has size of 10 to match the number of object classes in the CIFAR-10 dataset. Use the popular ReLU activation function [3] for all <em>conv</em> and all <em>fc</em> layers except for the output layer, which uses softmax [4] to compute the probabilities.




<strong>Figure 1:</strong> A CNN architecture derived from LeNet-5

The CIFAR-10 dataset consists of 60,000 RGB 32×32 pixel images in 10 classes (with 6000 images per class). It includes a labeled training set of 50,000 images and a test set of 10,000 images. Fig. 2 shows some exemplary images from the CIFAR-10 dataset.




<strong>Figure 2: </strong>CIFAR-10 images

<h2>(a) CNN Architecture (Basic: 20%)</h2>

Explain the architecture and operational mechanism of convolutional neural networks by performing the following tasks.

<ol>

 <li>Describe CNN components in your own words: 1) the fully connected layer, 2) the convolutional layer, 3) the max pooling layer, 4) the activation function, and 5) the softmax function. What are the functions of these components?</li>

 <li>What is the over-fitting issue in model learning? Explain any technique that has been used in CNN training to avoid the over-fitting.</li>

 <li>Why CNNs work much better than other traditional methods in many computer vision problems? You can use the image classification problem as an example to elaborate your points.</li>

 <li>Explain the loss function and the classical backpropagation (BP) optimization procedure to train such a convolutional neural network.</li>

</ol>

Show your understanding as much as possible in your own words in your report.

<h2>(b) CIFAR-10 Classification</h2>

Train the CNN given in Fig. 1 using the 50,000 training images from the CIFAR-10 dataset. You can adopt proper preprocessing techniques and the random network initialization to make your training work easy.

<ol>

 <li>Compute the accuracy performance curves using the epoch-accuracy (or iteration-accuracy) plot on training and test datasets separately. Plot the performance curves under 5 different yet representative initial parameter settings (filter weights, learning rate, decay and etc.). Discuss your observations and the effect of different settings.</li>

 <li>Find the best parameter setting to achieve the highest accuracy on the test set. Then, plot the performance curves for the test set and the training set under this setting.</li>

</ol>

<h2>(c) State-of-the-Art CIFAR-10 Classification</h2>

Check the state-of-art implementation on CIFAR-10 classification in [5]. Select one paper from the list for discussion.

<ol>

 <li>Describe what the authors did to achieve such a result. You do not have to implement the network.</li>

 <li>Compare the solution with LeNet-5 and discuss pros and cons of the two methods.</li>

</ol>

You can add pictures, flowcharts, and diagrams in your report. If you do so, you need to cite their sources.




<h1>Problem 2: EE569 Competition — CIFAR10 Classification</h1>

Feel free in modifying the baseline CNN in Fig. 1 to improve the classification accuracy obtained in Problem 1(b). For example, you can increase the depth of the network by adding more layers, or/and change the number of filters in some layers. You can augment the dataset. You can also try different activation functions or optimization algorithms. They all have a potential to improve the result. You may need to fine-tune the training parameters to get the training job done.

Your score will be determined by three aspects:

<ul>

 <li>Motivation and logics behind your design (25%): You can draw the diagram of your network architecture and explain in your own words. Describe the training parameter setting to reach the result below. Discuss the sources of performance improvement compared with Problem 1(b).</li>

 <li>Classification accuracy (15%): Report the best accuracy that you can achieve; report the training time and inference time; draw the train and test accuracy performance curves using the epochaccuracy (or iteration-accuracy) plot; draw the test accuracy curve by randomly drop training samples to see how the performance degrades.</li>

 <li>Model size (10%): You have limited resources at hand, such as GPU, we don’t want you to waste time on long and complex training process or using pre-trained models on GitHub. For example, using ResNet in this homework is meaningless. Therefore, you are required to compute the model size or parameter numbers that you use, which helps to release your stress on obtaining the accuracy as high as possible.</li>

</ul>

Your grading in this part will be based on your obtained performance in both accuracy and model size in comparison with other students in the same class.




<strong>WARNING</strong>: You can borrow the ideas from other papers. But you need to state clearly how you construct the baseline and why you do that. Don’t forget to cite their sources. Code copying and other types of plagiarism is strictly forbidden. Don’t waste your time on playing tricks such as training on test data. To be fair for other students, we will double check your code. Please make sure to submit a single python file that is executive.