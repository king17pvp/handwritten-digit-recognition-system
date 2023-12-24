# handwritten-digit-recognition-system
Handwritten digits recognition system using K-nearest neighbors, Multilayer Perceptron, and Convolutional neural network.

# Features
The app will allow you to draw digits on the canvas using the computer's mouse. Then you will have three options to predict your drawn digits.

There will be 5 buttons. In which three buttons to predict digits using three different approaches. One button to clear canvas, and one to show the bounding box of each digit.

# How to use: 
To use this project using git, please do the following.
First, clone the repository's main branch into your desired directory using your git command prompt.

```git clone -b main https://github.com/king17pvp/handwritten-digit-recognition-system.git```

Secondly, you can access the directory by this command.

```cd handwritten-digit-recognition-system```

Thirdly, install required libraries via requirement.txt

```pip install -q -r requirement.txt```

Finally, run the project by 

```python main.py```


This project will have a simple UI with 3 ways of predicting numbers. 
# K-nearest neighbors
We will implement with Gaussian weight and sigma value = 2 with K = 7.

# Mutlilayer Perceptron
We will implement with 3 hidden layers of the size of each hidden layer respectively, is 512, 256, 128.

# Convolutional Neural network

Implement using 2 convolutional-subsampling layers, 24 filters in the first layer, 48 filters in second layer. With one hidden fully connected layer of size 256 and output layer size 10.


