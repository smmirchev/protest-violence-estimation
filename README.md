# protest-violence-estimation
The project software aims to calculate the violence factor in protests. 

## Getting Started
Steps to estimate the violence in images.
1. Download the protest-violence-estimation repository
2. Open the **predict.py** file
3. Replace imagePath with the desired image
4. Specify whether the image has any of the ten categories in **num_vector**. The categories are ["sign", "photo", "fire", "police", "children", "group_20", "group_100", "flag", "night", "shouting"]. 

### Prerequisites
The following python libraries must be installed to run the project.
* Python 3.6
* flask 1.1.1
* keras 2.3.1
* numpy 1.18.1
* tensorflow 2.1.0
* setuptools 45.2.0
* matplotlib 3.2.1
* pandas 1.0.3

### Overview

The project is described in [Original Paper](https://arxiv.org/abs/1709.06204) - [GitHub repository](https://github.com/wondonghyeon/protest-detection-violence-estimation)
This project aims to solve a regression problem unlike classification in the [other project](https://github.com/smmirchev/ImageNet-Object-Identification). Regression in machine learning is supervised learning – an algorithm attempts to produce a continues output (quantities, such as size, length and amount etc) or a probability which is depended on the input variables. The input variables for this model are sign, photo, fire, police, children, group of 20, group of 100, flag, nigh and shouting. Based on these 10 parameters the violence factor in the images can be calculated. 
However, this project's neural network is a multi-head model which can accept multiple inputs and produce multiple outputs.
Results - Mean Squared Error: 0.0038.

## Author
* Stefan Mirchev

## References
* [ImageNet Dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge)
* [Protest Project Original Paper](https://arxiv.org/abs/1709.06204)
* [Protest GitHub Repository](https://github.com/wondonghyeon/protest-detection-violence-estimation)
