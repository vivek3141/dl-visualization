# Chapter 2

This folder contains all the code for chapter 2 of the series:

* `transform.py` contains the code for the 2-100-2-5 transformation and the probability surfaces
* `anims.py` contains the rest of the animations from the video
* `gen_*.py` contains the code used to generate the images for the decision boundaries and the point higlighting mask
* `train_*.py` contains the code for training the models used in the animations
 - `model1` is the 2-100-2-5 NN trained using ReLUs
 - `model2` is the 2-100-2-5 NN trained using sin(x) as the hidden activation function
 - `model3` is a 2-100-100-5 NN used for generating the decision boundaries for the extrapolation example
* `utils.py` contains helper functions used for training