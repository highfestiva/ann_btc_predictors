# Neural Stock Market Prediction

This is a test of three different types of networks to predict the future price of BitCoin. One feed forward, one convolutional and one LSTM.

I'm learning still learning ANN's, so the code is copy-pasted from here and there, but I especially want to thank
[Philip Xu](https://github.com/philipxjm) for his excellent CNN-stock-sample
[here](https://github.com/philipxjm/Convolutional-Neural-Stock-Market-Technical-Analyser).


## Requirements

```pip install tensorflow pandas numpy sklearn keras matplotlib```


## Speed it up

If you want ANN training to run acceptably fast, you need to install CUDA 9.0 and cuDNN 7.1 first and then:

```pip install tensorflow-gpu```
