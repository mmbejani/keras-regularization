# Deep Neural Network Regularization

This library consist of some of the newset approach to regulariztion the neural network.
As we mention, the library is developed based on **Keras** library. The version 1.0.0 can be used on just CNNs!
We are working to extend the regularization to other type of networks (Recurrent networks e.g. LSTM).

The library has two major regularization categories.

* Regularization Layer
    * Adaptive Dropout
    * Bridgeout
    * Dropconnect
    * Shakeout
    * Spectral Dropout
* Regularization based on Loss Function
    * Adaptive Spectral
    * Global Weight Decay

If you propose a new approach or find a suitable regularization scheme, contact with us.

## How to use
### Installation

first download the package with following command:
```
git clone https://github.com/mmbejani/keras-regularization
```
then change directory to the target folder and run following command:
```
python setup.py install [--user]
```
The `--user` command is optional.

### Okey! let see some examples
We bring two example of the usage this lib. First, we show how to use the implicit regularization scheme.
....

# References
[1] Bejani, M.M. and Ghatee, M., 2019. Convolutional neural network with adaptive regularization to classify driving styles on smartphones. IEEE Transactions on Intelligent Transportation Systems.

[2] Khan, N., Shah, J. and Stavness, I., 2018. Bridgeout: stochastic bridge regularization for deep neural networks. IEEE Access, 6, pp.42961-42970.

[3] Wan, L., Zeiler, M., Zhang, S., Le Cun, Y. and Fergus, R., 2013, February. Regularization of neural networks using dropconnect. In International conference on machine learning (pp. 1058-1066).

[4] Kang, G., Li, J. and Tao, D., 2016, February. Shakeout: A new regularized deep neural network training scheme. In Thirtieth AAAI Conference on Artificial Intelligence.

[5] Khan, S.H., Hayat, M. and Porikli, F., 2019. Regularization of deep neural networks with spectral dropout. Neural Networks, 110, pp.82-90.
