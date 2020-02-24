# Deep Neural Network Regularization

This library consist of some of the newset approach to regulariztion the neural network.
The first version of this library is consist the following regularization scheme:

1. LRFout
2. Adaptive Spectral Regularization
3. Adaptive Tikhonov Spectral Regularization
4. Spectral Dropout
5. Adaptive Dropout

As we mention, the library is developed based on **Keras** library. The version 1.0.0 can be used on just CNNs!
We are working to extend the regularization to other type of networks (Recurrent networks).

The library has three major regularization categories.

* Regularization based on Callbacks
    * Adaptive Spectral
    * Adaptive Spectral with Condition Number
    * LRFout
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

To install this library based on `pip` you can use the following command:
```
pip install keras-regularization
```
or to install based on `git`, first download the package with following command:
```
git clone url.git
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
[1] Mohammad Mahdi Bejani and Mehdi Ghatee, Adaptive

[2] Mohammad Mahdi Bejani and Mehdi Ghatee, Adaptive

[3] Mohammad Mahdi Bejani and Mehdi Ghatee, Adaptive

[4] Mohammad Mahdi Bejani and Mehdi Ghatee, Adaptive