# XOR-Nerual-Network
this repo is for implementing an xor nerual network using pytorch.


to install dependencies please run 
```
pip install -r requirements.txt
```


### XOR Model using Neural Network
What does it mean?

In a feed-forward neural network, the functions $f_i \forall i \in [n]$ takes specific forms. All this is abstracted away in pytorch. so using pytorch i implemented a neural netwerk non linear model as a function approximater(1-layer feedforwared neural network with 2 neurons). 
  
  #### Why choosing Binary cross entropy
1. [Cross-entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html), or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.the loss function should return high values for bad predictions and low values for good predictions. In binary classification like our case, where the number of classes M equals 2, cross-entropy can be calculated as:
> −(ylog(p)+(1−y)log(1−p))

 It can be used in classification, in binary classification.
####  Augment the training data. We have only four values. But that is not enough to train the network effectively. How do we do that?

Augemnting the data can be done by including the points that are closely located to the four points of the two classes [(1,0), (0,1)] and [(0,0), (1,1)] 
for example f(0.93,.01) = 1, f(0.032,.01) = 0

#### Why do we need a neural network ? Why not just use a linear classifier?
if we visualize the The XOR function as in the below figure we would find that we can not use a linear classifier to seperate the two domains of **Y** (class A, Class B). So the function is non linear in its nature, that is why a neural network is preferable in this case.

![XOR plot](http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/nn04_mlp_xor/nn04_mlp_xor_04.png)
