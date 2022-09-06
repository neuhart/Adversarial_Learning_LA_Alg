# Adversarial Learning with Lookahead
This is the repository used for my master's thesis "Adversarial Learning with Lookahead". <br>

## Scope of the Thesis
### Adversarial Examples:
Adversarial Examples are deliberately perturbed versions of inputs crafted by adversaries with the goal of fooling the classifier under attack. Example:<br> 
<img src="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Images/MNIST_5_plots_miscl_as_2_withLabels.png" height="250" width="500" align="center"> <br>
The image on the left depicting the number ’5’ is unperturbed and is classified correctly by a neural network classifier. Using the PGD attack, the classifier is
misled into predicting the number ’2’ instead. On the other hand, most humans would still be able to recognize the perturbed version as a ’5’.
### Adversarial Training:
Adversarial training is a defense strategy in the domain of adversarial learning in which the defender incorporates adversarial
examples into the learning process of the model by either re-training the model on the adversarial examples or training the model solely on them in the first place. 
<br>
### Lookahead
The Lookahead optimizer proposed by <a href="https://www.cs.toronto.edu/~hinton/absps/lookahead.pdf">Zhang et al.</a> is an optimization algorithm that
has been shown to be able to outperform standard optimization algorithms like SGD
or Adam in machine learning tasks. Intuitively, the optimizer is using a set of weights
obtained from another optimization algorithm, called the fast weights, to ’look ahead’
k steps and then taking a more conservative step in that direction, yielding a second
set of weights, called the ’slow’ weights. In its inner loop, i.e., when looking ahead, any
standard optimizer, including SGD and Adam, can be used. Furthermore, Zhang et al.
reported that the Lookahead algorithm is less dependent on optimal hyperparameters
and therefore requires less hyperparameter tuning in comparison to other optimization
algorithms used in the domain of machine learning.

## Experiments
The Experiments were organized as follows: <br>
Numerical Experiments via Adversarial Training on three different standard machine learing datasets, MNIST, Fashion MNIST and CIFAR-10. The performance of several different optimizers was compared: SGD, Adam, ExtraSGD, ExtraAdam, OGD as well as five instances of Lookahead each equipped with one of the before mentioned optimizers as its inner optimizer.
### Datasets 
- <a href="https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST">MNIST</a>
- <a href="https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST">FashionMNIST</a>
- <a href="https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10">CIFAR-10</a>

### Optimizer
The following optimizers were used for the experiments:
- SGD & Adam (<a href="https://pytorch.org/docs/stable/optim.html">Pytorch</a>)
- OGD (<a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Optimizer/OGD.py">OGD.py</a>) 
- ExtraSGD & ExtraAdam (<a href="https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py">Hugo Berard, Facebook</a>) 
- Lookahead (<a href="https://github.com/michaelrzhang/lookahead">Zhang et al.</a>)

### Models 
The neural networks used for the experiments:
- CIFAR-10:  <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18">Resnet-18</a>
- MNIST & FashionMNIST: <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Models">MNIST CNN</a>
<br>
<img src="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Images/MNIST_CNN.png" height="250" width="600" align="center"> <br>

### Attacks
For Adversarial Training, I used the PGD attack, and for model evaluation I used both the FGSM and PGD attack (<a href="https://github.com/cleverhans-lab/cleverhans">cleverhans</a>)

### Hyperparameter Tuning
- The hyperparameters for each optimizer were tuned in  <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/param_tuning.py">param_tuning.py</a>
- The analysis of the results (training loss & validation accuracy) can be found in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Hyperparam_tuning">Hyperparameter Tuning</a>. <br>
<img src="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Hyperparam_tuning/Analysis/CIFAR10/LR_robustness/adv_pgd_valid_results_mean_std/ExtraSGD.png" height="350" width="1000" align="center"> <br>

### Slow weights vs Fast weights
- In <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/slow_weight_evaluation">slow_weight_evaluation</a>, a comparison of the performance of the slow weight and fast weights for LA is provided 
- Corresponding Code: <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/slow_weights.py">slow_weights.py</a>. <br>
<img src="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/slow_weight_evaluation/Analysis/CIFAR10/adv_pgd_valid_slow/LA-OGD.png" height="350" width="1000" align="center"> <br>


### Adversarial Training
- Adversarial training for each dataset and for each optimizer was done in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/main_file.py">main_file.py</a>
- The actual training loop and model validation were outsourced to <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/training.py">training.py</a> and <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/evaluation.py">evaluation.py</a> , respectively.
- The <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Utils">Utils</a> folder contains the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Utils/data_utils.py">data_utils.py</a> file used to load and transform the datasets, and 
the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Utils/project_utils.py">project_utils.py</a> file where helper functions are stored to ensure better readability of the main code, as well as the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Utils/visualization_utils.py.py">visualization_utils.py.py</a> file where some helper functions for the purpose of visualizing the results are stored.
- The results can be found in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/results">results</a> <br>
- <img src="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/results/Analysis/FashionMNIST/ExtraSGD.png" height="350" width="450" align="center"> <br>
- results files with the prefix "adv" contain data about adversarially trained networks, e.g.:<a href="adv_fgsm_valid_results">adv_fgsm_valid_results</a>
- file names containing 'train' correspond to training loss data (per epoch, sum of training loss for whole training set), e.g.: <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/results/CIFAR10/adv_train_results">adv_train_results</a>
- file names containing 'valid' correspond to validation accuracy data (per epoch, ratio of correct predictions and number of test inputs), e.g.: <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/results/CIFAR10/adv_valid_results">adv_valid_results</a>
- validation files with the words 'fgsm' or 'pgd' in their names contain evluations on adversarial examples generated with the fgsm or pgd attack, respectively), e.g. <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/results/CIFAR10/adv_pgd_valid_results">adv_pgd_valid_results</a>
- The code for visualizations of the results of the experiments can be found in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/result_eval.py">result_eval.py</a>

<a href="url"></a>
