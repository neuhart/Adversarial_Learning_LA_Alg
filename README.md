# Adversarial Learning with Lookahead
This is the repository used for the code of my master's thesis "Adversarial Learning with Lookahead". <br>

## Scope of the Thesis
### Adversarial Examples:
Adversarial Examples are deliberately perturbed versions of inputs crafted by adversaries with the goal to fool the classifier under attack. Example: 
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
Numerical Experiments via Adversarial Training on three different standard machine learing datasets, MNIST, Fashion MNIST and CIFAR-10. The performance of several different optimizers was compared: SGD, Adam, ExtraSGD, ExtraAdam, OGDA as well as five instances of Lookahead each with one of the before mentioned optimizers as its inner optimizer. 

### Hyperparameter Tuning
- The hyperparameter for each optimizer were tuned in  <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/param_tuning.py">param_tuning.py</a>
- The results (training loss & validation accuracy) were compared in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Hyperparam_tuning/hyper_param_tuning_eval.py">hyper_param_tuning_eval.py</a>

### Adversarial Training
- Adversarial training for each dataset for each optimizer was done in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/main_file.py">main_file.py</a>
- The actual training loop and model validation were outsourced to <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/training.py">training.py</a> and <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/evaluation.py">evaluation.py</a> , respectively.
- The <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Utils">Utils</a> folder contains the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Utils/data_utils.py">data_utils.py</a> file used to load and transform the datasets, and 
the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/Utils/project_utils.py">project_utils.oy</a> file where helper functions are stored to ensure better readability of the main code.

### Optimizer
The code for the optimizers used for the experiments can be found in the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Optimizer">Optimizer</a> folder.

### Models 
The neural networks used for the experiments can be found in the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/Models">Models</a> folder together with the required data transformations for each model.

### Results
- The results of the experiments (training loss, validation accuracy) are stord in the <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/tree/main/results"><results/a> folder
- The code for visualizations of the results of the experiments can be found in <a href="https://github.com/neuhart/Adversarial_Learning_LA_Alg/blob/main/result_eval.py"></a>

<a href="url"></a>
