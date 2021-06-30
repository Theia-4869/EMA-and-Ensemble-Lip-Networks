# Boosting Certified $\ell_\infty$-dist Robustness with EMA Method and Ensemble Model

## Introduction

This is the code for [Boosting Certified $\ell_\infty$-dist Robustness with EMA Method and Ensemble Model](). We use the EMA technique and model ensemble method to improve the performance and robustness of our model. We also use $\ell_\infty$-dist neurons to build commonly used CNN architectures. The $\ell_\infty$-dist neurons we use are implemented in [$\ell_\infty$-dist Net](https://github.com/zbh2047/L_inf-dist-net). We achieve state-of-the-art performance on commonly used datasets: **93.14%** certiﬁed accuracy on MNIST under eps = 0.3 and **35.42%** on CIFAR-10 under eps = 8/255. We also use lightweight network $\ell_\infty$-dist LeNet with very few parameters to achieve **33.42%** on CIFAR-10 under eps = 8/255. Our paper is on [arxiv]().

## Dependencies

- torch 1.8.1
- torchvision 0.9.1
- numpy 1.20.2
- matplotlib 3.4.0
- tensorboard

## Getting Started with the Code

### Installation

After cloning this repo into your computer, first run the following command to install the CUDA extension, which can speed up the training procedure considerably.

```
python setup.py install --user
```

### Usage

You can train your $\ell_\infty$-dist nets and test their performance using the command below:
```
python main.py
```

Choose `--model`(MLP, Conv, LeNet, AlexNet, VGGNet) for network architecture, `--dataset`(MNIST, FashionMNIST, CIFAR10, CIFAR100) for dataset, `--predictor-hidden-size` for the hidden size of Predictor, `--loss`(hinge, cross_entropy) for loss function type and `--opt`(adamw, madam) for optimizer type.  

You can also train your ensemble $\ell_\infty$-dist nets and test their performance using the command below:
```
python main_ensemble.py
```

In addition to the above options, you can choose `--model-num` for number of ensemble models.  

In this repo, we provide complete training scripts as well. You can run the scripts directly to reproduce the results on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 datasets in our paper. The scripts are in the `command` folder.  

For example, to reproduce the results of MNIST using a single $\ell_\infty$-dist Net+MLP , simply run

```
bash command/lipnet++_mnist.sh
```

And to reproduce the results of CIFAR-10 using ensemble $\ell_\infty$-dist LeNet+MLP, simply run

```
bash command/liplenet++_ensemble_cifar10.sh
```

## Advanced Training Options

### Multi-GPU Training

We also support multi-GPU training using distributed data parallel. By default the code will use all available GPUs for training. To use a single GPU, add the following parameter `--gpu GPU_ID` where `GPU_ID` is the GPU ID. You can also specify `--world-size`, `--rank` and `--dist-url` for advanced multi-GPU training.

### Saving and Loading

The model is automatically saved when the training procedure finishes. Use `--checkpoint model_file_name.pth` to load a specified model before training. You can use `--start-epoch NUM_EPOCHS` to skip training and only test the model's performance for a pretrained model, where `NUM_EPOCHS` is the number of epochs in total.

### Displaying training curves

By default the code will generate three files named `train.log`, `test.log` and `log.txt` which contain all training logs. If you want to further display training curves, you can add the parameter `--visualize` to show these curves using Tensorboard.

## Contact

Please contact [theia@pku.edu.cn](theia@pku.edu.cn)  if you have any question on our paper or the codes. Enjoy!
