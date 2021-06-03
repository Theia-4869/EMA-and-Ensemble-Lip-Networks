python main.py --model 'ResNetFeature' --dataset CIFAR10 --predictor-hidden-size 512 --loss 'cross_entropy' --p-start 8 --p-end 1000 --epochs 0,100,100,750,800 --kappa 0.95 --eps-test 0.03137 --eps-train 0.03451 -b 128 --lr 0.005 --wd 5e-3 --gpu 0 --visualize -p 200