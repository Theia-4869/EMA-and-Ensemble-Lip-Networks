python main.py --model 'MLP(depth=6,width=6)' --dataset CIFAR100 --predictor-hidden-size 0 --loss 'hinge(mix=0.75)' --p-start 8 --p-end 1000 --epochs 0,100,0,750,800 --kappa 1.0 --eps-test 0.03137 --eps-train 0.1568 -b 512 --lr 0.02 --wd 5e-3 --gpu 0 --visualize -p 200