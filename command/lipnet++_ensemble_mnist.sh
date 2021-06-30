python main_ensemble.py --model 'MLPFeature(depth=4,width=5)' --dataset MNIST --predictor-hidden-size 256 --loss 'cross_entropy' --model-num 5 --p-start 8 --p-end 1000 --epochs 0,50,50,350,400 --kappa 0.5 --eps-test 0.3 --eps-train 0.35 -b 512 --lr 0.02 --wd 5e-3 --gpu 0 --visualize -p 200