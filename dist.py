'''
Todo:
1. 加速距离矩阵计算
2. 在测试集上做相同操作
3. 在fashionmnist, cifar10上做相同操作
3. 把结果发到群里
'''

import numpy as np
from fastdist import fastdist
from torchvision import datasets, transforms

transform = transforms.ToTensor()
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)
data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)
data = data_train.data.numpy().astype(float)
label = data_train.targets.numpy()
size = data.shape[0]  # 数据集大小
flatdata = data.reshape(size, -1)
res = fastdist.matrix_to_matrix_distance(
    flatdata, flatdata, fastdist.chebyshev, "chebyshev")
# 计算l无穷距离矩阵

dist = np.zeros((10, 10), dtype=int)+255
for i in range(size):
    for j in range(i+1, size):
        dist[label[i], label[j]] = min(
            dist[label[i], label[j]], res[i, j])
print(dist)
