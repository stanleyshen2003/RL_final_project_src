import numpy as np

array = np.random.rand(3,5,5)
print(array)
array = np.transpose(array, (1, 2, 0))
print(array)