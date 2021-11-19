import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([0, 3, 1, 5])
c = np.absolute(a - b)
print(a - b)

a = np.array([[1, 3], [2, 4]])
b = np.array([0, 1])
mean = a.mean(axis=1, keepdims=True)
std = a.std(axis=1, keepdims=True)
print(mean)
print(std)

print((a - mean) / std)

a = np.array([[1, 2], [3, 4]])
b = np.array([[2], [4]])

print(a / b)

a = np.array([[2.7810836, 2.550537003]])
mean = a.mean(axis=1, keepdims=True)
std = a.std(axis=1, keepdims=True)
print(mean)
print(std)
print((a - mean) / std)
