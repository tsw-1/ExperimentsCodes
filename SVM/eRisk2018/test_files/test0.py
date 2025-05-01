import numpy as np

# 创建一个包含零的NumPy数组
arr = np.array([1.0, 2.0, 0.0, 4.0])

# 尝试除以零，将产生特殊的浮点数值
result = arr / 0

print(result)
