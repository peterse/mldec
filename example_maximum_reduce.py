import numpy as np

# Simple 2D arrays as stand-ins for y_dist, x_dist, t_dist
y_dist = np.array([[1, 2, 3],
                   [4, 5, 6]])

x_dist = np.array([[7, 2, 1],
                   [3, 8, 4]])

t_dist = np.array([[2, 9, 3],
                   [1, 6, 7]])

print("y_dist:")
print(y_dist)
print("\nx_dist:")
print(x_dist)
print("\nt_dist:")
print(t_dist)

# Using np.maximum.reduce
result = np.maximum.reduce([y_dist, x_dist, t_dist])
print("\nResult of np.maximum.reduce([y_dist, x_dist, t_dist]):")
print(result)

# This is equivalent to:
print("\nManual calculation (element-wise maximum):")
manual_result = np.maximum(np.maximum(y_dist, x_dist), t_dist)
print(manual_result)

# Let's break it down step by step:
print("\nStep-by-step breakdown:")
print("Step 1: np.maximum(y_dist, x_dist)")
step1 = np.maximum(y_dist, x_dist)
print(step1)

print("\nStep 2: np.maximum(step1, t_dist)")
step2 = np.maximum(step1, t_dist)
print(step2)

print(f"\nAll results are equal: {np.array_equal(result, manual_result) and np.array_equal(result, step2)}") 