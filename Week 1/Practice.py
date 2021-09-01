import math

import numpy as np
import pandas as pd


def numpy_practice():
    ones = np.ones(4)
    zeros = np.zeros((1, 2), dtype=float)
    print('Ones= ', ones, '\nZeros= ', zeros)
    print(type(ones))

    np.random.seed(0)
    A = np.random.rand(2, 3)
    B = np.random.rand(2, 3)
    np.random.seed(0)
    C = np.random.rand(2, 4)
    with np.printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}):
        print('A=', A, '\nB=', B)
        print('C=', C)

    X = np.array([[1, 2], [3, 4]])
    Y = [[1, 2], [1, 3]]
    print('X elem Y= ', np.multiply(X, Y))  # element-wise multiplication
    print('X*Y= ', np.matmul(X, Y))  # matrix multiplication
    print('X.Y= ', np.dot(X, Y))
    print('X cross Y= ', np.cross(X, Y))
    print('Inverse= ', np.linalg.inv(X))
    print('Transpose= ', np.transpose(X))
    print('Determinant= ', np.linalg.det(X))


# Pandas Practice!
def pandas_practice():
    df = pd.DataFrame({"A": ['14', '4', math.nan, '4', '1'],
                       "B": [5, 2, 54, 3, 2],
                       "C": [20, 20, math.nan, 3, 8],
                       "D": [14, 3, 6, 2, 6]})
    mode = df['A'].mode()[0]
    data = df
    # print(mode, type(mode))
    data["C"] = df["C"].fillna(mode)
    print(data)

    df = pd.DataFrame({"A": [14, 4, math.nan, 4, 1],
                       "B": [5, 2, 54, 3, 2],
                       "C": [20, 20, math.nan, 3, 8],
                       "D": [14, 3, 6, 2, 6]})
    # new_df = df[df["B"] > 3]
    # print(new_df)
    g = df.groupby("B")
    print(g["A"].mean())


def compute_cofactor(matrix):
    cofactor_matrix = np.zeros(matrix.shape)
    rows, cols = cofactor_matrix.shape
    for row in range(rows):
        for col in range(cols):
            minor = matrix[np.array(list(range(row)) + list(range(row + 1, rows)))[:, np.newaxis],
                           np.array(list(range(col)) + list(range(col + 1, cols)))]
            cofactor_matrix[row, col] = (-1) ** (row + col) * np.linalg.det(minor)
    return cofactor_matrix


M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

if __name__ == '__main__':
    # numpy_practice()
    pandas_practice()
    # ans = compute_cofactor(M)
    # print(ans)
    try:
        x1 = np.array([[1, 2]])
        x2 = np.array([[1], [2], [3]])
        print(np.matmul(x1, x2))
        x = 2 / 0
    except np.linalg.LinAlgError:
        print("Singular!")
    except ZeroDivisionError:
        print('Divide by 0')
    except ValueError:
        print('Dimension mismatch')
    pass

# # Other stuff
# A = [[1, 1], [0, 1]]
# try:
#     print(np.linalg.inv(A))
#     x = 2 / 0
# except np.linalg.LinAlgError:
#     print("Singular!")
# except ZeroDivisionError:
#     print('Divide by 0')
