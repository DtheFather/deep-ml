import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    A = np.array(a)
    B = np.array(b)
    if len(A[0]) == len(B) :
        return A @ b
    else:
        return -1