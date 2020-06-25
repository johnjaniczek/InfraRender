import numpy as np
import cvxpy as cp

def main():
    m, n, k = 4, 3, 2
    A = np.random.rand(m, n)
    B = np.random.rand(m, k)
    coeff = DFFweights(A, B)
    print(coeff)

def DFFweights(A, B):
    """
    This function minimizes (total variance(B/mean(A*x)) where A.shape (m x n), X.shape (n x k), and B.shape (m x k)
    where m = image flattened in row-major order, n = number of EigenFlats, and k = observations.
    A is the array of EigenFlats, X is the coefficients for each EigenFlat, B is the observation to correct.
    """
    k = B.shape[1]
    m = B.shape[0]
    n = A.shape[1]

    # Preallocate the empty arrays
    coeff = np.arange(0, n * k, 1, dtype='f8').reshape(n, k) * 0

    b = cp.Parameter(m, nonneg=True)
    x = cp.Variable(n, nonneg=True)

    totalVariance = cp.tv((b*cp.inv_pos(cp.sum(A * x))))
    objective = cp.Minimize(totalVariance)
    prob = cp.Problem(objective)

    # Loop through all spectra and find optimal solutions with the warm start
    for ind in range(0, k):
        b.value = B[:, ind]
        loss = prob.solve()
        coeff[:, ind] = x.value

    return coeff

if __name__ == '__main__':
    main()
