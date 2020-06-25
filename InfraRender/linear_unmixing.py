import numpy as np
from numpy.linalg import norm
from scipy.optimize import *



def FCLS_unmix(A, b, surface=None, lam_atm=0, maxiter=300, ftol=1e-10):
    m = A.shape[0]
    n = A.shape[1]

    # encode binary vectors indicating surface endmembers and atmospheric endmembers
    if surface is None:
        surface = range(n)

    s = np.zeros(n)
    s[surface] = 1

    atm = np.ones(n)
    atm[surface] = 0

    def func(x, A=A, b=b, s_idx=surface, lam_atm=lam_atm):
        return np.sum((A @ x - b) ** 2) + lam_atm / 2 * norm(atm*x, 2) ** 2

    def func_deriv(x, A=A, b=b, s_idx=surface, lam_atm=lam_atm):

        # sum of squares term
        dfdx = 2 * A.T @ A @ x - 2 * A.T @ b

        # atmospheric component L2 norm term
        dfdx += lam_atm * atm * x

        return dfdx


    cons = ({'type': 'eq',
             'fun': lambda x: s.T @ x - 1,
             'jac': lambda x: s},
            {'type': 'ineq',
             'fun': lambda x: s * x,
             'jac': lambda x: np.diag(s)})
    x0 = np.zeros(n)
    x0[surface] = 1/len(surface)
    res = minimize(func, x0, jac=func_deriv, constraints=cons, method="SLSQP",
                   options={'disp': False, 'maxiter': maxiter, 'ftol': ftol})
    return res.x


def LASSO_unmix(A, b, lam=0.01, surface=None, lam_atm=0, maxiter=300, ftol=1e-10):
    m = A.shape[0]
    n = A.shape[1]

    # encode which endmembers are surface rocks and which are atmospheric endmembers
    if surface is None:
        surface = range(n)

    s = np.zeros(n)
    s[surface] = 1

    atm = np.ones(n)
    atm[surface] = 0

    def func(x, A=A, b=b, lam=lam, s_idx=surface, lam_atm=lam_atm):
        return np.sum((A @ x - b) ** 2) + lam*norm(x[s_idx], ord=1) + lam_atm / 2 * norm(atm*x, 2) ** 2

    def func_deriv(x, A=A, b=b, lam=lam, s_idx=surface, lam_atm=lam_atm):
        # sum of squares term
        dfdx = 2 * A.T @ A @ x - 2 * A.T @ b + lam*s*np.sign(x)

        # atmospheric component L2 norm term
        dfdx += lam_atm * atm * x

        return dfdx

    cons = ({'type': 'ineq',
             'fun': lambda x: s * x,
             'jac': lambda x: np.diag(s)})
    x0 = np.zeros(n)
    x0[surface] = 1 / len(surface)
    res = minimize(func, x0, jac=func_deriv, constraints=cons, method="SLSQP",
                   options={'disp': False, 'maxiter': maxiter, 'ftol': ftol})
    return res.x



def inftyNorm_unmix(A, b, lam=1e-6, surface=None, lam_atm=0, maxiter=300, ftol=1e-10):
    m = A.shape[0]
    n = A.shape[1]

    # encode which endmembers are surface rocks (constraints and
    # regularization will not apply to atmospheric endmembers)
    if surface is None:
        surface = range(n)

    s = np.zeros(n)
    s[surface] = 1

    atm = np.ones(n)
    atm[surface] = 0

    temp_loss = np.zeros(n)

    # find the i that minimizes loss when regularizing with 1/x[i]
    # for all i in surface indices
    # return the abundance vector x that results from the arg min i

    def optimization(A=A, b=b, lam=lam, i=0, maxiter=300, ftol=1e-10):

        def func(x, A=A, b=b, lam=lam, idx=i):
            x_i = np.clip(x[idx], 1e-10, None)
            return np.sum((A @ x - b) ** 2) + lam * 1/x_i + lam_atm / 2 * norm(atm*x, 2) ** 2

        def func_deriv(x, A=A, b=b, lam=lam, idx=i, lam_atm=lam_atm):
            # sum of squares term
            dfdx = 2 * A.T @ A @ x - 2 * A.T @ b

            x_i = np.clip(x[i], 1e-10, None)
            dfdx[idx] -= lam * 1 / (x_i * x_i)

            # atmospheric component L2 norm term
            dfdx += lam_atm * atm * x

            return dfdx

        cons = ({'type': 'eq',
                 'fun': lambda x: s.T @ x - 1,
                 'jac': lambda x: s},
                {'type': 'ineq',
                 'fun': lambda x: s * x,
                 'jac': lambda x: np.diag(s)})
        x0 = np.zeros(n)
        x0[surface] = 1 / len(surface)
        res = minimize(func, x0, jac=func_deriv, constraints=cons, method="SLSQP",
                       options={'disp': False, 'maxiter': maxiter, 'ftol': ftol})

        return res

    for i in surface:

        res = optimization(i=i)
        temp_loss[i] = res.fun


    i_min = temp_loss.argmin()
    res = optimization(i=i_min)

    return res.x


def pNorm_unmix(A, b, lam=0.01, p=0.8, surface=None, lam_atm=0, maxiter=300, ftol=1e-10):
    # setup problem
    m = A.shape[0]
    n = A.shape[1]

    # encode which endmembers are surface rocks (constraints and
    # regularization will not apply to atmospheric endmembers)
    if surface is None:
        surface = range(n)

    s = np.zeros(n)
    s[surface] = 1

    atm = np.ones(n)
    atm[surface] = 0


    def func(x, A=A, b=b, lam=lam, p=p, lam_atm=lam_atm):
        return np.sum((A @ x - b) ** 2) + lam * norm(s*x, ord=p) + lam_atm / 2 * norm(atm*x, 2) ** 2

    def func_deriv(x, A=A, b=b, lam=lam, p=p, lam_atm=lam_atm):
        # sum of squares term
        dfdx = 2 * A.T @ A @ x - 2 * A.T @ b

        # p norm term
        inv_pnorm = 1 / norm(x, ord=p)
        for i in surface:
            x_i = np.clip(x[i], 1e-13, None)
            dfdx[i] += lam * np.sign(x_i) * (np.abs(x_i) * inv_pnorm) ** (p - 1)

        # atmospheric component L2 norm term
        dfdx += lam_atm * atm * x

        return dfdx

    cons = ({'type': 'eq',
             'fun': lambda x: s.T @ x - 1,
             'jac': lambda x: s},
            {'type': 'ineq',
             'fun': lambda x: s * x,
             'jac': lambda x: np.diag(s)})
    x0 = np.zeros(n)
    x0[surface] = 1 / len(surface)
    res = minimize(func, x0, jac=func_deriv, constraints=cons,
                   method="SLSQP", options={'disp': False, 'maxiter': maxiter, 'ftol': ftol})
    return res.x



