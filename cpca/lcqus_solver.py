'''
Created on Nov 18, 2013

@author: doglic
'''

import numpy as np
import scipy.optimize as scopt
import sys

sys.path.append('./../common')
import utils

THRESHOLD = 0.1
PRECISION = 0.000001

'''
    READ ME:
    
    Optimization of a quadratic (with an arbitrary Hessian) over the r-hyperellipsoid and subject to
    a set of linear constraints. The problem is non-convex and we give a close form solution of 
    complexity O(n^3).
    
    Please call the method `maximizer' and see instructions on ordering of parameters at the 
    header of the method.
'''


'''
    Input matrix and hyperellipsoid constraint matrix need to be symmetric. This
    method is used for validation of these matrices (when passed).
'''
def __is_symmetric(W):
    if W.shape[0] != W.shape[1]:
        return False
    
    for i in range(W.shape[0]):
        for j in range(i):
            if np.abs(W[i][j] - W[j][i]) > PRECISION:
                return False
    
    return True


'''
    Method for validation of all input parameters:
    i)     matrices W and K need to be symmetric matrices;
    ii)    dimension of the linear term in the optimization objective must be in
           accordance with the dimension of the matrix W (forming the quadratic term);
    iii)   number of rows in L must be less than the number of rows in W (otherwise, the 
           solution is already defined by constraints or there is no feasible one;
    iv)    constraint value vector y must be in accordance with matrix L.
'''
def __validate_parameters(W, r, b, K, L, y):
    if not __is_symmetric(W) or (None != K and not __is_symmetric(K)):
        raise ValueError('The matrix W and the matrix K, if passed, must be symmetric!')
    
    if None != b and b.shape[0] != W.shape[0]:
        raise ValueError('The b vector is of inappropriate dimension. Its expected dimension is ' + str(W.shape[0]) + '!')
    
    if None != L and (L.shape[1] != W.shape[1] or L.shape[0] >= W.shape[0]):
        raise ValueError('The constraint matrix L is of inappropriate dimension. Its expected dimension is m x ' + str(W.shape[1]) + ' with m < ' + str(W.shape[0]) + '!')
    
    if None != y and (L == None or y.shape[0] != L.shape[0]):
        raise ValueError('The constraint value vector is of invalid dimension or it is specified without the constraint matrix L!')
    
    if r <= 0:
        raise ValueError('The radius must be positive!')


'''
    Return the eigenvector corresponding to the largest eigenvalue of the matrix W.
'''
def __top_eigenvector(W):
    eig_vals, eig_vecs = np.linalg.eigh(W)
    U = np.asmatrix(eig_vecs)
    return np.asarray(U[:, np.argmax(eig_vals)]).reshape(-1)


'''
    Reformulate the optimization problem by introducing a substitution K^{1/2} x = s.
'''
def __from_ellipsoid_to_sphere(W, L, b, K, pre_cmp_ell2sph_trans=None):
    if None == K:
        return (W, L, b, None)
    
    if None == pre_cmp_ell2sph_trans:
        inv_transform = utils.sqrt_inverse_sym_pd_mat(K)
    else: 
        inv_transform = pre_cmp_ell2sph_trans
        
    sph_W = np.dot(np.dot(inv_transform, W), inv_transform)
    sph_b = np.dot(inv_transform, b.reshape(-1, 1)) if None != b else None    
    sph_L = np.dot(L, inv_transform) if None != L else None
    
    return (sph_W, sph_L, sph_b, inv_transform)


'''
    Transform the solution from the ellipsoidal domain to the spherical one.
'''
def __from_sphere_to_ellipsoid(sph_optim, inv_transform):
    if None != inv_transform:
        return np.asarray(np.dot(inv_transform, sph_optim.reshape(-1, 1))).reshape(-1)
    return sph_optim
    

'''
    Reformulate the optimization problem (the one we get after __from_ellipsoid_to_sphere) using substitution
    Q^T s = t, where Q comes from QR factorization of the matrix L^T.
        
    Remark: We append with zeros the transpose of the constraint matrix as `numpy' computes the QR in a `correct' form only
    for the square matrices. 
'''
def __qr_transform(W, b, r, L, y):
    m = L.shape[0]
    
    qr_res = utils.rect_mat_qr(L)
    Q = qr_res[0]
    R = qr_res[1]    
    trans_Q = np.transpose(Q)
    P = np.transpose(R)[:m, :m]
    
    ort_map_W = np.dot(np.dot(trans_Q, W), Q)
    C = ort_map_W[m:, m:]
    u = np.linalg.solve(P, y.reshape(-1, 1))
    f = np.dot(trans_Q, b.reshape(-1, 1))[m:, :].reshape(-1, 1)
    d = f - np.dot(ort_map_W[m:, :m], u)
    
    sq_s = r * r - np.dot(np.transpose(u), u)
    if sq_s < 0:
        raise ValueError('Incompatible constraints! The linear constraint is not compatible with the unit hyperellipsoid constraint! Unable to find a feasible solution!') 
    
    return (u, C, d, np.sqrt(sq_s), Q)


'''
    An alternative transformation (to QR approach) that can be used to eliminate the linear constraint.
'''
def __svd_reform(W, b, r, L, y):
    m = L.shape[0]
    svd_result = np.linalg.svd(L)
    U = svd_result[0]
    V = np.transpose(svd_result[2])
    D = np.dot(svd_result[1], np.identity(m))
    
    refd_W = np.dot(np.dot(svd_result[2], W), V)
    u = np.linalg.solve(np.dot(U, D), y).reshape(-1, 1)
    C = refd_W[m:, m:]
    f = np.dot(svd_result[2], b.reshape(-1, 1))[m:, :].reshape(-1, 1)
    d = f - np.dot(refd_W[m:, :m], u)
    
    sq_s = r * r - np.dot(np.transpose(u), u) 
    if sq_s < 0:
        raise ValueError('Incompatible constraints! The linear constraint is not compatible with the unit hyperellipsoid constraint! Unable to find a feasible solution!')
    
    return (u, C, d, np.sqrt(sq_s), V)


'''
    Find the value of the parameter lambda for which the maximum is attained 
    (for details please see Gander et. all (1989)).
    
    IMPORTANT: Due to numerical instabilities more than one will be calculated!
'''
def __compute_lambda_candidates(C, b, r, k=2):
    W = __gander_matrix(C, b, r)
    eigvals = np.linalg.eigvals(W)
    eigvals = np.real(eigvals) - 1000000 * np.abs(np.sign(np.imag(eigvals)))
    sorted_eig_inds = np.argsort(eigvals)
    return eigvals[sorted_eig_inds[-k:]]


def __gander_matrix(C, b, r):
    m = C.shape[0]
    b = b.reshape(-1, 1)
    W = np.zeros((2 * m, 2 * m))
    
    W[:m, :m] = C
    W[:m, m:] = -np.identity(m)
    W[m:, :m] = -float(1.0 / (r * r)) * np.dot(b, np.transpose(b))
    W[m:, m:] = C
    
    return W


'''
    Evaluate lambda parameter using secular equations (the second approach for computation of the correct lambda parameter)
'''
def __secular_function(l, d, eigvals, r):
    u = d / (eigvals - l)
    return float(np.dot(u, u) - r * r)


def __secular_function_derivative(l, d, eigvals):
    u = 2 * np.power(d, 2) / (np.power(eigvals - l, 3))
    return float(np.dot(u, u))
    
    
'''
    Compute the correct lambda value using the secular equation approach.
'''
def __compute_secular_lambda_max(C, b, r):
    eigvals, eigvecs = np.linalg.eigh(C)
    U = np.asmatrix(eigvecs)
    d = np.asarray(np.dot(np.transpose(U), b.reshape(-1, 1))).reshape(-1)
    
    eigvals = np.real(eigvals)
    sorted_eig_inds = np.argsort(eigvals)
    top_eigval = float(eigvals[sorted_eig_inds[-1]])
    epsilon = max(float(np.abs(d[0]) / r), 0.05)
    return scopt.fsolve(lambda x: __secular_function(np.real(x), d, eigvals, r), top_eigval + epsilon, xtol=1.49012e-10)
    

''' 
    Returns the largest eigenvalue and eigenvector of a given matrix
    
    @author: dpaurat 
'''
def __largest_eig(matrix, epsilon=10e-10):
    n = len(matrix)
    probe = np.ones((n, 1)) / np.sqrt(n)
    probe = probe / np.linalg.norm(probe)
    delta = 1.0
    max_eigenvalue = -1.0
    while delta > epsilon:
        proj = matrix.dot(probe)
        norm = np.linalg.norm(proj)
        proj = proj / norm
        delta = np.linalg.norm(probe - proj)
        max_eigenvalue = norm
        probe = proj
    
    return max_eigenvalue


def __solver_secular_gander(d, eigvals, r):
    past_approx = None
    current_approx = max(eigvals) + float((np.abs(d[0]) + 0.01) / r)
    while past_approx == None or current_approx > past_approx:
        print('iter', past_approx, current_approx)
        past_approx = current_approx
        tmp_expr_1 = __secular_function(past_approx, d, eigvals, r) + r * r
        current_approx = past_approx - 2 * float((tmp_expr_1) / (__secular_function_derivative(past_approx, d, eigvals))) * float(np.sqrt(tmp_expr_1) / r - 1)
    
    return current_approx


'''
    Compute the solution from the stationary constraint given the parameter lambda for
    which the optimum is attained.
'''
def __find_optimizer(W, lambda_max, b):
    try:
        return np.asarray(np.linalg.solve(W - lambda_max * np.identity(W.shape[0]), b.reshape(-1, 1))).reshape(-1)
    except:
        print('WARNING: Unique solution does not exist! Trying to compute one feasible global optimum using pseudo-inverse instead of the actual inverse of the matrix from the stationary constraint!')
        '''
            The computed lambda value is equal to one of the eigenvalues of matrix W and therefore we need to compute the pseudo-inverse of the matrix W.
            
            Remark:
                i) in this case the solution does not necessarily exists (see e.g. Gander et. all (1989) for details);
                ii) for the computed solution to be the `actual' global optimum it must satisfy the stationary constraints - !!! check to be done by a user !!!
        '''
        
        return np.asarray(np.dot(np.linalg.pinv(W - lambda_max * np.identity(W.shape[0])), b.reshape(-1, 1))).reshape(-1)
    

'''
    Check whether the offered solution is feasible. In the eigen approach not every lambda is matched with a solution to stationary constraints
    of the quadratics over sphere.
'''
def __valid_norm(x, r):
    return np.abs(np.linalg.norm(x) - r) < PRECISION


'''
    Choose the best candidate solution, i.e. choose the best value for the lambda parameter. The algorithm is based on the following assumptions:
    
        i) with high probability there is no numerical issues when the top two eigenvalues are `far enough'
        
        ii) the second eigenvalue does not have to yield the solution of the original problem
        
        iii) if both eigenvalues, the first and the second one, satisfy the stationary constraints of the problem and are `far enough' we choose the
             first one as it is larger and theoretically it should be the parameter value matched with the global optimum. On the other hand, if
             the values are close to each other, i.e. |lambda_2 - lambda_1| < THRESHOLD, we choose the second one as the empirical evidence show that
             this is the one at which the global optimum is attained. 
'''
def __best_optim_cand(lambdas, W, b, r):
    cand_sols = []  
     
    for i in range(lambdas.shape[0]):
        opt_i = __find_optimizer(W, lambdas[i], b)
        if __valid_norm(opt_i, r):
            cand_sols.append((lambdas[i], opt_i))
        
    if len(cand_sols) == 0:
        raise ValueError('The problem is not well defined! Unable to compute the global optimizer!')
    elif len(cand_sols) == 1:
        return cand_sols[0][1]
    
    l1 = cand_sols[-1][0]
    l2 = cand_sols[-2][0]
    if np.abs(float(l2 - l1)) < THRESHOLD:
        return cand_sols[-2][1]
    return cand_sols[-1][1]
    

'''
    Compute the maximizer of a quadratic function over the unit sphere, i.e. solve the following optimization problem
    
        argmax_x    x^T W x - 2 b^T x
         s.t.       x^T x = r^2.
'''
def __quad_over_sphere_maximizer(W, b, r, slv_mode='eigen'):
    if slv_mode == 'eigen':
        lambdas = __compute_lambda_candidates(W, b, r)
#        print 'Eigen lambdas', lambdas
        return __best_optim_cand(lambdas, W, b, r)
    elif slv_mode == 'power_it':
        lambda_max = __largest_eig(__gander_matrix(W, b, r))
        return __find_optimizer(W, lambda_max, b)
    else:
        lambda_max = __compute_secular_lambda_max(W, b, r)
#        print 'Secular lambda', lambda_max
        return __find_optimizer(W, lambda_max, b)


'''
    The solver finds the maximizer of a linearly constrained quadratic over the r-hyperellipsoid:
        argmax    x^T W x - 2 b^T x
         s.t.     x^T K x = r^2, 
                  L x = y.
         
    Input parameters (in this order): 
        W - [mandatory parameter] a symmetric real valued matrix (not necessarily positive definite) of dimension n x n,
        r - [mandatory parameter] a radius of the hyperellipsoid,
        b - a vector of dimension n x 1,   
        L - a linear constraint matrix of dimension m x n (m < n),
        y - a linear constraint value vector of dimension m x 1,
        K - a positive definite symmetric real valued matrix of dimension n x n (when omitted it will be treated as the identity matrix).
    
    Output:
        Solution vector in the form (n,)
    
    Requirements: numpy
'''
def maximizer(W, r, b=None, slv_mode='eigen', L=None, y=None, K=None, pre_cmp_ell2sph_trans=None):
    __validate_parameters(W, r, b, K, L, y)
    n = W.shape[0]
    optimizer = np.zeros(n)
    
    b = None if b != None and all(b == 0) else b
    sph_tuple = __from_ellipsoid_to_sphere(W, L, b, K, pre_cmp_ell2sph_trans)
    
    if b == None and L == None:
        '''
            Case I:        b = None, L = None, y = None
            In this case we solve an eigenvalue problem.
        '''
        optimizer[:] = r * __top_eigenvector(sph_tuple[0]).reshape(-1)
    elif L == None:
        ''' 
            Case II:       L = None, y = None
            In this case we are optimizing a quadratic over the unit hypersphere. The solution is
            implemented per instructions given in Gander et. all (1989).
        '''
        optimizer[:] = __quad_over_sphere_maximizer(sph_tuple[0], sph_tuple[2], r, slv_mode)
    else:
        '''
            Case III:      All parameters passed
            In this case we need a QR factorization to transform the problem to Case II
        '''
        sph_b = np.zeros((n, 1)) if sph_tuple[2] == None else sph_tuple[2]
        ort_tuple = __qr_transform(sph_tuple[0], sph_b, r, sph_tuple[1], y)
#        ort_tuple = __svd_reform(k_refd_W, k_refd_b, r, k_refd_L, y)
        v = __quad_over_sphere_maximizer(ort_tuple[1], ort_tuple[2], ort_tuple[3], slv_mode)
        
        '''
            Solution must be packed correctly. The first part of the solution was computed from the constraint and the rest as
            the optimizer of the problem similar to the one solved in Case II.
        '''
        m = ort_tuple[0].shape[0]
        optimizer[:m] = ort_tuple[0].reshape(-1)[:]
        optimizer[m:] = v
        optimizer = np.asarray(np.dot(ort_tuple[4], optimizer.reshape(-1, 1))).reshape(-1)
    
    return __from_sphere_to_ellipsoid(optimizer, sph_tuple[3])
