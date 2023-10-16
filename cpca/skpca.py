'''
Created on Nov 28, 2013

@author: doglic
'''

import numpy as np
import sys
import scipy.optimize as scopt
import warnings
sys.path.append('./../constraints')
sys.path.append('./../optimization')
sys.path.append('./../common')

import cons_interpreter as ci
import lcqus_solver as slv
import utils


class embedder(object):
    
    def __init__(self, slv_mode='secular', sec_root_finder_meth='gander'):
        self.__slv_mode = slv_mode
        self.__sec_root_finder_meth = sec_root_finder_meth
        self.__max_secular_iters = 800
        self.__gander_secular_precision = 1e-15
        self.__worst_case_secular_precision = 1e-5
    
    '''
        Reformulate the optimization problem by introducing a substitution K^{1/2} x = s.
    '''
    def __from_ellipsoid_to_sphere(self, W, L, b, K, transf_opers=None):
        if None == K:
            return (W, L, b, (None, None))
        
        if None == transf_opers:
            transf_opers = self.sym_psd_sqrt_and_sqrt_inv(K)
        sph_W = transf_opers[1].dot(W).dot(transf_opers[1]) 
        sph_b = transf_opers[1].dot(b) if None != b else None    
        sph_L = L.dot(transf_opers[1]) if None != L else None
        
        return (sph_W, sph_L, sph_b, transf_opers)
    
    def sym_psd_sqrt_and_sqrt_inv(self, K):
        eigvals, eigvecs = np.linalg.eigh(K)
        min_real = float(min(np.real(eigvals)))
        if min_real <= 0:
            warnings.warn('WARNING: The kernel matrix K is poorly conditioned! Adding noise to the diagonal...') 
            eigvals = eigvals + np.abs(min_real) + 1.0e-6 * np.random.rand()
        
        sqrt_eig_vals = np.sqrt(eigvals)
        sqrt_diag = np.diag(sqrt_eig_vals)
        sqrt_inv_diag = np.diag(1.0 / sqrt_eig_vals)
        U = np.asmatrix(eigvecs)
        sqrt_mat = (U.dot(sqrt_diag)).dot(U.T)
        sqrt_inv_mat = (U.dot(sqrt_inv_diag)).dot(U.T)
        return (sqrt_mat, sqrt_inv_mat)
    
    '''
        Reformulate the optimization problem (the one we get after __from_ellipsoid_to_sphere) using substitution
        Q^T s = t, where Q comes from QR factorization of the matrix L^T.
    '''
    def __qr_transform(self, W, b, r, y, qr_res):
        m = y.shape[0]            
        Q = qr_res[0]
        R = qr_res[1]    
        P = R.T[:m, :m]
        
        ort_map_W = Q.T.dot(W).dot(Q)
        C = ort_map_W[m:, m:]
        u = np.linalg.solve(P, y.reshape(-1, 1))
        f = Q.T.dot(b)[m:, :].reshape(-1, 1)
        d = f - ort_map_W[m:, :m].dot(u)
        
        sq_s = r * r - u.T.dot(u)
        if sq_s < 0:
            raise ValueError('Incompatible constraints! The linear constraint is not compatible with the unit hyperellipsoid constraint! Unable to find a feasible solution!') 
        
        return (u, C, d, np.sqrt(sq_s), Q)
    
    def __hc_orth_qr_transform(self, W, b, r, m, qr_res):   
        ort_map_W = qr_res[0] .T.dot(W).dot(qr_res[0] )
        C = ort_map_W[m:, m:]
        f = qr_res[0] .T.dot(b)[m:, :].reshape(-1, 1)   
        return (np.zeros((m, 1)), C, f, r, qr_res[0] )
    
    '''
        Return the top k eigenvectors of the matrix W.
    '''
    def __top_k_eigenvectors(self, W, k):
        eig_vals, eig_vecs = np.linalg.eigh(W)
        U = np.asmatrix(eig_vecs)
        sorted_idxs = np.argsort(eig_vals)
        return U[:, sorted_idxs[-k:]]
    
    '''
        Compute the maximizer using the eigen approach (for details please see Gander et. all (1989)).
        
        IMPORTANT: Due to numerical instabilities the top eigenvalue might not be the right one!
    '''
    def __compute_eigen_maximizer(self, C, b, r):
        W = self.__gander_matrix(C, b, r)
        eigvals = np.linalg.eigvals(W)
        eigvals = np.real(eigvals) - 1000000 * np.abs(np.sign(np.imag(eigvals)))
        top_eigval = max(eigvals)
        return self.__find_optimizer(C, top_eigval, b)
    
    def __gander_matrix(self, C, b, r):
        m = C.shape[0]
        b = b.reshape(-1, 1)
        W = np.zeros((2 * m, 2 * m))
        
        W[:m, :m] = C
        W[:m, m:] = -np.identity(m)
        W[m:, :m] = -float(1.0 / (r * r)) * np.dot(b, np.transpose(b))
        W[m:, m:] = C
        
        return W
    
    '''
        Evaluate the secular equation at the given point.
    '''
    def __secular_function(self, l, d, eigvals, r):
        u = d / (eigvals - l)
        return float(u.T.dot(u) - r * r)
    
    
    def __secular_function_derivative(self, l, d, eigvals):
        u = 2 * np.power(d, 2) / (np.power(eigvals - l, 3))
        return float(np.sum(u))
            
    '''
        Compute the maximizer using secular approach relying on the SciPy root finder
    '''
    def __powel_secular_root(self, d, eigvals, r, top_eigen_val, epsilon):
        try:
            return scopt.fsolve(lambda x: self.__secular_function(np.real(x), d, eigvals, r), x0=top_eigen_val + epsilon, xtol=1.49012e-10)
        except:
            warnings.warn('SciPy failed to compute the root of the secular equations!')
            return top_eigen_val
    
    '''
        Compute the secular root using spline smoothing algorithm.
    '''
    def __gander_secular_root(self, d, eigvals, r, top_eigen_val, epsilon, restarted=False):
        current_approx = top_eigen_val + epsilon
        squared_radius = r * r
        iteration = 0
        previous_approx = None
        while (previous_approx == None or np.abs(current_approx - previous_approx) > self.__gander_secular_precision) and iteration < self.__max_secular_iters:
            previous_approx = current_approx
            u = self.__secular_function(previous_approx, d, eigvals, r) + squared_radius
            v = self.__secular_function_derivative(previous_approx, d, eigvals)
            # if the derivative is equal to 0 then the point is a stationary point of the secular equation and we need to break the loop
            if np.abs(v) < self.__gander_secular_precision:
                break
            
            current_approx = previous_approx - 2 * float(u / v) * float(np.sqrt(u) / r - 1.0)
            # the unique solution does not exist, we must settle with a degenerate one (we must prevent the division with 0 in the computation of the derivative)
            if np.abs(current_approx - top_eigen_val) < self.__gander_secular_precision:
                current_approx = top_eigen_val
                break
            elif restarted == False and current_approx < top_eigen_val:
                # the solution must not be less than the top eigen value
                # must move the initial guess more to the right and retry
                return self.__gander_secular_root(d, eigvals, r, top_eigen_val, 10 * epsilon, True)
        
            iteration += 1
        
        return (current_approx, iteration)
    
    def __secular_maximizer(self, C, b, r):
        eigvals, eigvecs = np.linalg.eigh(C)
        U = np.asmatrix(eigvecs)
        d = np.asarray(U.T.dot(b)).reshape(-1)
        
        # if np.linalg.eigh returns sorted eigenvalues this step can be omitted
        sorted_eig_idxs = np.argsort(eigvals)
        top_eigen_val = eigvals[sorted_eig_idxs[-1]]
        
        epsilon = self.__worst_case_secular_precision
        for i in range(d.shape[0]):
            if d[sorted_eig_idxs[-i - 1]] != 0.0:     
                epsilon = float(np.abs(d[sorted_eig_idxs[-i - 1]]) / r)
                break 
        
        iteration = -1
        if self.__sec_root_finder_meth == 'scipy':
            lambda_max = self.__powel_secular_root(d, eigvals, r, top_eigen_val, epsilon)
        else:
            gander_rf_res = self.__gander_secular_root(d, eigvals, r, top_eigen_val, epsilon)
            lambda_max = gander_rf_res[0]
            iteration = gander_rf_res[1]
        
        dist_to_top_eigen = np.abs(top_eigen_val - lambda_max)
        if dist_to_top_eigen < self.__gander_secular_precision:
            warnings.warn('It is impossible to compute the unique solution according to the secular condition. Attempting with the pseudo-inverse...')
            estimated_diag = np.diag(eigvals - lambda_max)
            pinv_estimated_diag = np.linalg.pinv(estimated_diag)
            solution_estimate = pinv_estimated_diag.dot(d)
            solution_estimate_norm = np.linalg.norm(solution_estimate) 
            if np.abs(solution_estimate_norm - r) < self.__worst_case_secular_precision:
                warnings.warn('The pseudo-inverse solution computed successfully!')
                return np.asarray(U.dot(solution_estimate)).reshape(-1)
            elif solution_estimate_norm < r:
                warnings.warn('A solution from the set of feasible solutions computed with the help of pseudo-inverse of the quadratic term!')
                norm_delta = r - solution_estimate_norm
                solution_estimate[sorted_eig_idxs[-1]] += norm_delta
                return np.asarray(U.dot(solution_estimate)).reshape(-1)
            else:
                raise ValueError('The optimization problem does not have a solution!')
        elif iteration == self.__max_secular_iters:
            secular_delta = self.__secular_function(lambda_max, d, eigvals, r)
            if np.abs(secular_delta) > self.__worst_case_secular_precision:
                raise ValueError('The optimization problem does not have a solution!')
            
        return np.asarray(U.dot(d / (eigvals - lambda_max))).reshape(-1)
    
    '''
        Compute the solution from the stationary constraint given the parameter lambda for
        which the optimum is attained.
    '''
    def __find_optimizer(self, W, lambda_max, b):
        try:
            return np.asarray(np.linalg.solve(W - lambda_max * np.identity(W.shape[0]), b.reshape(-1, 1))).reshape(-1)
        except:
            warnings.warn('WARNING: Unique solution does not exist! Trying to compute one feasible global optimum using pseudo-inverse instead of the actual inverse of the matrix from the stationary constraint!')            
            return np.asarray(np.dot(np.linalg.pinv(W - lambda_max * np.identity(W.shape[0])), b.reshape(-1, 1))).reshape(-1)
            
    '''
        Transform the solution from the ellipsoidal domain to the spherical one.
    '''
    def __from_sphere_to_ellipsoid(self, sph_optim, inv_transform):
        if None != inv_transform:
            return inv_transform.dot(sph_optim) 
        return sph_optim
    
    '''
        Compute the maximizer of a quadratic function over the unit sphere, i.e. solve the following optimization problem
        
            argmax_x    x^T W x - 2 b^T x
             s.t.       x^T x = r^2.
    '''
    def __quad_over_sphere_maximizer(self, W, b, r):
        if self.__slv_mode == 'eigen':
            return self.__compute_eigen_maximizer(W, b, r)
        return self.__secular_maximizer(W, b, r)
        
    def __validate_parameters(self, W, r, b, K, L, y):
        n = W.shape[0]
        if None != b and b.shape[0] != n and len(b.shape) == 2 and b.shape[1] != y.shape[1]:
            raise ValueError('The b vector is of inappropriate dimension. Its expected dimension is ' + str(n) + '!')
        
        if None != L and (L.shape[1] != n or L.shape[0] >= n):
            raise ValueError('The constraint matrix L is of inappropriate dimension. Its expected dimension is m x ' + str(n) + ' with m < ' + str(n) + '!')
        
        if None != y and (L == None or y.shape[0] != L.shape[0]):
            raise ValueError('The constraint value vector is of invalid dimension or it is specified without the constraint matrix L!')
        
        if r <= 0:
            raise ValueError('The radius must be positive!')
    
    def compute_hard_orth_dirs(self, W, r, d, b=None, K=None, transf_opers=None):
        self.__validate_parameters(W, r, b, K, None, None)
        n = W.shape[0]
        directions = np.zeros((n, d))
        if None != b and len(b.shape) == 1:
            b = np.asarray(b).reshape(-1, 1)
        
        sph_tuple = self.__from_ellipsoid_to_sphere(W, None, b, K, transf_opers)
        L = np.zeros((d - 1, n)) if d > 1 else None
        for i in range(d):
            if b == None and L == None:
                directions[:, i] = r * np.asarray(self.__top_k_eigenvectors(sph_tuple[0], 1)).reshape(-1)
            elif L == None:
                directions[:, i] = self.__quad_over_sphere_maximizer(sph_tuple[0], sph_tuple[2][:, i], r)
            else:
                qr_res = utils.rect_mat_qr(L[:i, :])
                lin_term = np.zeros((n, 1)) if sph_tuple[2] == None else sph_tuple[2][:, i].reshape(-1, 1)
                ort_tuple = self.__hc_orth_qr_transform(sph_tuple[0], lin_term, r, i, qr_res)
                v = self.__quad_over_sphere_maximizer(ort_tuple[1], ort_tuple[2], ort_tuple[3])
                
                m = ort_tuple[0].shape[0]
                directions[:m, i] = ort_tuple[0].reshape(-1)[:]
                directions[m:, i] = v
                directions[:, i] = ort_tuple[4].dot(directions[:, i])
            
            if i < (d - 1):
                L[i, :] = directions[:, i]
            directions[:, i] = self.__from_sphere_to_ellipsoid(directions[:, i], sph_tuple[3][1])
            
        return directions
    
    def compute_directions(self, W, r, d, orth_nu, b=None, L=None, y=None, K=None, transf_opers=None):
        self.__validate_parameters(W, r, b, K, L, y)
        n = W.shape[0]
        directions = np.zeros((n, d))
        if None != b and len(b.shape) == 1:
            b = np.asarray(b).reshape(-1, 1)
        
        sph_tuple = self.__from_ellipsoid_to_sphere(W, L, b, K, transf_opers)
        qr_res = utils.rect_mat_qr(sph_tuple[1]) if L != None else None
        orth_term = np.zeros((n, n))        
        for i in range(d):
            quad_term = sph_tuple[0] - float((n * orth_nu) / max(i, 1)) * orth_term
            if b == None and L == None:
                directions[:, i] = r * np.asarray(self.__top_k_eigenvectors(quad_term, 1)).reshape(-1)
            elif L == None:
                directions[:, i] = self.__quad_over_sphere_maximizer(quad_term, sph_tuple[2][:, i], r)
            else:
                lin_term = np.zeros((n, 1)) if sph_tuple[2] == None else sph_tuple[2][:, i].reshape(-1, 1)
                ort_tuple = self.__qr_transform(quad_term, lin_term, r, y[:, i], qr_res)
                v = self.__quad_over_sphere_maximizer(ort_tuple[1], ort_tuple[2], ort_tuple[3])
                
                m = ort_tuple[0].shape[0]
                directions[:m, i] = ort_tuple[0].reshape(-1)[:]
                directions[m:, i] = v
                directions[:, i] = ort_tuple[4].dot(directions[:, i])
            
            directions[:, i] = self.__from_sphere_to_ellipsoid(directions[:, i], sph_tuple[3][1])
            if i < (d - 1):
                directional_update_mat = directions[:, i].reshape(-1, 1).dot(directions[:, i].reshape(1, -1))
                if None != sph_tuple[3][0]:
                    orth_term += sph_tuple[3][0].dot(directional_update_mat).dot(sph_tuple[3][0])
                else:
                    orth_term += directional_update_mat
        
        return directions

    def cl_mode_directions(self, kernel, y, labelled_idxs, params, transf_opers=None):
        n = kernel.shape[0]
        d = y.shape[1]
        kb_terms = self.__interpret_cl_constraint(kernel, y, labelled_idxs, params)
        H = np.identity(n) - float(1.0 / n) * np.ones(kernel.shape)
        quad_term = kernel.dot(H).dot(kernel)
        return self.compute_directions(quad_term, params['r'], d, params['orth_nu'], -0.5 * kb_terms[1], None, None, kernel, transf_opers)
    
    def hc_cl_mode_directions(self, kernel, y, labelled_idxs, params, transf_opers=None):
        n = kernel.shape[0]
        d = y.shape[1]
        kb_terms = self.__interpret_cl_constraint(kernel, y, labelled_idxs, params)
        H = np.identity(n) - float(1.0 / n) * np.ones(kernel.shape)
        quad_term = kernel.dot(H).dot(kernel)
        return self.compute_hard_orth_dirs(quad_term, params['r'], d, -0.5 * kb_terms[1], kernel, transf_opers)
    
    def lk_lr_cl_mode_directions(self, X, y, labelled_idxs, params, quad_term):
        n = X.shape[0]
        l = labelled_idxs.shape[0]
        w = float((params['clsf_nu'] * n) / l)
        lin_term = w * X[labelled_idxs, :].T.dot(y) 
        return self.compute_directions(quad_term, params['r'], params['dim'], params['orth_nu'], -0.5 * lin_term)
    
    def lk_lr_quad_term(self, X):
        n = X.shape[0]
        mu = np.mean(X, axis=0).reshape(-1, 1)
        return X.T.dot(X) - float(1.0 / n) * mu.dot(mu.T)
    
    def sph_cl_mode_directions(self, kernel, y, labelled_idxs, params):
        n = kernel.shape[0]
        d = y.shape[1]
        kb_terms = self.__interpret_cl_constraint(kernel, y, labelled_idxs, params)
        H = np.identity(n) - float(1.0 / n) * np.ones(kernel.shape)
        quad_term = kernel.dot(H).dot(kernel)
        return self.compute_directions(quad_term, params['r'], d, params['orth_nu'], -0.5 * kb_terms[1])
    
    def __interpret_cl_constraint(self, K, y, labelled_idxs, params):
        n = K.shape[0]
        l = labelled_idxs.shape[0]
        w = float((params['clsf_nu'] * n) / l)
        lin_term = w * K[:, labelled_idxs].dot(y) 
        return (np.zeros(K.shape), lin_term)
    
    def sl_reg_mode_directions(self, kernel, y, labelled_idxs, params, transf_opers=None):
        n = kernel.shape[0]
        d = y.shape[1]
        kb_terms = self.__interpret_sl_reg_constraint(kernel, y, labelled_idxs, params)
        H = np.identity(n) - float(1.0 / n) * np.ones(kernel.shape)
        quad_term = kernel.dot(H).dot(kernel) - kb_terms[0] 
        if kb_terms[1].all() == 0.0:
            lin_term = None
        else:
            lin_term = 0.5 * kb_terms[1]
        return self.compute_directions(quad_term, params['r'], d, params['orth_nu'], lin_term, None, None, kernel, transf_opers)
    
    def __interpret_sl_reg_constraint(self, K, y, labelled_idxs, params):
        n = K.shape[0]
        l = labelled_idxs.shape[0]
        w = float((params['scp_nu'] * n) / l)
        lab_sub_K = K[labelled_idxs, :]
        quad_term = w * lab_sub_K.T.dot(lab_sub_K) 
        lin_term = -2 * w * lab_sub_K.T.dot(y) 
        return (quad_term, lin_term)
    
    def hc_reg_mode_directions(self, kernel, y, labelled_idxs, params, transf_opers=None):
        n = kernel.shape[0]
        d = y.shape[1]
        L = kernel[labelled_idxs, :]
        H = np.identity(n) - float(1.0 / n) * np.ones(kernel.shape)
        quad_term = kernel.dot(H).dot(kernel) 
        return self.compute_directions(quad_term, params['r'], d, params['orth_nu'], None, L, y, kernel, transf_opers)


def __skpca_dirs(K, params, proj_dim, c_quad_term, c_lin_term, hc_mat=None, hc_y=None, pre_cmp_ell2sph_trans=None):
    n = K.shape[0]
    H = np.identity(n) - float(1.0 / n) * np.ones((K.shape))
    if None == pre_cmp_ell2sph_trans:
        pre_cmp_ell2sph_trans = utils.sqrt_inverse_sym_pd_mat(K)
    
    quad_term = np.dot(np.dot(K, H), K) + c_quad_term
    orth_quad_term = np.zeros(K.shape)
    
    alpha = np.zeros((n, proj_dim))
    
    for i in range(proj_dim):
        if i == 0:
            alpha[:, i] = slv.maximizer(quad_term, params['r'], c_lin_term, params['slv_mode'], hc_mat, hc_y, K, pre_cmp_ell2sph_trans)
        else:
            orth_quad_term = orth_quad_term + np.dot(np.dot(K, np.dot(alpha[:, i - 1].reshape(-1, 1), alpha[:, i - 1].reshape(1, -1))), K)
            combined_quad_term = quad_term - float((n * params['orth_nu']) / i) * orth_quad_term
            alpha[:, i] = slv.maximizer(combined_quad_term, params['r'], c_lin_term, params['slv_mode'], hc_mat, hc_y, K, pre_cmp_ell2sph_trans)
    
    return alpha


def ss_hard_cp_kpca_dirs(K, L, y, orthogonality_term, proj_dim, params, sqrt_inverse_K=None):
    return __skpca_dirs(K, params, proj_dim, orthogonality_term, None, L, y, sqrt_inverse_K)


def ss_ml_cl_cp_kpca_dirs(K, y, labelled_idxs, proj_dim, params):
    quad_term, lin_term = ci.ml_cl_term(K, y, labelled_idxs, params['mlcl_nu'])
    return __skpca_dirs(K, params, proj_dim, -quad_term, lin_term)


def ss_slack_cp_kpca_dirs(K, y, labelled_idxs, proj_dim, params):
    quad_term, lin_term = ci.control_points_term(K, y, labelled_idxs, params['scp_nu'])
    return __skpca_dirs(K, params, proj_dim, -quad_term, lin_term)
    

def ss_clsf_kpca_dirs(K, y, labelled_idxs, proj_dim, params):
    quad_term, lin_term = ci.classification_term(K, y, labelled_idxs, params['clsf_nu'])
    return __skpca_dirs(K, params, proj_dim, quad_term, lin_term)


# done by daniel --> check me ;)
def daniels_ss_slack_cp_kpca_dirs(K, y, labelled_idxs, orthogonality_term, proj_dim, params, sqrt_inverse_K=None):
    quad_term, lin_term = ci.control_points_term(K, y, labelled_idxs, params['scp_nu'])
    return __skpca_dirs(K, params, proj_dim, -quad_term + orthogonality_term, lin_term, pre_cmp_ell2sph_trans=sqrt_inverse_K)
