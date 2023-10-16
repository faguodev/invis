'''
Created on Nov 26, 2013

@author: doglic
'''

import numpy as np

'''
    Input:
        K - kernel matrix
        y - label vector
        idx_mask - array of indices corresponding to labeled data
        c_nu - measure importance of the term in the SKPCA optimization problem
        
    Output:
        quadratic term, linear term
'''
def classification_term(K, y, idx_mask, c_nu):
    n = K.shape[0]
    l = idx_mask.shape[0]
    w = float((c_nu * n) / l)
    lin_term = w * np.asarray(np.dot(y[idx_mask, :].reshape(1, -1), K[idx_mask, :])).reshape(-1)
    return np.zeros(K.shape), lin_term


'''
    Input:
        K - kernel matrix
        y - label vector
        idx_mask - array of indices corresponding to labeled data
        mlcl_nu - measure importance of the term in the SKPCA optimization problem
    
    Output:
        quadratic term, linear_term
'''
def ml_cl_term(K, y, labelled_idxs, mlcl_nu):
    n = K.shape[0]
    l = labelled_idxs.shape[0]
    L = np.zeros(K.shape)
    lab_y = y[labelled_idxs, :]
    
    for i in range(l):
        ilab_pos = labelled_idxs[i]
        for j in range(i + 1, l):
            jlab_pos = labelled_idxs[j]
            if lab_y[i, :] == lab_y[j, :]:
                L[ilab_pos, jlab_pos] = 1.0
                L[jlab_pos, ilab_pos] = 1.0
                L[ilab_pos, ilab_pos] = L[ilab_pos, ilab_pos] - 1.0
                L[jlab_pos, jlab_pos] = L[jlab_pos, jlab_pos] -1.0
            else:
                L[ilab_pos, jlab_pos] = -1.0
                L[jlab_pos, ilab_pos] = -1.0
                L[ilab_pos, ilab_pos] = L[ilab_pos, ilab_pos] + 1.0
                L[jlab_pos, jlab_pos] = L[jlab_pos, jlab_pos] + 1.0
    
    quad_term = float((mlcl_nu * n) / l) * np.dot(np.dot(K, L), K)
    
    return quad_term, np.zeros(K.shape[0])


'''
    Input:
        K - kernel matrix
        y - real valued vector
        idx_mask - array of indices corresponding to labeled data
        cp_nu - measures importance of the term in the SKPCA optimization problem
        
    Output:
        quadratic term, linear term
'''
def control_points_term(K, y, idx_mask, cp_nu):
    n = K.shape[0]
    l = idx_mask.shape[0]
    w = float((cp_nu * n) / l)
    lab_sub_K = K[idx_mask, :]
    quad_term = w * np.dot(np.transpose(lab_sub_K), lab_sub_K)
    lin_term = -2 * w * np.asarray(np.dot(y[idx_mask, :].reshape(1, -1), lab_sub_K)).reshape(-1)
    
    return quad_term, lin_term