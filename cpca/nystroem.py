
import numpy as np
import logging
import time

from abc import (ABCMeta, abstractmethod)

class Nystroem(object):

    def __init__(self, sklearn_kernel_function, params):
        self.logger = logging.getLogger('Nystroem')
        self.kernel_function = sklearn_kernel_function
        self.params = params

        self.np_linalg_norm_type = {'frobenius': 'fro', 'spectral': 2}

    """
        The method computes a low-rank factorization of kernel matrix using the Nystroem approach, i.e., K ~= SS^t.

        Parameters:

            X : numpy array
                shape=(n, d), where n is the number of instances and d is the dimension of the problem

            landmarks : numpy array
                        shape=(m, d), where m is the number of landmarks and d is the dimension of the problem

        Output:

            S : numpy array
                shape=(n, m), where n is the number of instances and m is the number of landmarks
    """
    def transform(self, X, landmarks):
        L = self.kernel_function(landmarks, **self.params)
        eig_vals, eig_vecs = np.linalg.eigh(L)
        sqrt_eig_vals = np.sqrt(np.maximum(eig_vals, 1e-12))
        mixed_kmat_block = self.kernel_function(X, landmarks, **self.params)
        return mixed_kmat_block.dot(np.divide(eig_vecs, sqrt_eig_vals.reshape(1, -1)))

    """
        The method computes a low-rank factorization of kernel matrix using the one-shot Nystroem method where m landmarks are selected
        to make a rank k approximation of the kernel matrix (with m > k).

        Parameters:

            X : numpy array
                shape=(n, d), where n is the number of instances and d is the dimension of the problem

            landmarks : numpy array
                        shape=(m, d), where m is the number of landmarks and d is the dimension of the problem

            approximation_rank : int

        Output:

            S : numpy array
                shape=(n, k), where n is the number of instances and k is the rank of the approximation
                K ~= SS^t
    """
    def one_shot_transform(self, X, landmarks, approximation_rank):
        S = self.transform(X, landmarks)
        eig_vecs = np.linalg.eigh(S.T.dot(S))[1][:, -approximation_rank:]
        return S.dot(eig_vecs)

    """
        The method evaluates a landmark selection strategy.

        Parameters:

            X : numpy array
                shape=(n, d), where n is the number of instances and d is the dimension of the problem

            landmarks : numpy array
                        shape=(m, d), where m is the number of landmarks and d is the dimension of the problem

            approximation_rank : int

            K : numpy array
                shape=(n, n), where n is the number of instances

            norm : [optional: frobenius | spectral] string
                   type of the norm for the evaluation of the approximation error

        Output:

            error : float
    """
    def approximation_error(self, X, landmarks, approximation_rank, K, norm='frobenius'):
        S = self.one_shot_transform(X, landmarks, approximation_rank)
        err = np.linalg.norm(K - S.dot(S.T), ord=self.np_linalg_norm_type[norm])
        self.logger.debug('approximation error with selected landmarks: ' + str(err))
        return err


class SampleQuantizer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def select_landmarks(self, X, num_landmarks, params=None):
        pass


class KernelKMeansSQ(SampleQuantizer):

    def __init__(self, sklearn_kernel_function, kernel_params):
        self.logger = logging.getLogger('KernelKMeansSQ')
        self.sklearn_kernel_function = sklearn_kernel_function
        self.kernel_params = kernel_params

    def __sample_cluster_centers(self, X, kmat_diagonal, cp_distances, cp_potential, num_local_trials):
        rand_vals = np.random.rand(num_local_trials) * cp_potential
        candidate_ids = np.unique(np.searchsorted(cp_distances.cumsum(), rand_vals))
        distance_to_candidates = kmat_diagonal[candidate_ids].reshape(-1, 1) + kmat_diagonal.reshape(1, -1) - \
                                 2 * self.sklearn_kernel_function(X[candidate_ids], X, **self.kernel_params)

        best_candidate, best_cp_distances, best_cp_potential = None, None, None
        for trial, candidate in enumerate(candidate_ids):
            candidate_cp_distances = np.minimum(cp_distances, distance_to_candidates[trial])
            candidate_potential = candidate_cp_distances.sum()
            if (best_candidate is None) or (candidate_potential < best_cp_potential):
                best_candidate = candidate
                best_cp_potential = candidate_potential
                best_cp_distances = np.copy(candidate_cp_distances)

        return best_candidate, best_cp_distances, best_cp_potential

    """
        A set of landmarks is sampled using the kernel K-means++ sampling scheme (without Lloyd refinements).

        Parameters:

            X : numpy array
                shape=(n, d), where n is the number of instances and d is the dimension of the problem

            num_landmarks : int

            params : [optional] dictionary
                     num_local_trials : int
                                        number of trials for each centroid (except the first) -- among these the one with the largest
                                        reduction in the clustering potential is selected as the next sample (Arthur & Vassilvitskii, 2007)

        Output:

            landmarks : numpy array
                        shape=(m, d), where m is the number of landmarks and d is the dimension of the problem
    """
    def select_landmarks(self, X, num_landmarks, params={'num_local_trials': None}):
        start_t = time.time()

        n = X.shape[0]
        if params['num_local_trials'] is None:
            num_local_trials = 2 + int(np.log(num_landmarks))
        else:
            num_local_trials = params['num_local_trials']
            
        kmat_diagonal = np.array([self.sklearn_kernel_function(X[i].reshape(1, -1), X[i].reshape(1, -1), **self.kernel_params)[0] for i in range(n)])
        kmat_diagonal = np.squeeze(kmat_diagonal)

        init_landmark_id = np.random.randint(n)
        selected_landmarks = [init_landmark_id]

        cp_dists = kmat_diagonal[init_landmark_id] + kmat_diagonal - \
                   2 * self.sklearn_kernel_function(X[init_landmark_id].reshape(1, -1), X, **self.kernel_params)
        cp_potential = cp_dists.sum()

        for i in range(1, num_landmarks):
            landmark_id, cp_dists, cp_potential = self.__sample_cluster_centers(X, kmat_diagonal, cp_dists, cp_potential, num_local_trials)
            selected_landmarks.append(landmark_id)

        self.logger.debug('[Kernel K-Means++ Sample Quantization] TIME: ' + str(time.time() - start_t))

        return X[selected_landmarks]