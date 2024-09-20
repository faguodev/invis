#!/usr/bin/python
import numpy as np
from copy import copy
try:
    import cupy as cp
    # Attempt to allocate memory on a GPU to confirm CUDA availability
    try:
        _ = cp.ones(3)
        print(_)
        print("Using CuPy for GPU acceleration.")
    except Exception as e:
        import numpy as cp
        print("Failed to use GPU. Falling back to NumPy.")

except ImportError:
    import numpy as cp
    print("CuPy not installed. Using NumPy.")

from sklearn import decomposition
from collections import defaultdict
# from scipy.spatial import distance
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# explicitly imported "hidden imports" for pyinstaller
#from sklearn.utils import weight_vector, lgamma
from sklearn.metrics import pairwise_distances
import sklearn.manifold as manifold

# Dinos solver
import cpca.solvers as solvers
import cpca.kernel_gen as kernel_gen
import cpca.utils as utils
from cpca.nystroem import Nystroem, KernelKMeansSQ
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow import keras

class PopupSlider(QDialog):
    def __init__(self, parent=None, label_text='', default=4, minimum=1, maximum=20):
        super().__init__(parent)  # Pass the parent to the QDialog constructor
        self.slider_value = default

        name_label = QLabel()
        name_label.setText(label_text)
        name_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(default)

        self.value_label = QLabel()
        self.value_label.setText('%d' % (self.slider.value()))
        self.slider.valueChanged.connect(self.slider_changed)

        self.button = QPushButton('Ok', self)
        self.button.clicked.connect(self.handleButton)

        layout = QGridLayout(self)
        layout.addWidget(name_label      , 1, 1, 1, 4, Qt.AlignLeft)
        layout.addWidget(self.slider     , 2, 1, 2, 1, Qt.AlignLeft)
        layout.addWidget(self.value_label, 2, 2, 2, 2, Qt.AlignCenter)
        layout.addWidget(self.button     , 2, 4, 2, 4, Qt.AlignRight)

        self.setWindowTitle('Parameter choice')
        self.setModal(True)  # Ensure the dialog is modal

    def slider_changed(self):
        val = self.slider.value()
        self.value_label.setText('%d' % val)
        self.slider_value = val

    def handleButton(self):
        self.accept()  # Closes the dialog and returns a success state

    def exec_(self):
        if self.parent():
            # Center the dialog on the parent window
            parent_geometry = self.parent().geometry()
            self_geometry = self.geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self_geometry.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self_geometry.height()) // 2
            self.move(x, y)
        return super().exec_()



class Embedding(object):
    def __init__(self, data, points, parent):
        self.data = data
        self.original_control_points = None
        self.original_control_point_indices = None
        self.control_points = None
        self.control_point_indices = None
        self.parent = parent
        self.X = np.array([])
        self.Y = np.array([])
        self.ml = []
        self.cl = []
        self.has_ml_cl_constraints = False
        self.projection_matrix = np.zeros((2, len(self.data[0])))
        self.name = ''
        self.is_dynamic = False
        self.update_control_points(points)

    def get_embedding(self):
        pass

    def update_must_and_cannot_link(self, ml, cl):
        self.ml = ml
        self.cl = cl
        if (len(self.ml) > 0) or (len(self.cl) > 0):
            self.has_ml_cl_constraints = True
        else:
            self.has_ml_cl_constraints = False

    def augment_control_points(self, e):
        avg_median = np.average(abs(np.median(e, axis=0)))
        tmp_points = defaultdict(list)
        if len(self.cl) > 0:
            for pair in self.cl:
                if len(pair) == 2:
                    i, j = list(pair)
                    x1 = e[i]
                    x2 = e[j]
                    diff = x1 - x2
                    norm = np.linalg.norm(diff)
                    new_x1 = x1 + (diff/norm)*5*avg_median
                    new_x2 = x2 - (diff/norm)*5*avg_median
                    if i not in self.control_point_indices:
                        e[i] = new_x1
                        tmp_points[i] = new_x1
                    if j not in self.control_point_indices:
                        e[j] = new_x2
                        tmp_points[j] = new_x2
        if len(self.ml) > 0:
            for pair in self.ml:
                if len(pair) == 2:
                    i, j = list(pair)
                    x1 = e[i]
                    x2 = e[j]
                    diff = x1 - x2
                    new_x1 = x1 - 0.45*diff
                    new_x2 = x2 + 0.45*diff
                    if i not in self.control_point_indices:
                        e[i] = new_x1
                        tmp_points[i] = new_x1
                    if j not in self.control_point_indices:
                        e[j] = new_x2
                        tmp_points[j] = new_x2
        for k,v in tmp_points.items():
            self.control_point_indices.append(k)
            self.control_points.append(v)
        self.X = self.data[self.control_point_indices]
        self.Y = np.array(self.control_points)

    def update_control_points(self, points):
        self.control_point_indices = []
        self.control_points = []
        for i, coords in points.items():
            self.control_point_indices.append(i)
            self.control_points.append(coords)
        self.X = self.data[self.control_point_indices]
        self.Y = np.array(self.control_points)

    def finished_relocating(self):
        pass





class PCA(Embedding):
    def __init__(self, data, control_points, parent):
        super(PCA, self).__init__(data, control_points, parent)
        self.name = "PCA"
        self.projection_matrix = None

        try:
            pca = decomposition.PCA(n_components=2)
            pca.fit(data)
            self.projection_matrix = pca.components_
            self.embedding = np.array(pca.transform(data))
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg) 
    

    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass





class LLE(Embedding):
    def __init__(self, data, control_points, parent):
        super(LLE, self).__init__(data, control_points, parent)
        self.name = "LLE"
        try:
            self.w = PopupSlider(parent, 'Enter number of neighbors to consider (default is 4):')
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 4
            try:
                lle = manifold.LocallyLinearEmbedding(n_neighbors=int(num), out_dim=2)
            except:
                lle = manifold.LocallyLinearEmbedding(n_neighbors=int(num), n_components=2)
            lle.fit(data)
            self.embedding = np.array(lle.transform(data))
        except Exception as e:
            print(e)
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg)


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass






class XY(Embedding):
    def __init__(self, data, control_points, parent):
        super(XY, self).__init__(data, control_points, parent)
        self.name = "XY"
        used_attributes = []
        for row in range(self.parent.series_list_model.rowCount()):
            model_index = self.parent.series_list_model.index(row, 0)
            checked = self.parent.series_list_model.data(model_index, Qt.CheckStateRole) == QVariant(Qt.Checked)
            if checked:
                if len(used_attributes) < 2:
                    name = str(self.parent.series_list_model.data(model_index))
                    used_attributes.append(list(self.parent.data.attribute_names).index(name))
                    # print self.parent.data.attribute_names[used_attributes[-1]]
                else:
                    break

        self.embedding = np.array(self.parent.data.original_data.T[used_attributes].T)   


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass







class ISO(Embedding):
    def __init__(self, data, control_points, parent):
        super(ISO, self).__init__(data, control_points, parent)
        self.name = "ISO"
        try:
            self.w = PopupSlider(parent, 'Enter number of neighbors to consider (default is 4):')
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 4
            iso = manifold.Isomap(n_neighbors=num, n_components=2)

            self.embedding = iso.fit_transform(data)  
        except Exception as e:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting. Error: " + str(e)
            QMessageBox.about(parent, "Embedding error", msg)


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass








class tSNE(Embedding):
    def __init__(self, data, control_points, parent):
        super(tSNE, self).__init__(data, control_points, parent)
        self.name = "t-SNE"
        try:
            self.w = PopupSlider(parent, 'Enter perplexity (default is 30):', default=30, minimum=1, maximum=100)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 30
            m, ok = QInputDialog.getText(parent, 'Metric', 'Enter number of the desired metric:\n1) Euclidean (Default)\n2) Jaccard\n3) L1 norm')
            metric = 'euclidean'
            if m == '2':
                metric = 'jaccard'
            elif m == '3':
                metric = 'l1'            
            tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=num, metric=metric)
            self.embedding = np.array(tsne.fit_transform(data))
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg) 

    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass






class MDS(Embedding):
    def __init__(self, data, control_points, parent):
        super(MDS, self).__init__(data, control_points, parent)
        self.name = "MDS"
        metric, ok = QInputDialog.getText(parent, 'Metric', 'Please select a metric:\n\n1) L1\n2) Euclidean (Default)\n3) Cosine\n4) Mahalanobis')
        if metric == '1':
            m = 'l1'
        elif metric == '2':
            m = 'euclidean'
        elif metric == '3':
            m = 'cosine'
        elif metric == '4':
            m = 'mahalanobis'
        else:
            m = 'euclidean'
        parent.setWindowTitle('InVis: ' + parent.data.dataset_name + ' (MDS [%s])'%m)
        dists = pairwise_distances(data, metric=m)
        dists = (dists + dists.T)/2.0
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
        self.embedding = mds.fit_transform(dists)

    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass





class ICA(Embedding):
    def __init__(self, data, control_points, parent):
        super(ICA, self).__init__(data, control_points, parent)
        self.name = "ICA"
        try:
            ica = decomposition.FastICA(n_components=2)
            ica.fit(data)
            self.embedding = np.array(ica.transform(data))   
        except:
            msg = "It seems like the embedding algorithm did not converge with the given parameter setting"
            QMessageBox.about(parent, "Embedding error", msg)


    def get_embedding(self):
        return self.embedding.T
    

    def update_control_points(self, points):
        pass


        
        



class LSP(Embedding):
    def __init__(self, data, points, parent):
        super(LSP, self).__init__(data, points, parent)
        self.name = "LSP"
        self.is_dynamic = True 
    

    def get_embedding(self, X=[]):
        if X == []:
            X=self.data.T
        return np.dot(self.projection_matrix, X)
    

    def update_control_points(self, points):
        super(LSP, self).update_control_points(points)
        if len(self.Y) > 0:
            self.projection_matrix = np.dot(self.Y.T, np.linalg.pinv(self.X.T))
        else:
            self.projection_matrix =  np.zeros((2, len(self.data[0])))
        if self.has_ml_cl_constraints:
            self.augment_control_points(self.get_embedding().T)
            if len(self.Y) > 0:
                self.projection_matrix = np.dot(self.Y.T, np.linalg.pinv(self.X.T))
            else:
                self.projection_matrix =  np.zeros((2, len(self.data[0])))


        

class cPCA_dummy(Embedding):
    def __init__(self, data, points, parent):
        super(cPCA, self).__init__(data, points, parent)
        self.name = "cPCA"
        self.is_dynamic = True 
        self.control_point_indices = []
        self.old_control_point_indices = []
        self.finished_relocating()
    

    def get_embedding(self):
        if set(self.control_point_indices) != self.old_control_point_indices:
            self.finished_relocating()
        self.old_control_point_indices = set(self.control_point_indices)
        return np.dot(self.projection_matrix, self.data.T)


    def finished_relocating(self):
        if len(self.Y) > 0:
            self.projection_matrix = np.dot(self.Y.T, np.linalg.pinv(self.X.T))
        else:
            self.projection_matrix =  np.zeros((2, len(self.data[0])))

        
        
class ConstrainedKPCAIterative(Embedding):
    # points here is a predefined dictionary of control points. That's why update control points is called in the init part
    def __init__(self, data, points, verbose, parent):
        """
        Initialize constrained Kernel PCA embedding.

        Args:
            data (np.ndarray): Input data for the embedding.
        """
        self.data = data
        self.parent=parent
        self.control_points = []
        self.control_point_indices = []
        self.X = None
        self.Y = None
        self.projection_matrix = cp.zeros((2, self.data.shape[0]))
        self.verbose = verbose

        try:
            self.w = PopupSlider(self.parent, 'Enter desired frame-rate (default is 30):', default=30, minimum=1, maximum=100)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 30
            self.frame_rate = num
        except Exception as e:
            msg = "It seems like something went wrong in the parameter selection"
            print(e)
            QMessageBox.about(parent, "Embedding error", msg) 

        
        # control points
        self.cp_selector_m_by_n = None
        self.old_control_point_indices = []

        # Must link canont link
        self.ml = []
        self.cl = []
        self.old_ml_constraints = []
        self.old_cl_constraints = []

        # Algorithm details
        self.name = "cKPCA_iterative"
        self.is_dynamic = True 
        
        self.n = len(data)

        try:
            m, ok = QInputDialog.getText(parent, 'Kernel choice', 'Enter number of the desired kernel:\n1) Linear (Default)\n2) Gaussian\n3) Polynomial')
            
            sklearn_kernel_function = rbf_kernel

            params = {}

            match m:
                case '3':
                    sklearn_kernel_function = polynomial_kernel
                    degree, ok = QInputDialog.getText(parent, 'Degree', 'Enter degree of polynomial kernel (default is 2):')
                    if degree == '':
                        degree = 2
                    params['degree'] = float(degree)
                    print("Degree: " + str(degree))
                case '2':
                    sklearn_kernel_function = rbf_kernel
                    params = {'gamma' :  1/(2*utils.median_pairwise_distances(data[:10000])**2)}
                case _:
                    sklearn_kernel_function = linear_kernel
        except Exception as e:
            msg = "It seems like something went wrong in the parameter selection"
            print(e)
            QMessageBox.about(parent, "Embedding error", msg) 

        if self.verbose:
            print("computing landmarks")

        self.parent.update_status_bar("Calculating initial KPCA Embedding.")

        kkmeanspp = KernelKMeansSQ(sklearn_kernel_function, params)
        self.num_landmarks = 100
        self.landmarks = kkmeanspp.select_landmarks(data, self.num_landmarks)

        if self.verbose:
            print("computing kernel")
        nystroem = Nystroem(sklearn_kernel_function, params)

        S = nystroem.transform(data, self.landmarks)
        self.S = cp.asarray(S)

        temp = self.S.T @ cp.ones((self.n, 1))
        self.C_var = ((1/self.n) * ((self.S.T @ self.S) - ((1/self.n) * cp.outer(temp, temp))))

        v1 = cp.random.rand(self.num_landmarks)
        self.v1 = (v1 / cp.linalg.norm(v1)).reshape(-1, 1)

        v2 = cp.random.rand(self.num_landmarks)
        self.v2 = (v2 / cp.linalg.norm(v2)).reshape(-1, 1)

        self.C_cp = 0
        self.C_ml_cml = 0

        self.cp_const_mu = 50
        self.cl_ml_const_mu = 0.1 #0.1 seems good (upper bound) choice for cl
        self.orth_mu = 10

        self.learning_rate = 1e-1
        self.beta1 = 0.95

        self.finished_iterations = 0

        self.update_control_points(points)

        self.parent.update_status_bar("Iterative Constrained KPCA Embedding.")

    def update_must_and_cannot_link(self, ml, cl):
        self.ml = [constraint for constraint in ml if isinstance(constraint, set) and len(constraint) == 2]
        self.cl = [constraint for constraint in cl if isinstance(constraint, set) and len(constraint) == 2]

        if(self.ml != self.old_ml_constraints or self.cl != self.old_cl_constraints):
            self.old_ml_constraints = self.ml
            self.old_cl_constraints = self.cl
            self._benchmark_update_ml_cl_params()
            self._benchmark_iteration()
            self._benchmark_construct_projection_matrix()


    def _benchmark_update_ml_cl_params(self):
        if (len(self.ml) > 0) or (len(self.cl) > 0):
            # construct laplacian matrix and weigh it
            self.C_ml_cml = (self.cl_ml_const_mu / (len(self.ml) + len(self.cl))) * utils.construct_ml_cl_laplacian_matrix(self.ml, self.cl, self.S, self.num_landmarks)
        else:
            self.C_ml_cml = 0

    # points seems to be a dictionary of pointindices and their xy coordinates
    def update_control_points(self, points):

        if len(points.items()) == 0:
            self.parent.update_status_bar("Calculating initial KPCA Embedding.")


        self.finished_iterations = 0
        self.control_points = []
        self.control_point_indices = []
        for i, coords in points.items():
            self.control_point_indices.append(i)
            self.control_points.append(coords)
        self.X = self.data[self.control_point_indices]
        self.Y = cp.array(self.control_points)

        self._benchmark_update_cp_params(points)
        self._benchmark_iteration()
        self._benchmark_construct_projection_matrix()
        
        if len(points.items()) == 0:
            self.parent.update_status_bar("Iterative Constrained KPCA Embedding.")

    def _benchmark_update_cp_params(self, points):
        if points is not None and set(self.control_point_indices) != set(self.old_control_point_indices):
            self.old_control_point_indices = self.control_point_indices
            self.cp_selector_m_by_n = utils.construct_mn_selector_matrix(self.n, self.control_point_indices)
            if(len(self.control_point_indices) > 0):
                C_cp = cp.zeros((self.num_landmarks, self.num_landmarks))
                for i in self.control_point_indices:
                    C_cp = C_cp + np.outer(self.S[i], self.S[i])
                self.C_cp = (self.cp_const_mu / len(self.control_point_indices)) * (C_cp)
            else:
                self.C_cp = 0

    def _benchmark_iteration(self):
        self.v1 = self.solve_iteratively(None, 0, self.v1)
        self.v2 = self.solve_iteratively(self.v1, 1, self.v2) 

    def _benchmark_construct_projection_matrix(self):
        self.projection_matrix = cp.vstack((self.v1.T, self.v2.T))

    def start_timer(self):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        return start_event, end_event

    def check_elapsed_time(self, start_event, end_event, iteration):
        end_event.record()
        end_event.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
        if elapsed_time > 1000 / (self.frame_rate * 2):
            if self.verbose:
                print(f"Broke off after {iteration} iterations")
            return True
        return False
    
    def adjust_learning_rate(self, v_new, v, iteration):
        continue_flag = False

        if cp.linalg.norm(v_new - v) > 1.5:
            self.learning_rate = self.learning_rate / 2
            if self.verbose:
                print(f"Norm of v_new - v: {cp.linalg.norm(v_new - v)}")
                print(f"Reducing learning rate to: {self.learning_rate}")
            continue_flag = True

        v_new = v_new / cp.linalg.norm(v_new)

        if iteration % 100 == 0 and self.verbose:
            print(f"Iteration {iteration} - {cp.linalg.norm(v_new - v)}")

        if cp.linalg.norm(v_new - v) > 1.5:
            self.learning_rate = self.learning_rate / 2
            if self.verbose:
                print(f"Norm of v_new (normalized) - v: {cp.linalg.norm(v_new - v)}")
                print(f"Reducing learning rate to: {self.learning_rate}")
            continue_flag = True

        if cp.linalg.norm(v_new) < 1e-8:
            self.learning_rate = self.learning_rate / 2
            if self.verbose:
                print(f"v_new is zero vector")
                print(f"Reducing learning rate to: {self.learning_rate}")
            continue_flag = True

        return v_new, continue_flag

    def solve_iteratively(self, previous_v, dimension, v):
        """
        Solve the constrained optimization problem iteratively.

        Args:
            previous_v (cp.ndarray): The previous v vector.
            dimension (int): The dimension to solve for.
            v (cp.ndarray): The v vector.
        """

        start_time = tf.timestamp()
        C = self.C_var - self.C_cp- self.C_ml_cml
        if previous_v is not None:
            temp = self.orth_mu * cp.outer(previous_v, previous_v)
            C = C - temp

        b = cp.zeros((1,self.num_landmarks))

        if len(self.control_point_indices) > 0:
            Y_s = self.Y[:, dimension]
            b = (-1 * self.cp_const_mu / len(self.control_point_indices) * (Y_s.T @ self.cp_selector_m_by_n @ self.S)).reshape(1, -1)

        # Iteration takes very roughly 1 millisecond

        iteration = 0
        v_dw = cp.zeros((self.num_landmarks, 1))

        while True:

            current_time = tf.timestamp()
            elapsed_time = (current_time - start_time).numpy() * 1000  # Convert to milliseconds
            if self.control_point_indices and elapsed_time > 1000 / (self.frame_rate * 2):
                self.finished_iterations += iteration
                break

            iteration += 1
            
            # Nesterov lookahead
            v_lookahead = v + self.beta1 * v_dw

            # Compute gradient at lookahead position
            dw = C @ v_lookahead - b.T

            # Update velocities
            v_dw = self.beta1 * v_dw + self.learning_rate * dw

            # Update parameters
            v_new = v + v_dw

            v_new, continue_flag = self.adjust_learning_rate(v_new, v, iteration)

            if continue_flag:
                v_dw = cp.zeros((self.num_landmarks, 1))
                continue

            if cp.linalg.norm(v_new - v) < 1e-9:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
            
            v = v_new

        return v
    
    # resposible for getting the already calculated embedding
    def get_embedding(self, X=None):
        if(type(self.projection_matrix) == np.ndarray):
            return self.projection_matrix @ self.S.T
        else: 
            return cp.asnumpy(self.projection_matrix @ self.S.T)

class cPCA(Embedding):
    def __init__(self, data, points, parent):
        self.data = data
        self.control_points = []
        self.control_point_indices = []
        self.parent = parent
        self.X = None
        self.Y = np.array([])
        self.projection_matrix = np.zeros((2, len(self.data[0])))
        self.name = ''
        self.is_dynamic = False

        self.ml = []
        self.cl = []
        self.has_ml_cl_constraints = False

        self.parent.update_status_bar("Calculating initial Constrained KPCA Embedding.")

        self.name = "cPCA"
        self.projection = np.zeros((2, len(data)))
        self.pca_projection = np.zeros((2, len(data)))
        self.is_dynamic = True 
        self.old_control_point_indices = []

        # r = radius? slv_mode: type of solver, sigma, degree, epsilon are parameters for the kernel
        self.params = {'r' : 3.0, 'slv_mode' : 'secular', 'sigma' : None, 'epsilon' : 0.5, 'degree' : 1}
        self.params['const_nu'] = 5e+3 # constraint parameter?
        self.params['orth_nu'] = 5e+3 # orthogonality parameter?
        self.params['sigma'] = utils.median_pairwise_distances(data)
        gk = kernel_gen.gaussian_kernel()
        # gk = kernel_gen.polynomial_kernel()
        K = gk.compute_matrix(data, self.params)
        self.embedder = solvers.embedder(2.56e-16, 800, True)
        self.kernel_sys = self.embedder.kernel_sys(K)

        label_mask = np.array([0])
        self.quad_eig_sys = self.embedder.sph_cl_var_term_eig_sys(self.kernel_sys)
        self.quad_eig_sys_original = copy(self.quad_eig_sys)
        if len(self.control_point_indices) == 0:
            placement_mask = np.array([0])
        else:
            placement_mask = np.array(self.control_point_indices)
        self.const_mu = self.embedder.const_nu(self.params, placement_mask, self.kernel_sys)
        self.update_control_points(points)
        self.finished_relocating()
        if len(self.Y) == 0:
            pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, label_mask, np.ones((1,2)), self.kernel_sys, self.params, 1e-20)
        else:
            for i in range(len(self.control_point_indices)):
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, self.control_point_indices[i], self.const_mu)
            pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
        self.projection_matrix = pca_dirs.T
        self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
        self.parent.update_status_bar("Constrained KPCA Embedding.")


    def get_embedding(self, X=None):
        if set(self.control_point_indices) != self.old_control_point_indices:
            self.pca_projection = self.finished_relocating()
        self.old_control_point_indices = set(self.control_point_indices)
        return self.pca_projection.T


    # this seems to be responsible for doing the actual calculation fo the directions
    def finished_relocating(self):
        if len(self.control_point_indices) > 0:
            directions = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
            self.pca_projection = self.kernel_sys[0].dot(directions)
            self.projection_matrix = directions.T
        return self.pca_projection


    # points seems to be a dictionary of pointindices and their xy coordinates
    # this seems to only be responible for adjustments when control points are added or removed
    def update_control_points(self, points):
        super(cPCA, self).update_control_points(points)
        if len(self.control_point_indices) > len(self.old_control_point_indices):
                selected_point = self.parent.selected_point
                if selected_point == None:
                    selected_point = (list(set(self.control_point_indices) - set(self.old_control_point_indices)))[0]
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, selected_point, self.const_mu)
                directions = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
                self.pca_projection = self.kernel_sys[0].dot(directions)
        elif len(self.control_point_indices) < len(self.old_control_point_indices):
            self.quad_eig_sys = copy(self.quad_eig_sys_original)
            for i in range(len(self.control_point_indices)):
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, self.control_point_indices[i], self.const_mu)
            if len(self.control_point_indices) == 0:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, np.array([0]), np.ones((1,2)), self.kernel_sys, self.params, 1e-20)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
            else:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
        self.old_control_point_indices = set(self.control_point_indices)
        self.finished_relocating()

        if self.has_ml_cl_constraints:
            self.augment_control_points(self.get_embedding().T)
            self.quad_eig_sys = copy(self.quad_eig_sys_original)
            for i in range(len(self.control_point_indices)):
                self.quad_eig_sys = self.embedder.sph_cp_quad_term_eig_sys(self.kernel_sys, self.quad_eig_sys, self.control_point_indices[i], self.const_mu)
            if len(self.control_point_indices) == 0:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, np.array([0]), np.ones((1,2)), self.kernel_sys, self.params, 1e-20)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)
            else:
                pca_dirs = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
                self.pca_projection = self.kernel_sys[0].dot(pca_dirs)

        
        



class MLE(Embedding):
    def __init__(self, data, points, parent):
        self.data = data
        self.control_points = []
        self.control_point_indices = []
        self.old_control_point_indices = []
        self.parent = parent
        self.X = None
        self.Y = None
        self.projection_matrix = None
    
        self.ml = []
        self.cl = []
        self.has_ml_cl_constraints = False

        pca = decomposition.PCA(n_components=2)
        pca.fit(self.data)
        self.M_base = pca.components_ # init M with PCA[1,2]
        self.M = self.M_base
        self.Psi_base = np.cov(self.data.T)
        self.sigma = 0.1*abs(np.min(self.Psi_base))
        self.Psi = self.Psi_base
        self.update_M_matrix()
        self.update_Psi_matrix()
        self.name = "MLE"
        self.is_dynamic = True 
        self.probabilities = None

        self.update_control_points(points)

    def update_Psi_matrix(self):
        Y = self.data[self.control_point_indices].T
        W = np.array(self.control_points).T
        # print "M  :", self.M_base.shape
        # print "X_m:", Y.shape
        # print "Psi:", self.Psi_base.shape
        # print "Y_m:", W.shape
        if len(self.control_point_indices) == 0:
            self.Psi = self.Psi_base
        else:
            self.Psi = self.Psi_base - self.Psi_base.dot(Y).dot(np.linalg.pinv(Y.T.dot(self.Psi_base).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi_base)


    def update_M_matrix(self):
        Y = self.data[self.control_point_indices].T
        W = np.array(self.control_points).T
        if len(self.control_point_indices) == 0:
            self.M = self.M_base
        else:
            # print "M  :", self.M_base.shape
            # print "X_m:", Y.shape
            # print "Psi:", self.Psi.shape
            # print "Y_m:", W.shape
            #self.M = self.M_base + (W - self.M_base.dot(Y)).dot(np.linalg.pinv(Y.T.dot(self.Psi).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi)
            self.M = self.M_base + (W - self.M_base.dot(Y)).dot(np.linalg.pinv(Y.T.dot(self.Psi_base).dot(Y) + self.sigma*np.eye(len(Y[0])))).dot(Y.T).dot(self.Psi_base)


    def get_embedding(self, X=[]):
        if (isinstance(X, list) and not X) or (isinstance(X, np.ndarray) and X.size == 0):
            # Handle empty list
            X=self.data.T
        self.projection_matrix = self.M
        return self.M.dot(X)
    

    def update_control_points(self, points):
        super(MLE, self).update_control_points(points)
        if set(self.control_point_indices) == self.old_control_point_indices:
            self.update_M_matrix()
        else:
            self.update_M_matrix()
            self.update_Psi_matrix()
        self.old_control_point_indices = set(self.control_point_indices)
        if self.has_ml_cl_constraints:
            self.augment_control_points(self.get_embedding().T)
            self.update_M_matrix()
            self.update_Psi_matrix()



class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))#, dtype='float16'
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(Layer):

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj1 = Dense(intermediate_dim, activation="relu")
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj1(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    
class Decoder(Layer):

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj2 = Dense(intermediate_dim, activation="relu")
        self.dense_output = Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj2(inputs)
        return self.dense_output(x)
    
class VariationalAutoEncoder(keras.Model):

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        return reconstructed, z_mean, z_log_var

class VariationalAutoencoderEmbedding(Embedding):
    def __init__(self, data: np.ndarray, points, verbose, parent):
        if tf.test.gpu_device_name() != '/device:GPU:0':
            print('WARNING: GPU device not found.')
        else:
            print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

        self.data = data
        self.verbose = verbose
        self.parent = parent
        
        try:
            self.w = PopupSlider(self.parent, 'Enter desired frame-rate (default is 30):', default=30, minimum=1, maximum=100)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 30
            self.frame_rate = num
        except Exception as e:
            msg = "It seems like something went wrong in the parameter selection"
            print(e)
            QMessageBox.about(parent, "Embedding error", msg) 

        self.original_dim = self.data.shape[1]
        self.intermediate_dim = self.original_dim // 2
        self.latent_dim = 2
        self.beta = 1
        self.rho = 10

        self.control_point_indices = []
        self.X = None
        self.Y = None

        self.name = "vae"
        self.is_dynamic = True 

        self.model = VariationalAutoEncoder(self.original_dim, self.intermediate_dim, self.latent_dim)
        self.optimizer = keras.optimizers.Adam()

        self.is_trained = False
        self.epochs = 20

        try:
            self.w = PopupSlider(self.parent, 'Enter desired batch-size (default is 50):\n(Higher batchsize will make the embedding more stable, \nlower batchsize will make it adapt more readily to changes)', default=50, minimum=1, maximum=200)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 50
            self.batch_size = num
        except Exception as e:
            msg = "It seems like something went wrong in the parameter selection"
            print(e)
            QMessageBox.about(parent, "Embedding error", msg) 

        self.finished_iterations = 0

        self.parent.update_status_bar("Calculating initial VAE Embedding.")

        self._prepare_dataset()

        self.train()

        #self.epochs = 1
        self.is_trained = True

        self.parent.update_status_bar("Interactive VAE Embedding.")

    def _prepare_dataset(self):
        control_point_indices_int = tf.cast(self.control_point_indices, tf.int32)
        control_points = tf.gather(self.data, control_point_indices_int)
        
        mask = tf.logical_not(tf.reduce_any(tf.equal(tf.range(tf.shape(self.data)[0])[:, None], control_point_indices_int), -1))
        rest_dataset = tf.boolean_mask(self.data, mask)

        self.control_points = control_points
        self.rest_dataset = rest_dataset

        self.dataset = tf.data.Dataset.from_tensor_slices(self.rest_dataset)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.map(lambda x: tf.concat([self.control_points, x], axis=0))
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(self, x_batch_train, cp_indices, cp_values):
        with tf.GradientTape() as tape:
            (reconstructed, z_mean, z_log_var) = self.model(x_batch_train, training=True)

            reconstruction_loss = mse(x_batch_train, reconstructed)
            reconstruction_loss *= self.original_dim

            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

            if cp_indices is not None and cp_values is not None:
                control_points_loss = mse(z_mean[0:len(cp_indices)], tf.cast(cp_values, dtype=tf.float32))#16
                control_points_loss *= self.latent_dim
                vae_loss = K.mean(reconstruction_loss + self.beta * kl_loss) + self.rho * K.mean(control_points_loss)
            else:
                control_points_loss = 0
                vae_loss = K.mean(reconstruction_loss + self.beta * kl_loss)

        grads = tape.gradient(vae_loss, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return vae_loss

    def train(self):

        self.finished_iterations = 0
        break_flag = False

        start_time = tf.timestamp()

        for epoch in range(self.epochs):
            if self.verbose:
                print("\nStart of epoch %d" % (epoch,))

            # Create start event
            total_loss = 0
            num_batches = 0

            # Custom batching within the training loop
            for x_batch_train in self.dataset:
                loss = self.train_step(x_batch_train, self.control_point_indices, self.Y)
                total_loss += loss
                num_batches += 1
                
                # Record end event
                current_time = tf.timestamp()
                elapsed_time = (current_time - start_time).numpy() * 1000  # Convert to milliseconds

                if elapsed_time > (1000 / self.frame_rate) and self.is_trained:
                    if self.verbose:
                        print("Broke off after %d iterations" % num_batches)
                        print("Elapsed time: %f" % elapsed_time)
                    break_flag = True
                    break

            if self.verbose:
                avg_loss = total_loss / num_batches
                print("Average loss for epoch %d: %.4f" % (epoch, avg_loss))

            if self.is_trained:
                self.finished_iterations += num_batches

            if break_flag:
                break

    def update_control_points(self, points) -> None:


        if len(points.items()) == 0:
            self.parent.update_status_bar("Calculating initial VAE Embedding.")

        self.Y = []
        cp_indices = []
        for i, coords in points.items():
            cp_indices.append(i)
            self.Y.append(coords)

        if len(self.Y) == 0:
            self.Y = None
        else:
            self.Y = np.array(self.Y, dtype=np.float64)

        self.X = self.data[cp_indices]

        if set(self.control_point_indices) != set(cp_indices):
            self.control_point_indices = cp_indices
            self._prepare_dataset()
        
        self.train()

        if len(points.items()) == 0:
            self.parent.update_status_bar("Interactive VAE Embedding.")

    def get_iteration_count(self):
        return self.finished_iterations

    def get_embedding(self) -> tuple:
        x_test = self.data
        (_, z_mean, _) = self.model(x_test, training=False)
        
        x_embedding = z_mean[:, 0].numpy()
        y_embedding = z_mean[:, 1].numpy()

        embedding = np.vstack((x_embedding, y_embedding))

        return embedding