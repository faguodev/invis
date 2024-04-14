#!/usr/bin/python
import numpy as np
from copy import copy
import cupy as cp
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
from cupyx.profiler import benchmark

class PopupSlider(QDialog):
    def __init__(self, label_text, default=4, minimum=1, maximum=20):
        QWidget.__init__(self)
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
        self.button.pressed.connect(self.handleButton)

        layout = QGridLayout(self)
        layout.addWidget(name_label      , 1, 1, 1, 4, Qt.AlignLeft)
        layout.addWidget(self.slider     , 2, 1, 2, 1, Qt.AlignLeft)
        layout.addWidget(self.value_label, 2, 2, 2, 2, Qt.AlignCenter)
        layout.addWidget(self.button     , 2, 4, 2, 4, Qt.AlignRight)

        self.setWindowTitle('Parameter choice')


    def slider_changed(self):
        val = self.slider.value()
        self.value_label.setText('%d' %val)
        self.slider_value = val
 

    def handleButton(self):
        self.hide()


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
            self.w = PopupSlider('Enter number of neighbors to consider (default is 4):')
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
            self.w = PopupSlider('Enter number of neighbors to consider (default is 4):')
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
            self.w = PopupSlider('Enter perplexity (default is 30):', default=30, minimum=1, maximum=100)
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
    def __init__(self, data, points, parent):
        """
        Initialize constrained Kernel PCA embedding.

        Args:
            data (pd.DataFrame): Input data for the embedding.
        """
        self.data = data
        self.control_points = []
        self.control_point_indices = []
        self.X = None
        self.Y = None
        self.projection_matrix = cp.zeros((2, self.data.shape[0]))
        
        # control points
        self.cp_selector_m_by_n = None
        self.cp_selector_n_by_n = None
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
            m, ok = QInputDialog.getText(parent, 'Metric', 'Enter number of the desired kernel:\n1) Gaussian (Default)\n2) Polynomial\n3) Cosine Linear\n4) Isomap\n5) Pseudoinverse of Laplacian-Kernel\n6) Pseudoinverse of Laplacian-Kernel (using k-NN)\n7) Linear')
            
            kernel = kernel_gen.gaussian_kernel()
            params = {'sigma' : utils.median_pairwise_distances(data)}

            match m:
                case '2':
                    kernel = kernel_gen.polynomial_kernel()
                    degree, ok = QInputDialog.getText(parent, 'Degree', 'Enter degree of polynomial kernel (default is 1):')
                    if degree == '':
                        degree = 1
                    params['degree'] = float(degree)
                    print("Degree: " + str(degree))
                case '3':
                    kernel = kernel_gen.cos_linear_kernel()
                    params['degree'] = 1
                case '4':
                    kernel = kernel_gen.isomap_kernel()
                case '5':
                    kernel = kernel_gen.pinv_laplacian_kernel()
                    epsilon, ok = QInputDialog.getText(parent, 'Epsilon', 'Enter epsilon for Laplacian kernel (default is 0.5):')
                    if epsilon == '':
                        epsilon = 0.5
                    params['epsilon'] = float(epsilon)
                    print("Epsilon: " + str(epsilon))
                case '6':
                    kernel = kernel_gen.pinv_k_nn_laplacian_kernel()
                    k, ok = QInputDialog.getText(parent, 'Degree', 'Enter k for k-NN (default is 3):')
                    if k == '':
                        k = 3
                    params['k'] = int(k)
                    params['sigma'] = utils.median_pairwise_distances(data)
                case '7':
                    kernel = kernel_gen.linear_kernel()
                case _:
                    kernel = kernel_gen.gaussian_kernel()
        except:
            msg = "It seems like something went wrong in the parameter selection"
            QMessageBox.about(parent, "Embedding error", msg) 

        print("before compute kernel matrix")

        self.K = kernel.compute_matrix(data, params)

        print("after compute kernel matrix")

        self.K_sqrt, self.K_sqrt_inv = utils.construct_kernel_sys(self.K)

        # Transfer to GPU
        self.K = cp.asarray(self.K)
        self.K_sqrt = cp.asarray(self.K_sqrt)
        self.K_sqrt_inv = cp.asarray(self.K_sqrt_inv)

        # compute W
        H = cp.eye(len(self.K)) - (1 / len(self.K)) * cp.ones((len(self.K), len(self.K)))
        self.W = (1/self.n) * H

        v1 = cp.random.rand(self.n)
        self.v1 = v1 / cp.linalg.norm(v1)

        v2 = cp.random.rand(self.n)
        self.v2 = v2 / cp.linalg.norm(v2)

        self.C_var = self.K_sqrt @ self.W @ self.K_sqrt
        self.C_cp = 0
        self.C_ml_cml = 0

        self.cp_const_mu = 10
        self.cl_ml_const_mu = 0.1 #0.1 seems good (upper bound) choice for cl
        self.orth_mu = 10
        
        self.alpha_1 = None
        self.alpha_2 = None

        self.max_iter = 10 ** 100

        print(benchmark(self.update_control_points, (points,), n_repeat=1, n_warmup=0))

        self.max_iter = 100

    def update_must_and_cannot_link(self, ml, cl):
        self.ml = [constraint for constraint in ml if isinstance(constraint, set) and len(constraint) == 2]
        self.cl = [constraint for constraint in cl if isinstance(constraint, set) and len(constraint) == 2]
        self.max_iter = 10 ** 100

        print("ml", self.ml)
        print("cl", self.cl)

        if (len(self.ml) > 0):
            i,j = tuple(self.ml[0])
            coord_i = (self.projection_matrix @ self.K)[:, i]
            coord_j = (self.projection_matrix @ self.K)[:, j]

            dist = cp.linalg.norm(coord_i - coord_j)
            print("Distance between ml points: " + str(dist))

        if(self.ml != self.old_ml_constraints or self.cl != self.old_cl_constraints):
            self.old_ml_constraints = self.ml
            self.old_cl_constraints = self.cl
            self._benchmark_update_ml_cl_params()
            self._benchmark_iteration()
            self._benchmark_construct_projection_matrix()

        
        if (len(self.ml) > 0):
            i,j = tuple(self.ml[0])
            coord_i = (self.projection_matrix @ self.K)[:, i]
            coord_j = (self.projection_matrix @ self.K)[:, j]

            dist = cp.linalg.norm(coord_i - coord_j)
            print("Distance between ml points: " + str(dist))

        self.max_iter = 100

    def _benchmark_update_ml_cl_params(self):
        if (len(self.ml) > 0) or (len(self.cl) > 0):
            # construct laplacian matrix and weigh it
            self.C_ml_cml = (self.cl_ml_const_mu / (len(self.ml) + len(self.cl))) * utils.construct_ml_cl_laplacian_matrix(self.n, self.ml, self.cl)
        else:
            self.C_ml_cml = 0

    # points seems to be a dictionary of pointindices and their xy coordinates
    def update_control_points(self, points):
        self.control_points = []
        self.control_point_indices = []
        for i, coords in points.items():
            self.control_point_indices.append(i)
            self.control_points.append(coords)
        self.X = self.data[self.control_point_indices]
        self.Y = cp.array(self.control_points)
        
        if len(self.control_point_indices) == 0:
            self.max_iter = 10 ** 100
        else:
            self.max_iter = 100

        self._benchmark_update_cp_params(points)
        self._benchmark_iteration()
        self._benchmark_construct_projection_matrix()

    def _benchmark_update_cp_params(self, points):
        if points is not None and set(self.control_point_indices) != set(self.old_control_point_indices):
            self.old_control_point_indices = self.control_point_indices
            self.cp_selector_m_by_n, self.cp_selector_n_by_n = utils.construct_cp_selector_matrices(self.n, self.control_point_indices)
            if(len(self.control_point_indices) > 0):
                self.C_cp = self.K_sqrt @ (self.cp_const_mu / len(self.control_point_indices) * self.cp_selector_n_by_n) @ self.K_sqrt
            else:
                self.C_cp = 0

    def _benchmark_iteration(self):
        self.alpha_1, self.v1 = self.solve_iteratively(None, 0, self.v1)
        self.alpha_2, self.v2 = self.solve_iteratively(self.alpha_1, 1, self.v2)

    def _benchmark_construct_projection_matrix(self):
        self.projection_matrix = cp.vstack((self.alpha_1, self.alpha_2))

    def solve_iteratively(self, alpha, dimension, v, verbose = 0):
        
        """
        Solve the constrained optimization problem iteratively.

        Args:
            alpha (cp.ndarray): The previous alpha vector.
            dimension (int): The dimension to solve for.
        """

        # Create start event
        # start_event = cp.cuda.Event()
        # end_event = cp.cuda.Event()

        # Record start event
        # start_event.record()

        C = self.C_var - self.C_cp - self.C_ml_cml
        if alpha is not None:
            temp = self.K_sqrt @ alpha
            C_orth = self.orth_mu * np.outer(temp, temp)
            C = C - C_orth
        

        # Record end event
        # end_event.record()

        # Wait for the end event to complete
        # end_event.synchronize()

        # Calculate elapsed time
        # elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)

        # print("Setting up marices " + str(dimension) + " " + str(elapsed_time))

        d = cp.zeros(len(self.K))

        if len(self.control_point_indices) > 0:
            Y_s = self.Y[:, dimension]
            d = -1 * self.cp_const_mu / len(self.control_point_indices) * Y_s.T @ self.cp_selector_m_by_n @ self.K_sqrt

        # Iteration takes very roughly 1 millisecond

        learning_rate = 7e-2
        beta1 = 0.95
        iteration = 0
        v_dw = cp.zeros(self.n)

        while iteration < self.max_iter:
            iteration += 1
            # Nesterov lookahead (Check Sign)
            v_lookahead = v + beta1 * v_dw

            # Compute gradient at lookahead position
            dw = C @ v_lookahead - d

            # Update velocities
            v_dw = beta1 * v_dw + learning_rate * dw

            # Update parameters
            v_new = v + v_dw
            v_new = v_new / cp.linalg.norm(v_new)

            if iteration % 100 == 0 and verbose == 1:
                print(f"Iteration {iteration} - Norm: {cp.linalg.norm(v_new - v)}")

            if cp.linalg.norm(v_new - v) < 1e-6:
                print(f"Converged after {iteration} iterations")
                break

            v = v_new

        alpha_s = self.K_sqrt_inv @ v
        return alpha_s, v
    
    # resposible for getting the already calculated embedding
    def get_embedding(self, X=None):
        return cp.asnumpy(self.projection_matrix @ self.K)






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
        self.parent.status_text.setText("Done, calculating Gaussean kernel.")

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
