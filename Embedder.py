#!/usr/bin/python
import numpy as np
from copy import copy
from sklearn import decomposition
from collections import defaultdict
# from scipy.spatial import distance
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# explicitly imported "hidden imports" for pyinstaller
#from sklearn.utils import weight_vector, lgamma
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap, MDS, TSNE, LocallyLinearEmbedding

# Dinos solver
import cpca.solvers as solvers
import cpca.kernel_gen as kernel_gen
import cpca.utils as utils
import pandas as pd

try:
    from sklearn.utils.sparsetools import _graph_validation
    from sklearn.neighbors import typedefs
except:
    pass


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
                lle = LocallyLinearEmbedding(n_neighbors=int(num), out_dim=2)
            except:
                lle = LocallyLinearEmbedding(n_neighbors=int(num), n_components=2)
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
            iso = Isomap(n_neighbors=num, n_components=2)

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
            tsne = TSNE(n_components=2, random_state=0, perplexity=num, metric=metric)
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
        mds = MDS(n_components=2, dissimilarity='precomputed')
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

        # r = radius? slv_mode: type of solver, 
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
        self.pca_projection = self.kernel_sys[0].dot(pca_dirs)


    def get_embedding(self, X=None):
        if set(self.control_point_indices) != self.old_control_point_indices:
            self.pca_projection = self.finished_relocating()
        self.old_control_point_indices = set(self.control_point_indices)
        return self.pca_projection.T


    def finished_relocating(self):
        if len(self.control_point_indices) > 0:
            directions = self.embedder.soft_cp_mode_directions(self.quad_eig_sys, self.control_point_indices, self.Y, self.kernel_sys, self.params, self.const_mu)
            self.pca_projection = self.kernel_sys[0].dot(directions)
        return self.pca_projection


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
