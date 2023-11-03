#!/usr/bin/python
#from sip import setapi
#setapi('QString', 2)
#setapi('QVariant', 2)

try:
    import gr
    import os 
    os.environ['MPLBACKEND'] = "module://gr.matplotlib.backend_gr"
except:
    print("Not using accelerated matplotlib backend")
from Gui import MainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Dataset import Dataset
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from Embedder import *
from itemset_mining import ItemsetMiner




class InVis(MainWindow):
    """  """
    def __init__(self, pandasdata=''):
        super(InVis, self).__init__()
        self.data = None
        self.mask = None
        self.toggle_key = False
        self.offset = np.array([0.0, 0.0])
        self.control_points = None
        self.dummies = []
        if len(sys.argv) > 1:
            try:
                self.load_file(filename=sys.argv[1])
            except Exception as e:
                print(e)
                pass
        if type(pandasdata) == type(pd.DataFrame([])):
            """ Load a dataset from a file """
            self.show_singletons = False
            self.point_representation = True
            self.show_center_point = False
            self.control_points = {}
            self.searched_results = []
            self.highlighted = []
            self.splits = []
            self.opacity = 0.4
            self.show_search_as_color == False
            self.discretization_type = '1'
            self.selected_index = None
            self.only_show_names = True
            self.framesize = 3
            self.xlim=[-3,3]
            self.ylim=[-3,3]
            self.press = False
            self.boost_label_color = False
            self.ml_cl_state = None
            self.ml_cl_pending = False
            self.must_link = []
            self.show_singletons = False
            self.cannot_link = []
            self.pairwise_link_marker = 0
            self.ml_cl_available = True
            self.ml_cl_index = None
            self.show_links = True
            self.info_request = False
            self.cp_select_request = False
            self.lasso_request = False
            self.center_ind = None
            self.control_click = False
            self.lastevent = None	
            self.data_clipped = False
            self.color_scheme = "Blues"
            self.axes.set_aspect('auto')
            self.data = Dataset()

            self.clear()
            self.data.load_pandas_dataframe(pandasdata)
            self.status_text.setText("Loaded Pandas DataFrame")
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (MLE)')
            self.mask = np.ones(len(self.data.data)).astype(bool)
            self.fill_attribute_list(self.data.attribute_names)
            self.label_text_field.setText(self.data.label_name)
            self.embedding_algorithm = MLE(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.label_updated()
            self.point_size = np.ones(len(self.data.data))*60
            # self.splits defines a list of values where for each attribute if a value is above the value defined here, it is decretizied to be True, False otherwise
            self.splits = list(np.zeros(len(self.data.attribute_names))) 
            if not self.canvas_connected:
                self.connect_canvas()

        
    @pyqtSlot()
    def load_file(self, filename=None):
        """ Load a dataset from a file """
        self.show_singletons = False
        self.point_representation = True
        self.show_center_point = False
        self.control_points = {}
        self.searched_results = []
        self.highlighted = []
        self.splits = []
        self.discretization_type = '1'
        self.selected_index = None
        self.only_show_names = True
        self.framesize = 3
        self.xlim=[-3,3]
        self.ylim=[-3,3]
        self.press = False
        self.boost_label_color = False
        self.ml_cl_state = None
        self.ml_cl_pending = False
        self.must_link = []
        self.cannot_link = []
        self.pairwise_link_marker = 0
        self.ml_cl_available = True
        self.ml_cl_index = None
        self.show_links = True
        self.info_request = False
        self.cp_select_request = False
        self.lasso_request = False
        self.center_ind = None
        self.control_click = False
        self.lastevent = None
        self.data_clipped = False
        self.color_scheme = "Blues"
        self.axes.set_aspect('auto')
        self.data = Dataset()
        if filename == None:
            filename, _ = QFileDialog.getOpenFileName(self, 'Open a data file', '.', 'All Files (*.*)')
        if filename:
            self.clear()
            self.data.read_in_data(filename)
            self.status_text.setText("Loaded " + filename)
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (MLE)')
            self.mask = np.ones(len(self.data.data)).astype(bool)
            self.fill_attribute_list(self.data.attribute_names)
            self.label_text_field.setText(self.data.label_name)
            self.embedding_algorithm = MLE(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.label_updated()
            self.point_size = np.ones(len(self.data.data))*60
            self.splits = list(np.zeros(len(self.data.attribute_names)))
            if not self.canvas_connected:
                self.connect_canvas()


    def export_projection_matrix(self):
        """ Export the projection matrix """
        if self.data is not None:
            filename, _ = QFileDialog.getSaveFileName(self, "Export projection matrix", "")
            if filename:
                try:
                    np.savetxt(filename, self.embedding_algorithm.projection_matrix, delimiter=',', fmt='%10.5f')
                except:
                    pass


    def export_freq(self):
        """ Export frequent patterns """
        if self.data != None:


            # print(len(self.data.data))
            # write self.data.data to ./data/data-test.txt
            """ import csv

            f = open("./data/data-test.csv", "w")
        
            writer = csv.writer(f)
            for i in range(len(self.data.data)):

                writer.writerow(self.data.data[i]) """


            self.update_status_bar('Mining & exporting patterns')
            representation, ok = QInputDialog.getText(self, 'Pattern representation', 'Represent the patterns via extention <e>, or intention <i>? (default is intention)')
            if not representation in ['e', 'i']:
                representation = 'i'
            top_k, ok = QInputDialog.getText(self, 'Amount', 'Number of patterns to mine (default is 1000)')
            try:
                top_k = int(top_k)
            except:
                top_k = 1000
            filename, _ = QFileDialog.getSaveFileName(self, "Export to", "")
            if filename:
                if self.discretization_type == None:
                    self.generate_discretization_splits()
                iMiner = ItemsetMiner(self.data, self.splits, 1, 1, 50, range(len(self.data.data)), self.show_ignored_attributes)
                transactions = iMiner.build_itemsets(constraint='0')
                print(len(transactions))
                patterns = iMiner.get_frequent_itemsets(transactions, 's', top_k=top_k)
                if representation == 'i':
                    out = iMiner.export_patterns(patterns, 'Support')
                else:
                    out = iMiner.export_patterns_extension_representation(patterns, 'Support')
                f = open(filename, 'w')
                f.write(out)
                f.close()
                self.update_status_bar('Done mining & exporting patterns')


    def export_closed(self):
        """ Export closed frequent patterns """
        if self.data != None:
            self.update_status_bar('Mining & exporting patterns')
            representation, ok = QInputDialog.getText(self, 'Pattern representation', 'Represent the patterns via extention <e>, or intention <i>? (default is intention)')
            if not representation in ['e', 'i']:
                representation = 'i'
            top_k, ok = QInputDialog.getText(self, 'Amount', 'Number of patterns to mine (default is 500)')
            try:
                top_k = int(top_k)
            except:
                top_k = 500
            filename, _ = QFileDialog.getSaveFileName(self, "Export to", "")
            if self.discretization_type == None:
                self.generate_discretization_splits()
            iMiner = ItemsetMiner(self.data, self.splits, 1, 1, 50, range(len(self.data.data)), self.show_ignored_attributes)
            transactions = iMiner.build_itemsets(constraint='0')
            patterns = iMiner.get_frequent_itemsets(transactions, 'c', top_k=top_k)
            if representation == 'i':
                out = iMiner.export_patterns(patterns, 'Support')
            else:
                out = iMiner.export_patterns_extension_representation(patterns, 'Support')
            f = open(filename, 'w')
            f.write(out)
            f.close()
            self.update_status_bar('Done mining & exporting patterns')


    def export_subgroup(self):
        """ Export subgroup """
        if self.data != None:
            self.update_status_bar('Mining & exporting patterns')
            representation, ok = QInputDialog.getText(self, 'Pattern representation', 'Represent the patterns via extention <e>, or intention <i>? (default is intention)')
            if not representation in ['e', 'i']:
                representation = 'i'
            top_k, ok = QInputDialog.getText(self, 'Amount', 'Number of patterns to mine (default is 100)')
            try:
                top_k = int(top_k)
            except:
                top_k = 100
            filename, _ = QFileDialog.getSaveFileName(self, "Export to", "")
            if self.discretization_type == None:
                self.generate_discretization_splits()
            iMiner = ItemsetMiner(self.data, self.splits, 2, 1, 5, range(len(self.data.data)), self.show_ignored_attributes)
            patterns = iMiner.get_subgroups(top_k=top_k)
            if representation == 'i':
                out = iMiner.export_patterns(patterns, 'Quality')
            else:
                out = iMiner.export_patterns_extension_representation(patterns, 'Quality')
            f = open(filename, 'w')
            f.write(out)
            f.close()
            self.update_status_bar('Done mining & exporting patterns')


    def export_relevant(self):
        """ Export relevant subgroups """
        if self.data != None:
            self.update_status_bar('Mining & exporting patterns')
            representation, ok = QInputDialog.getText(self, 'Pattern representation', 'Represent the patterns via extention <e>, or intention <i>? (default is intention)')
            if not representation in ['e', 'i']:
                representation = 'i'
            top_k, ok = QInputDialog.getText(self, 'Amount', 'Number of patterns to mine (default is 100)')
            try:
                top_k = int(top_k)
            except:
                top_k = 100
            filename, _ = QFileDialog.getSaveFileName(self, "Export to", "")
            if self.discretization_type == None:
                self.generate_discretization_splits()
            iMiner = ItemsetMiner(self.data, self.splits, 1, 1, 50, range(len(self.data.data)), self.show_ignored_attributes)
            patterns = iMiner.get_delta_relevant_itemsets(1, 'absolute', top_k=top_k)
            if representation == 'i':
                out = iMiner.export_patterns(patterns, 'Quality')
            else:
                out = iMiner.export_patterns_extension_representation(patterns, 'Quality')
            f = open(filename, 'w')
            f.write(out)
            f.close()
            self.update_status_bar('Done mining & exporting patterns')



def start_InVis(pandasdata=None):
    app = QApplication(sys.argv)
    invis = InVis(pandasdata=pandasdata)
    invis.show()
    app.exec_()
    return {'selected points':invis.lassoed_points, 'embeding':invis.embedding}



if __name__ == "__main__":
    print("Matplotlib rendereing Backend is --" + pl.get_backend() + "--")
    invis = start_InVis()
