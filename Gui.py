#!/usr/bin/python
import os
from collections import defaultdict
from copy import copy
from os import path
from re import findall

import dill
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as pl
import numpy as np
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Lasso
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy.spatial import distance

import wordclouds
from Embedder import *
from Embedder import PopupSlider
from itemset_mining import ItemsetMiner


class MainWindow(QMainWindow):
    """ The main GUI class """
    def __init__(self, parent=None, file_name=""):
        super(MainWindow, self).__init__(parent)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.cwd = os.path.join(self.cwd,"ressources")
        self.embedding = None
        self.embedding_algorithm = None
        self.framesize = 3
        self.xlim = [-3,3]
        self.ylim = [-3,3]
        self.pick_sensitivity = 5
        self.show_origin = False
        self.color_scheme = "Blues"
        self.colors = None
        self.opacity = 0.4
        self.boost_label_color = False
        self.ml_cl_state = None
        self.ml_cl_pending = False
        self.must_link_original = []
        self.cannot_link_original = []
        self.must_link = []
        self.cannot_link = []
        self.pairwise_link_marker = 0
        self.ml_cl_available = True
        self.ml_cl_index = None
        self.show_links = True
        self.uncertainty_coloring_flag = False
        self.press = False
        self.show_search_as_color = False
        self.show_singletons = False
        self.control_click = False
        self.shift_click = False
        self.selected_point = None
        self.only_show_names = True
        self.searched_results = []
        self.info_requests = []
        self.highlighted = []
        self.splits = []
        self.show_ignored_attributes = False
        self.control_points = {}
        self.canvas_connected = False
        self.all_selected = True
        self.discretization_type = '1'
        self.show_singletons = False
        self.point_representation = True
        self.show_center_point = False
        self.data_clipped = False
        self.unclipped_data = None
        self.unclipped_original_data = None
        self.unclipped_instance_names = None
        self.saved_control_points = {}
        self.point_size = 60
        self.point_size_is_set_variable = False
        self.dummies = []

        self.info_request = False
        self.cp_select_request = False
        self.lasso_request = False
        self.image_displayer = self.create_tag_cloud

        self.scatter_plot = None
        self.lassoLock = False
        self.lassoed_points = []
        self.path = None
        self.center_ind = None
        self.tag_elements = []
        self.control_point_color = {"RdYlGn":'b', "Spectral":'k', "Blues":'m', "hot":'b'}

        self.setWindowTitle('InVis')
        self.series_list_model = QStandardItemModel()
        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        self.canvas.draw()


    def create_main_frame(self):
        """ Setup and create the main GUI window """
        self.main_frame = QWidget()
        
        # create a matplotlib figure
        plot_frame = QWidget()
        self.dpi = 100
        self.fig = Figure()
        self.fig.subplots_adjust(left=0.001, bottom=0.001, right=0.999, top=0.998)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMaximumWidth(1231324)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy( Qt.ClickFocus )
        self.canvas.setFocus()
        
        # configure the figure
        self.axes = self.fig.add_subplot(111)
        self.axes.set_aspect('auto')
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_visible(False)
        self.axes.set_xlim([-self.framesize, self.framesize])
        self.axes.set_ylim([-self.framesize, self.framesize])
        
        # configure the attribute list
        self.series_list_view = QListView()
        self.series_list_view.setMaximumWidth(310)
        self.series_list_view.setMinimumWidth(310)
        self.series_list_view.setModel(self.series_list_model)
        
        # configure the label field in the GUI
        self.target_label = QLabel("Attribute used for coloring:")
        self.target_label.setMaximumWidth(310)
        self.target_label.setMinimumWidth(310)
        self.label_text_field = QLineEdit("")
        self.label_text_field.setMaximumWidth(310)
        self.label_text_field.setMinimumWidth(310)

        # configure the colorize by next label button in the GUI
        self.next_label_button = QPushButton()
        self.next_label_button.setToolTip("select the next attribute as label")
        self.next_label_button.setIcon(QIcon(os.path.join(self.cwd,"next.png")))
        self.next_label_button.setMaximumWidth(50)
        
        # configure the search field in the GUI
        self.search_text_field = QLineEdit("", placeholderText="Search")
        self.search_text_field.setMaximumWidth(310)

        # configure the colorize by next label button in the GUI
        self.select_search_button = QPushButton()
        self.select_search_button.setToolTip("select the search result")
        self.select_search_button.setIcon(QIcon(os.path.join(self.cwd,"lasso.png")))
        self.select_search_button.setMaximumWidth(50)

        self.search_text_field.setMinimumWidth(310)

        # configure the (de)select all button in the GUI
        self.select_button = QPushButton("(De-)&Select all", shortcut="Ctrl+S")
        self.select_button.setMaximumWidth(310)

        # configure the update button
        self.update_button = QPushButton("Update", shortcut="Ctrl+U")
        self.update_button.setMaximumWidth(310)

        # configure the info button in the GUI
        self.info_button = QPushButton(shortcut="Ctrl+I")
        self.info_button.setToolTip("get information on data record <Right-click>, or <Ctrl+I>")
        self.info_button.setIcon(QIcon(os.path.join(self.cwd,"annotation.png")))
        self.info_button.setMaximumWidth(50)

        # configure the select button in the GUI
        self.cp_select_button = QPushButton()
        self.cp_select_button.setToolTip("(de-)select a control point <Middle-click>")
        self.cp_select_button.setIcon(QIcon(os.path.join(self.cwd,"select.png")))
        self.cp_select_button.setMaximumWidth(50)

        # configure the select button in the GUI
        self.auto_select_button = QPushButton(shortcut="Ctrl+N")
        self.auto_select_button.setToolTip("automatically select a control point (only MLE) <Ctrl+N>")
        self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select.png")))
        self.auto_select_button.setMaximumWidth(50)

        # configure the lasso button in the GUI
        self.lasso_button = QPushButton()
        self.lasso_button.setToolTip("lasso select data records <Ctrl+hold-left-mouse-button>")
        self.lasso_button.setIcon(QIcon(os.path.join(self.cwd,"lasso.png")))
        self.lasso_button.setMaximumWidth(50)

        # configure the cut button in the GUI
        self.cut_button = QPushButton(shortcut="Ctrl+X")
        self.cut_button.setToolTip("cut out selection and re-embed <Ctrl+X>")
        self.cut_button.setIcon(QIcon(os.path.join(self.cwd,"cut.png")))
        self.cut_button.setMaximumWidth(50)

        # configure the clear button in the GUI
        self.clear_button = QPushButton(shortcut="Ctrl+C")
        self.clear_button.setToolTip("clear all annotations <Ctrl+C>")
        self.clear_button.setIcon(QIcon(os.path.join(self.cwd,"clear.png")))
        self.clear_button.setMaximumWidth(50)

        # configure the must-/cannot-link button in the GUI
        self.ml_cl_button = QPushButton()
        self.ml_cl_button.setToolTip("Enable pairwise must-link and cannot-link constraints")
        self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl.png")))
        self.ml_cl_button.setMaximumWidth(50)

        # configure the layout of all the elements
        grid = QGridLayout()
        # grid.setSpacing(20)
        grid.addWidget(self.canvas,               1,1,  8,30) # <--- Here the matplotlib figure gets build in!!!

        grid.addWidget(self.select_button,        1,31,  1, -1, Qt.AlignLeft) 
        grid.addWidget(self.update_button,        1,31,  1,-1, Qt.AlignRight) 
        grid.addWidget(self.series_list_view,     2,31,  1,-1, Qt.AlignRight)

        grid.addWidget(self.target_label,         5,31,  1,-1, Qt.AlignRight)
        grid.addWidget(self.label_text_field,     6,31,  1,-1, Qt.AlignRight)
        grid.addWidget(self.next_label_button,    6,32,  1,-1, Qt.AlignRight)
        grid.addWidget(self.search_text_field,    7,31,  1,-1, Qt.AlignRight)
        grid.addWidget(self.select_search_button, 7,32,  1,-1, Qt.AlignRight)

        grid.addWidget(self.info_button,          8,31,  1,1, Qt.AlignRight)
        grid.addWidget(self.cp_select_button,     8,32,  1,1, Qt.AlignRight)
        grid.addWidget(self.auto_select_button,   8,33,  1,1, Qt.AlignRight)
        grid.addWidget(self.ml_cl_button,         8,34,  1,1, Qt.AlignRight)
        grid.addWidget(self.lasso_button,         8,35,  1,1, Qt.AlignRight)
        grid.addWidget(self.cut_button,           8,36,  1,1, Qt.AlignRight)
        grid.addWidget(self.clear_button,         8,37,  1,1, Qt.AlignRight)

        self.main_frame.setLayout(grid)
        self.setCentralWidget(self.main_frame)


    def connect_canvas(self):
        """ Bind the mouse, key and GUI events to functions """
        self.canvas_connected = True
        # connect the matplotlib figure
        self.axes.figure.canvas.mpl_connect('pick_event', self.on_point_pick)
        self.axes.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.axes.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.axes.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.axes.figure.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.axes.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.axes.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        # connect the GUI buttons and textfields
        self.label_text_field.returnPressed.connect(self.label_updated)
        self.search_text_field.returnPressed.connect(self.highlight_search_results)
        self.next_label_button.clicked.connect(self.set_next_attribute_as_label)
        self.select_search_button.clicked.connect(self.select_search_result)
        self.select_button.clicked.connect(self.select_all_action)
        self.update_button.clicked.connect(self.recalculate_embedding)
        self.auto_select_button.clicked.connect(self.next_uncertain)
        self.info_button.clicked.connect(self.set_info_request)
        self.cp_select_button.clicked.connect(self.set_cp_select_request)
        self.ml_cl_button.clicked.connect(self.toggle_ml_cl_state)
        self.lasso_button.clicked.connect(self.set_lasso_request)
        self.cut_button.clicked.connect(self.toggle_data_filter)
        self.clear_button.clicked.connect(self.clear)
        # connect the attribute list's checkboxes
        self.series_list_model.itemChanged.connect(self.attributes_updated)

    def toggle_ml_cl_state(self):
        self.pairwise_link_marker = 0
        self.ml_cl_pending = False
        if self.ml_cl_available:
            try:
                if len(self.must_link[-1]) == 1:
                    self.must_link = self.must_link[:-1]
            except:
                pass
            try:
                if len(self.cannot_link[-1]) == 1:
                    self.cannot_link = self.cannot_link[:-1]
            except:
                pass
            if self.ml_cl_state == None:
                self.ml_cl_state = 'ML'
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml.png")))
            elif self.ml_cl_state == 'ML':
                self.ml_cl_state = 'CL'
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"cl.png")))
            elif self.ml_cl_state == 'CL':
                self.ml_cl_state = None
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl.png")))
            else:
                pass


    def set_info_request(self):
        self.info_request = not self.info_request
        if self.info_request == True:
            self.info_button.setIcon(QIcon(os.path.join(self.cwd,"annotation_active.png")))
        else:
            self.info_button.setIcon(QIcon(os.path.join(self.cwd,"annotation.png")))
             

    def set_cp_select_request(self):
        self.cp_select_request = not self.cp_select_request
        if self.cp_select_request == True:
            self.cp_select_button.setIcon(QIcon(os.path.join(self.cwd,"select_active.png")))
        else:
            self.cp_select_button.setIcon(QIcon(os.path.join(self.cwd,"select.png")))


    def set_lasso_request(self):
        self.lasso_request = not self.lasso_request
        if self.lasso_request == True:
            self.lasso_button.setIcon(QIcon(os.path.join(self.cwd,"lasso_active.png")))
        else:
            self.lasso_button.setIcon(QIcon(os.path.join(self.cwd,"lasso.png")))


    def create_menu(self):  
        """ Create the menu in the GUI """      
        self.file_menu = self.menuBar().addMenu("&File")
        load_file        = self.prepare_menu_entry("&Load dataset", shortcut="Ctrl+L", slot=self.load_file, tip="Load a file")
        #load_session     = self.prepare_menu_entry("&Load session", slot=self.load_session)
        #save_session     = self.prepare_menu_entry("&Save session", slot=self.save_session)
        #export_matrix    = self.prepare_menu_entry("Export projection matrix", slot=self.export_projection_matrix, tip="Export the projection matrix which renders the current embedding")
        self.add_menu_entry(self.file_menu, (load_file, None)) #load_session, save_session, export_matrix
        self.pattern_mining_menu = self.file_menu.addMenu("Export Patterns as dataset")
        frequent_menu            = self.prepare_menu_entry("Frequent patterns", slot=self.export_freq, tip="Export the TOP-1000 FREQUENT PATTERNS as a regular datafile that can be read by this tool")
        closed_menu              = self.prepare_menu_entry("Closed frequent patterns", slot=self.export_closed, tip="Export the TOP-500 CLOSED FREQUENT PATTERNS as a regular datafile that can be read by this tool")
        subgroup_menu            = self.prepare_menu_entry("Subgroups", slot=self.export_subgroup, tip="Export the TOP-100 SUBGROUPS as a regular datafile that can be read by this tool (NEEDS A BINARY LABEL)")
        relevant_menu            = self.prepare_menu_entry("1-relevant Subgroups", slot=self.export_relevant, tip="Export the TOP-100 RELEVANT SUBGROUPS as a regular datafile that can be read by this tool (NEEDS A BINARY LABEL)")
        self.add_menu_entry(self.pattern_mining_menu, (frequent_menu, closed_menu, subgroup_menu, relevant_menu))
        quit_action   = self.prepare_menu_entry("&Quit", slot=self.close, shortcut="Ctrl+Q", tip="Close the application")
        self.add_menu_entry(self.file_menu, (None, quit_action))


        self.edit_menu = self.menuBar().addMenu("&Edit")
        normalize_z           = self.prepare_menu_entry("Change to Z-Normalization", slot=self.normalize_z, tip="-Mean / Std")
        normalize_max         = self.prepare_menu_entry("Change to Max=1-Normalization", slot=self.normalize_max, tip="-Mean / Max")
        normalize_median      = self.prepare_menu_entry("Change to non-zero Median Normalization", slot=self.normalize_nonzero_median, tip="-Mean / non-zero-Median (DON'T USE WITH NEGATIVE VALUES)")
        change_discretization = self.prepare_menu_entry("Change internal discretization", slot=self.generate_discretization_splits, tip="Change the internal discretization")
        click_sensitivity     = self.prepare_menu_entry('Set click sensitivity', shortcut=None, slot=self.set_sensitivity, tip="Set the pixel-radius for a point to be picked")
        link_label            = self.prepare_menu_entry('Clear link-constraints:', greyed_out=True)
        clear_must_links      = self.prepare_menu_entry("    Clear all must-link constraints", slot=self.clear_ml, tip=None)
        clear_cannot_links    = self.prepare_menu_entry("    Clear all cannot-link constraints", slot=self.clear_cl, tip=None)
        self.add_menu_entry(self.edit_menu, (normalize_z, normalize_max, normalize_median, None, click_sensitivity, change_discretization, None, link_label, clear_must_links, clear_cannot_links))


        self.projection_algorithm = self.menuBar().addMenu("&Projection Algorithm")
        static_label        = self.prepare_menu_entry('Static embeddings:', greyed_out=True)
        static_xy           = self.prepare_menu_entry("    XY", shortcut=None, slot=self.select_xy, tip="XY-scatter plot of the first two checked attributes")
        static_pca          = self.prepare_menu_entry("    PCA", shortcut=None, slot=self.select_pca, tip="Principal Component Analysis")
        static_lle          = self.prepare_menu_entry("    LLE", shortcut=None, slot=self.select_lle, tip="Locally Linear Embedding")
        static_iso          = self.prepare_menu_entry("    Isomap", shortcut=None, slot=self.select_isomap, tip="Isomap")
        static_mds          = self.prepare_menu_entry("    MDS", shortcut=None, slot=self.select_mds, tip="Multi dimensional Scaling")
        static_ica          = self.prepare_menu_entry("    ICA", shortcut=None, slot=self.select_ica, tip="Independent Component Analysis")
        static_tsne         = self.prepare_menu_entry("    t-SNE", shortcut=None, slot=self.select_tsne, tip="t-distributed stochastic nearest neighbor embedding")
        interactive_label   = self.prepare_menu_entry('Interactive embeddings:', greyed_out=True)
        lsp_selection       = self.prepare_menu_entry("    LSP", slot=self.select_lsp, tip="Least Squared error Projection")
        kb_pca_selection    = self.prepare_menu_entry("    c-KPCA", slot=self.select_cpca, tip="constrained Knowledge Based Kernel Principal Component Analysis (Slow + Initialization can take quite long)")
        c_kpca_iterative    = self.prepare_menu_entry("    c-KPCA iterative", slot=self.select_ckpca_iterative, tip="constrained Knowledge Based Kernel Principal Component Analysis (Slow + Initialization can take quite long)")
        mle_selection       = self.prepare_menu_entry("    MLE", slot=self.select_mle, tip="Maximum Likelihood Embedding")
        self.add_menu_entry(self.projection_algorithm, (static_label, static_xy, static_pca, static_lle, static_iso, static_mds, static_ica, static_tsne, None, interactive_label, lsp_selection, kb_pca_selection, c_kpca_iterative, mle_selection))
        
        self.view_menu = self.menuBar().addMenu("&View")
        color_scheme_label = self.prepare_menu_entry('Color schemes:', greyed_out=True)
        colormap_ryg       = self.prepare_menu_entry("    Red, Yellow, Green", shortcut=None, slot=self.set_color_RdYlGn, tip=None)
        colormap_spectral  = self.prepare_menu_entry("    Spectral", shortcut=None, slot=self.set_color_spectral, tip=None)
        colormap_blue      = self.prepare_menu_entry("    Blue", shortcut=None, slot=self.set_color_Blues, tip=None)
        colormap_hot       = self.prepare_menu_entry("    Hot", shortcut=None, slot=self.set_color_hot, tip=None)
        information_label  = self.prepare_menu_entry('Toggle display of informations:', greyed_out=True)
        show_names         = self.prepare_menu_entry("    Only names", shortcut=None, slot=self.toggle_only_show_names, tip=None)
        boost_label_color  = self.prepare_menu_entry("    Boost label color", shortcut=None, slot=self.toggle_boost_label_color, tip=None)
        show_singletons    = self.prepare_menu_entry("    Show singletons", shortcut=None, slot=self.toggle_show_singletons, tip=None)
        show_i_attributes  = self.prepare_menu_entry("    Show ignored items in wordcloud", shortcut=None, slot=self.toggle_show_ignored_attributes, tip=None)
        toggle_center      = self.prepare_menu_entry("    Show center point", shortcut=None, slot=self.toggle_show_center_point, tip=None)
        toggle_links       = self.prepare_menu_entry("    Show must-/cannot-link constraints", shortcut=None, slot=self.toggle_show_links, tip=None)
        toggle_origin      = self.prepare_menu_entry("    Show origin", shortcut=None, slot=self.toggle_show_origin, tip=None)
        toggle_search_col  = self.prepare_menu_entry("    Show search as color", shortcut=None, slot=self.toggle_show_search_as_color, tip=None) 
        toggle_image_view  = self.prepare_menu_entry("    Show item-sets or decision-tree", shortcut=None, slot=self.toggle_show_itemsets_or_tree, tip=None)
        point_size_label   = self.prepare_menu_entry('Point appearence:', greyed_out=True)
        toggle_size        = self.prepare_menu_entry("    Toggle point size ~ label", shortcut=None, slot=self.toggle_point_size, tip="Make the size of the points proportional to their label")
        toggle_points      = self.prepare_menu_entry("    Toggle names instead of points", shortcut=None, slot=self.toggle_point_representation, tip=None)
        point_opacity      = self.prepare_menu_entry("    Set point opacity", shortcut=None, slot=self.set_point_opacity, tip="Set the transcarency of the points")
        proposal_label     = self.prepare_menu_entry('Active control point proposal:', greyed_out=True)
        colormap_likely    = self.prepare_menu_entry("    Color by uncertainty (only for MLE)", shortcut=None, slot=self.color_by_uncertainty, tip=None)
        self.add_menu_entry(self.view_menu, (color_scheme_label, colormap_ryg, colormap_spectral, colormap_blue, colormap_hot, None,  point_size_label, toggle_size, toggle_points, point_opacity, None, information_label, show_names, boost_label_color, show_singletons, show_i_attributes, toggle_center, toggle_links, toggle_origin, toggle_search_col, toggle_image_view, None, proposal_label, colormap_likely))


        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.prepare_menu_entry("&Manual", shortcut='F1', slot=self.open_help, tip='Open the manual pdf in a webbrowser')
        help1        = self.prepare_menu_entry('(De)select a control point:', greyed_out=True)
        help2        = self.prepare_menu_entry('      Manual select, middle-mouse-button-click, or <Shift>+left-click', greyed_out=True)
        help2_       = self.prepare_menu_entry('      Auto select next "best" control point (only MLE) <Ctrl+N>', greyed_out=True)
        help3        = self.prepare_menu_entry('', greyed_out=True)
        help4        = self.prepare_menu_entry('Request infos/annotate a point:', greyed_out=True)
        help5        = self.prepare_menu_entry('      Right-click, or <Ctrl+I>', greyed_out=True)
        help6        = self.prepare_menu_entry('', greyed_out=True)
        help7        = self.prepare_menu_entry('Lasso-select:', greyed_out=True)
        help8        = self.prepare_menu_entry('      <Ctrl+draw lasso with left mouse button pressed>', greyed_out=True)
        help9        = self.prepare_menu_entry('', greyed_out=True)
        help10       = self.prepare_menu_entry('Zoom (focussing on mouse-pointer):', greyed_out=True)
        help11       = self.prepare_menu_entry('      Mouse-wheel, or +/-', greyed_out=True)
        help12       = self.prepare_menu_entry('', greyed_out=True)
        help13       = self.prepare_menu_entry('Cut out selection and re-embed:', greyed_out=True)
        help14       = self.prepare_menu_entry('      <Ctrl>+X', greyed_out=True)
        help15       = self.prepare_menu_entry('', greyed_out=True)
        help16       = self.prepare_menu_entry('Clear all annotation:', greyed_out=True)
        help17       = self.prepare_menu_entry('      <Ctrl>+C', greyed_out=True)
        self.add_menu_entry(self.help_menu, (about_action, None, help1, help2, help2_, help3, help4, help5, help6, help7, help8, help9, help10, help11, help12, help13, help14, help15, help16, help17))


    # def load_session(self):
    #     import cPickle
    #     f = open('session.pkl', 'rb')
    #     tmp_dict = cPickle.load(f)
    #     f.close()          

    #     self.__dict__.update(tmp_dict) 
    #     print "Session loaded"
    #     self.update()


    # def save_session(self):
    #     import cPickle
    #     f = open('session.pkl', 'wb')
    #     cPickle.dump(self.__dict__,f,2)
    #     f.close()


    def load_session(self):
        import dill
        self.__dict__.update(dill.load(open('session.pkl')))
        print("Session loaded")
        self.update()


    def save_session(self):
        import dill
        dill.dump(self.__dict__, open('session.pkl', "w"))
        print("Session saved")

    def toggle_show_itemsets_or_tree(self):
        if self.image_displayer == self.create_tag_cloud:
            self.image_displayer = self.create_tree
        else:
            self.image_displayer = self.create_tag_cloud
        self.image_displayer()
        self.update()


    def toggle_show_search_as_color(self):
        self.show_search_as_color = not self.show_search_as_color
        if self.show_search_as_color == False:
            self.set_labels_as_colors()
        self.update()



    def set_point_opacity(self):
        w = PopupSlider('Set point opacity (default is 40%):', default=40, minimum=1, maximum=100)
        w.exec_()
        self.opacity = float(w.slider_value)/100.
        self.update()



    def toggle_show_origin(self):
        self.show_origin = not self.show_origin
        self.update()
    
    
    
    def toggle_show_singletons(self):
        self.dummies = np.eye(len(self.data.attribute_names)-len(self.data.ignored_attributes))
        self.show_singletons = not self.show_singletons
        self.update()
    


    def toggle_boost_label_color(self):
        self.boost_label_color = not self.boost_label_color
        self.set_labels_as_colors()
        self.update()


    def toggle_only_show_names(self):
        self.only_show_names = not self.only_show_names
        self.update()


    def set_sensitivity(self):
        self.w = PopupSlider('Set the sensitivity when picking a point (default is 5):', default=5, maximum=15)
        self.w.exec_()
        self.pick_sensitivity = int(self.w.slider_value)


    def toggle_show_ignored_attributes(self):
        self.show_ignored_attributes = not self.show_ignored_attributes
        self.image_displayer()
        self.update()


    def toggle_show_links(self):
        self.show_links = not self.show_links
        self.update()


    def clear_ml(self):
        if self.data != None:
            self.ml_cl_state = None
            if self.ml_cl_available:
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl.png")))
            else:
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl_unavailable.png")))
            self.pairwise_link_marker = 0
            self.must_link = []
            self.ml_cl_pending = False
            self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
            self.embedding_algorithm.update_control_points(self.control_points)
            self.update()


    def clear_cl(self):
        if self.data != None:
            self.ml_cl_state = None
            if self.ml_cl_available:
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl.png")))
            else:
                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl_unavailable.png")))
            self.pairwise_link_marker = 0
            self.cannot_link = []
            self.ml_cl_pending = False
            self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
            self.embedding_algorithm.update_control_points(self.control_points)
            self.update()


    def prepare_menu_entry(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, signal="triggered", greyed_out=False):
        """ Generate a menu entry """
        entry = QAction(text, self)
        if icon is not None:
            print(icon)
            entry.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            entry.setShortcut(shortcut)
        if tip is not None:
            entry.setToolTip(tip)
            entry.setStatusTip(tip)
        if slot is not None:
            actual_signal = getattr(entry, signal)
            actual_signal.connect(slot)
        if checkable:
            entry.setCheckable(True)
        if greyed_out:
            entry.setEnabled(False)
        return entry


    def add_menu_entry(self, target, entries):
        """ Add the entries to the GUI menu """
        for entry in entries:
            if entry is None:
                target.addSeparator()
            else:
                target.addAction(entry)


    def fill_attribute_list(self, names):
        """ Fills the attribute list, visible  on the right side of the application """
        self.series_list_model.clear()
        all_ingredient_names = sorted(list(names[:-1]))
        all_ingredient_names.append(names[-1])
        for name in all_ingredient_names:
            item = QStandardItem(name)
            if name in self.data.ignored_attributes:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            value_range = self.data.get_range(name)
            item.setCheckable(True)
            item.setEditable(True)
            item.setToolTip("%s [%.2f, %.2f]" %(name, value_range[0], value_range[1]))
            self.series_list_model.appendRow(item)

    
    def attributes_updated(self, event):
        """ Masks the data if attributes are (un)ignored """
        if self.data != None:
            ignore = []
            unignore = []
            for row in range(self.series_list_model.rowCount()):
                model_index = self.series_list_model.index(row, 0)
                checked = self.series_list_model.data(model_index, Qt.CheckStateRole) == QVariant(Qt.Checked)
                name = str(self.series_list_model.data(model_index))
                if checked:
                    unignore.append(name)
                else:
                    ignore.append(name)
            self.data.ignore_attributes(ignore)
            self.data.unignore_attributes(unignore)
            
            

    def recalculate_embedding(self):
        if len(self.data.ignored_attributes)+2 > len(self.data.attribute_names):
            msg = "U can't touch this!\nYou need at least two attributes selected."
            QMessageBox.about(self, "Too few attributes", msg)
        else:
            self.embedding_algorithm.__init__(self.data.data, self.control_points, self)
            self.dummies = np.eye(len(self.data.attribute_names)-len(self.data.ignored_attributes))
            self.update()



    def normalize_nonzero_median(self):
        self.data.normalize = self.data.normalize_nonzero_median
        self.data.normalize()
        if self.dummies != []:
            self.dummies = self.data.normalize(X=np.eye(len(self.dummies)))
        self.embedding_algorithm.__init__(self.data.data, self.control_points, self)
        self.set_xy_limits()
        self.update()


                
    def normalize_z(self):
        self.data.normalize = self.data.normalize_z
        self.data.normalize()
        if self.dummies != []:
            self.dummies = self.data.normalize(X=np.eye(len(self.dummies)))
        self.embedding_algorithm.__init__(self.data.data, self.control_points, self)
        self.set_xy_limits()
        self.update()
    

    def normalize_max(self):
        self.data.normalize = self.data.normalize_max
        self.data.normalize()
        if self.dummies != []:
            self.dummies = self.data.normalize(X=np.eye(len(self.dummies)))
        self.embedding_algorithm.__init__(self.data.data, self.control_points, self)
        self.set_xy_limits()
        self.update()
    
    
    def label_updated(self):
        """ Re-colors the embedding according to the new set label """
        if self.data != None:
            text = str(self.label_text_field.text())
            old_label = self.data.label_name
            try:
                self.data.set_attribute_as_label(text)
            except:
                self.label_text_field.setText(old_label)
            self.set_labels_as_colors()
            self.uncertainty_coloring_flag = False
            self.update()

    
    def set_next_attribute_as_label(self):
        """ Re-colors the embedding according to the new set label """
        if self.data != None:
            label = self.data.get_next_label_name()
            self.data.set_attribute_as_label(label)
            self.label_text_field.setText(label)
            self.set_labels_as_colors()
            self.uncertainty_coloring_flag = False
            self.update()

    
    def toggle_point_size(self):
        """ Re-colors the embedding according to the new set label """
        if self.data != None:
            if self.point_size_is_set_variable == False:
                self.point_size_is_set_variable = True
                labels = self.data.get_labels()
                bottom = min(labels)
                top    = max(labels)
                labels -= bottom
                labels /= top
                labels *= 250
                labels += 50
                self.point_size = labels
            else:
                self.point_size_is_set_variable = False
                self.point_size = np.ones(len(self.data.data))*60
            self.update()


    def reset_label(self):
        """ resets the label and the coloring back to the last attribute """
        if self.uncertainty_coloring_flag == True:
            self.uncertainty_coloring_flag = False
            self.data.set_attribute_as_label(self.data.attribute_names[-1])
            self.label_text_field.setText(self.data.attribute_names[-1])
            self.set_labels_as_colors()


    def create_status_bar(self):
        """ Message in the status bar at the bottom of the GUI """
        self.status_text = QLabel("Please load a data file")
        self.statusBar().addWidget(self.status_text, 1)


    def update_status_bar(self, message):
        """ Update the status bar at the bottom of the GUI """
        self.statusBar().showMessage(message)



    def open_help(self):
        """ Opens a pdf manual """
        import webbrowser

        # webbrowser.open_new('http://www-kd.iai.uni-bonn.de/softwarefiles/31/manual.pdf')
        webbrowser.open_new(os.path.join(self.cwd,"manual.pdf"))



    def select_all_action(self):
        """ (de)select all attributes in the attribute list in the GUI """
        if self.data != None:
            self.series_list_model.itemChanged.disconnect(self.attributes_updated)
            attribute_range = range(self.series_list_model.rowCount())
            changed_attributes_name_list = []
            if self.all_selected:
                self.all_selected = False
                for row in attribute_range:
                    check_box = self.series_list_model.item(row)
                    check_box.setCheckState(Qt.Unchecked)
                    model_index = self.series_list_model.index(row, 0)
                    name = str(self.series_list_model.data(model_index))
                    changed_attributes_name_list.append(name)
                self.data.ignored_attributes = []
                self.data.ignore_attributes(changed_attributes_name_list)
            else:
                self.all_selected = True
                for row in attribute_range:
                    check_box = self.series_list_model.item(row)
                    check_box.setCheckState(Qt.Checked)
                    model_index = self.series_list_model.index(row, 0)
                    name = str(self.series_list_model.data(model_index))
                    changed_attributes_name_list.append(name)
                self.data.ignored_attributes = []
                self.data.ignore_attributes([])
            self.series_list_model.itemChanged.connect(self.attributes_updated)
            

    def highlight_search_results(self):
        """ Highlight the data records that match the search """
        if self.data != None:
            term = str(self.search_text_field.text())
            self.searched_results = [i for i,name in enumerate(self.data.instance_names) if len(findall(term, name)) > 0]
            self.update()


    def toggle_data_filter(self):
        """ Toggle if all data, or just a selection is used for the embedding """
        if self.data != None:
            if self.data_clipped == False:
                if len(self.lassoed_points) > 1:
                    self.cut_button.setIcon(QIcon(os.path.join(self.cwd,"cut_active.png")))
                    self.data_clipped = True
                    self.unclipped_data = copy(self.data.data)
                    self.must_link_original = copy(self.must_link)
                    self.cannot_link_original = copy(self.cannot_link)
                    self.saved_control_points = copy(self.control_points)
                    self.must_link = []
                    self.cannot_link = []
                    self.unclipped_instance_names = copy(self.data.instance_names)
                    self.unclipped_original_data = copy(self.data.original_data)
                    self.control_points = {}
                    self.data.instance_names = [self.data.instance_names[i] for i in self.lassoed_points]
                    self.data.data = self.data.data[self.lassoed_points]
                    self.data.original_data = self.data.original_data[self.lassoed_points]
                    self.embedding_algorithm.__init__(self.data.data, self.control_points, self)
                    self.set_labels_as_colors()
                    self.clear()
            else:
                self.cut_button.setIcon(QIcon(os.path.join(self.cwd,"cut.png")))
                self.data_clipped = False
                self.must_link = copy(self.must_link_original)
                self.cannot_link = copy(self.cannot_link_original)
                self.data.original_data = copy(self.unclipped_original_data)
                self.data.data = copy(self.unclipped_data)
                self.control_points = copy(self.saved_control_points)
                self.data.instance_names = copy(self.unclipped_instance_names)
                self.embedding_algorithm.__init__(self.data.data, self.control_points, self)
                self.set_labels_as_colors()
                self.clear()


    def toggle_point_representation(self):
        """ Toggle the representation of the data recordes between points and text """
        if self.data != None:
            self.point_representation = not self.point_representation
            self.update()


    def toggle_singleton_embedding(self):
        """ Additionally embedd the singletons, or not """
        if self.data != None:
            self.show_singletons = not self.show_singletons
            self.update()


    def toggle_show_center_point(self):
        """ Show the most centric point of the selected data points, or not """
        if self.data != None:
            self.show_center_point = not self.show_center_point
            self.update()

    def set_xy_limits(self):
        e = self.embedding_algorithm.get_embedding().T
        e_min = np.min(e, axis=0)
        e_max = np.max(e, axis=0)
        xadd = (e_max[0] - e_min[0])/2.
        yadd = (e_max[1] - e_min[1])/2.
        if (abs(e_min[0] - e_max[0]) < 0.01) or (abs(e_min[1] - e_max[1]) < 0.01):
            pass
        else:
            self.xlim = [e_min[0]-xadd, e_max[0]+xadd]
            self.ylim = [e_min[1]-yadd, e_max[1]+yadd]


    def set_mc_cl_unavailable(self):
        self.ml_cl_available = False
        self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl_unavailable.png")))


    def set_mc_cl_available(self):
        self.ml_cl_available = True
        self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml_cl.png")))


    def select_xy(self):
        """ Calculate a plain PCA embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (XY)')
            self.reset_label()
            self.embedding_algorithm = XY(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()


    def select_pca(self):
        """ Calculate a plain PCA embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (PCA)')
            self.reset_label()
            self.embedding_algorithm = PCA(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()


    def select_lle(self):
        """ Calculate a locally linear embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (LLE)')
            self.reset_label()
            self.embedding_algorithm = LLE(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()


    def select_isomap(self):
        """ Calculate an isometric embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (Isomap)')
            self.reset_label()
            self.embedding_algorithm = ISO(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()


    def select_mds(self):
        """ Calculate a multi dimensional scaling embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (MDS)')
            self.reset_label()
            self.embedding_algorithm = MDS(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()



    def select_ica(self):
        """ Calculate an independent component analysis embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (ICA)')
            self.reset_label()
            self.embedding_algorithm = ICA(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()



    def select_tsne(self):
        """ Calculate an t-SNE embedding """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_unavailable()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (t-SNE)')
            self.reset_label()
            self.embedding_algorithm = tSNE(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.update()


    def select_lsp(self):
        """ Selecte LSP as embedding algorithm """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_available()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (LSP)')
            self.reset_label()
            self.embedding_algorithm = LSP(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
            self.embedding_algorithm.update_control_points(self.control_points)
            self.update()


    def select_cpca(self):
        """ Selecte constrained knowledge based kernel PCA as embedding algorithm """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_available()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (c-KPCA)')
            self.reset_label()
            self.embedding_algorithm = cPCA(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
            self.embedding_algorithm.update_control_points(self.control_points)
            self.update()


    def select_ckpca_iterative(self):
        """ Selecte constrained knowledge based kernel PCA as embedding algorithm (iteratve version)"""
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select_unavailable.png")))
            self.set_mc_cl_available()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (c-KPCA iterative)')
            self.reset_label()
            self.embedding_algorithm = ConstrainedKPCAIterative(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
            self.embedding_algorithm.update_control_points(self.control_points)
            self.update()


    def select_mle(self):
        """ Selecte MLE as embedding algorithm """
        if self.data != None:
            self.auto_select_button.setIcon(QIcon(os.path.join(self.cwd,"auto_select.png")))
            self.set_mc_cl_available()
            self.setWindowTitle('InVis: ' + self.data.dataset_name + ' (MLE)')
            self.embedding_algorithm = MLE(self.data.data, self.control_points, self)
            self.set_xy_limits()
            self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
            self.embedding_algorithm.update_control_points(self.control_points)
            self.update()


    def set_labels_as_colors(self):
        """ 0-1 normalizes the labels values and sets them as colors """
        if self.data != None:
            labels = None
            if self.boost_label_color == True:
                labels = self.data.get_labels()
                labels -= min(labels)
                labels = np.log(labels+1)
            else:
                labels = self.data.get_labels()
                labels -= min(labels)
            self.colors = labels
            self.colors /= np.max(self.colors)


    def set_color_RdYlGn(self):
        """ Set the coloring of the embedded data records to Red/Yellow/Green """
        if self.data != None:
            self.color_scheme = "RdYlGn"
            self.update()


    def set_color_spectral(self):
        """ Set the coloring of the embedded data records to spectral """
        if self.data != None:
            self.color_scheme = "Spectral"
            self.update()


    def set_color_Blues(self):
        """ Set the coloring of the embedded data records to blue tones """
        if self.data != None:
            self.color_scheme = "Blues"
            self.update()


    def set_color_hot(self):
        """ Set the coloring of the embedded data records to a hot scale """
        if self.data != None:
            self.color_scheme = "hot"
            self.update()


    def color_by_uncertainty(self):
        """ Set the coloring of the embedded data records according to their informativeness (in blue tones) """
        if self.data != None:
            if self.embedding_algorithm.name == 'MLE':
                self.data.set_attribute_as_label(self.data.attribute_names[-1])
                self.uncertainty_coloring_flag = True
                self.label_text_field.setText("UNCERTAINTY LABELING")
                uncertainties = []
                for x in self.data.data:
                    uncertainties.append(x.dot(self.embedding_algorithm.Psi).dot(x.T))
                self.colors = np.array(uncertainties)
                self.colors -= np.min(self.colors)
                self.colors /= np.max(self.colors)
                self.update()


    def next_uncertain(self):
        if self.data != None:
            if self.embedding_algorithm.name == 'MLE':
                uncertainties = []
                for x in self.data.data:
                    uncertainties.append(x.dot(self.embedding_algorithm.Psi).dot(x.T))
                inds = np.argsort(uncertainties)[::-1]
                for ind in inds:
                    if ind in self.control_points:
                        continue
                    else:
                        self.toggle([ind])
                        self.embedding_algorithm.update_control_points(self.control_points)
                        break
                if self.uncertainty_coloring_flag:
                    self.color_by_uncertainty()
                else:
                    self.update()


    def select_search_result(self):
        if self.data != None:
            term = str(self.search_text_field.text())
            self.searched_results = [i for i,name in enumerate(self.data.instance_names) if len(findall(term, name)) > 0]
            self.lassoed_points = self.searched_results
            self.image_displayer()
            self.update()


    def generate_discretization_splits(self):
        if self.data != None:
            self.discretization_type, ok = QInputDialog.getText(self, 'Discretization', 'Enter number of the desired discretization method:\n1) True if value > 0 (Default)\n2) True if value > average\n3) True if value > median\n4) True if value > half range\n5) True if value > average + 1*std')
            if self.discretization_type in ['1', '']:
                self.splits = list(np.zeros(len(self.data.attribute_names)))
            elif  self.discretization_type == '2':
                self.splits = list(np.average(self.data.data, axis=0))
            elif  self.discretization_type == '3':
                self.splits = list(np.median(self.data.data, axis=0))
            elif  self.discretization_type == '4':
                self.splits = []
                for name in self.data.attribute_names:
                    min_val, max_val = self.data.get_range(name)
                    self.splits.append((max_val-min_val)/2.0)
            elif  self.discretization_type == '5':
                self.splits = list(np.average(self.data.data, axis=0) + np.std(self.data.data, axis=0))



    def create_tree(self):
        if self.discretization_type == None:
            self.generate_discretization_splits()
        d = path.dirname(__file__)

        import matplotlib.image as mplimg
        import pydotplus
        from PIL import Image
        from sklearn import model_selection, tree
        from io import StringIO

        attributes = [i for i,name in enumerate(self.data.attribute_names) if name not in self.data.ignored_attributes]
        attribute_names = [name for name in self.data.attribute_names if name not in self.data.ignored_attributes]
        X = self.data.original_data.T[attributes].T
        Y = np.zeros(len(self.data.original_data)) 
        Y[self.lassoed_points] = 1.0
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=10, class_weight='balanced')
        clf.fit(X, Y)

        dotfile = StringIO()
        tree.export_graphviz(clf, out_file=dotfile, feature_names=attribute_names, impurity=False, class_names=['Not-Selected', 'Selected'], filled=True, rounded=True)
        graph = pydotplus.graph_from_dot_data(dotfile.getvalue())

        acc = "Accuracy\n%.2f%%" %(abs(model_selection.cross_val_score(clf, X, Y, scoring='accuracy', cv=10).mean())*100.)
        node_a = pydotplus.Node(label=acc, style="filled", shape='note', fillcolor='yellow')
        graph.add_node(node_a)

        graph.set_size(5)
        graph.write_png("dtree.png")

        image = Image.open("dtree.png")
        image = image.convert("RGBA")
        datas = image.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item[:-1] + (160,))
        image.putdata(newData)

        self.fig.images = []
        self.fig.figimage(image, 2, 2, zorder=10)


    def create_tag_cloud(self):
        """ Generate a tag-cloud of the ingredients of the selected data records """
        if self.discretization_type == None:
            self.generate_discretization_splits()
        d = path.dirname(__file__)
        iMiner = ItemsetMiner(self.data, self.splits, 1, 1, 50, self.lassoed_points, show_ignored_attributes=self.show_ignored_attributes)
        transactions = iMiner.build_itemsets(constraint='0')
        patterns = iMiner.get_item_frequencies(transactions, top_k=10)
        beautyfied_tag_counts = iMiner.beautify_names_and_normalize_counts(patterns)
        self.tag_elements = wordclouds.fit_words(beautyfied_tag_counts, font_path=path.join(d, 'ressources/JosefinSans.ttf'))
        image = self.make_tag_cloud_image()
        self.fig.images = []
        self.fig.figimage(image, 2, 2, zorder=1)


    def make_tag_cloud_image(self):
        """ docstring for make_tag_cloud_image """
        d = path.dirname(__file__)
        font_path = path.join(d, 'ressources/JosefinSans.ttf')
        img = Image.new("RGBA", (400,200), color = (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        for (word, count), font_size, position, orientation in self.tag_elements:
            font = ImageFont.truetype(font_path, font_size)
            transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
            #draw.setfont(transposed_font)
            draw.font = transposed_font
            color = (0, 0, 0, 190)
            pos = (position[1], position[0])
            draw.text(pos, word, fill=color)
        return img


    def get_geometric_median_index_of_index_set(self, index_set):
        """ Finds the most middle point of a point cloud in the original space """
        selection = self.embedding.T[index_set]
        distances = distance.squareform(distance.pdist(selection, 'euclidean'))
        # return index_set[np.argmin(np.sum(distances, axis=0))]
        return index_set[np.argmin(np.sum(distances**2, axis=0))]


    def toggle(self, indices):
        """ Toggle the control point selection """
        if len(indices) == 1:
            index = indices[0]
            if index in self.control_points:
                del self.control_points[index]
            else:
                self.control_points[index] = self.embedding.T[index]
        elif len(indices) > 1:
            for index in indices:
                self.control_points[index] = self.embedding.T[index]
                

    def highlight_center_data_record(self):
        if self.center_ind != None and self.show_center_point:
            box_style=dict(boxstyle="round", fc="1.0", alpha=0.7, ec="b")
            font = pl.matplotlib.font_manager.FontProperties(family='monospace', size = 10)
            msg = "Most central Point in selection:\n\n" + self.data.instance_names[self.center_ind]
            for i,amount in enumerate(self.data.original_data[self.center_ind][:-1]):
                if amount > 0.0:
                    msg += "\n  %.2f - %s" %(amount, self.data.attribute_names[i])
            self.axes.annotate(msg, self.embedding.T[self.center_ind], zorder=10,
                               bbox=box_style, fontsize=14, 
                               xytext=(10,20), 
                               textcoords='offset points', 
                               arrowprops=dict(arrowstyle='->', 
                                               connectionstyle='angle,angleA=90,angleB=0,rad=10',
                                               color='b'
                                              )
                               )


    def draw_search_annotations(self):
        if self.show_search_as_color == False:
            """ Render text to all points that shall be annotated """
            box_style=dict(boxstyle="round", fc="1.0", alpha=0.7, ec="r")
            font = pl.matplotlib.font_manager.FontProperties(family='monospace', size = 10)
            for highlight in self.searched_results:
                msg = self.data.instance_names[highlight]
                self.axes.annotate(msg, self.embedding.T[highlight], zorder=10,
                                   bbox=box_style, fontsize=14,
                                   xytext=(30,-5), 
                                   textcoords='offset points', 
                                   arrowprops=dict(arrowstyle='->', 
                                                   color='r'
                                                  )
                                   )
            


    def draw_info_annotations(self):
        """ Render text to all points that shall be annotated """
        box_style=dict(boxstyle="round", fc="1.0", alpha=0.7, ec="g")
        font = pl.matplotlib.font_manager.FontProperties(family='monospace', size = 10)
        annotation_fontsize = 14
        for highlight in self.info_requests:
            name = self.data.instance_names[highlight]
            maxlen = 20
            if len(name) >= maxlen:
                trimmed_name = name[:maxlen]
                counter = maxlen
                for i in name[maxlen:]:
                    counter += 1
                    if i == " ":
                        if counter > maxlen:
                            trimmed_name += "\n"
                            counter = 0
                        else:
                            trimmed_name += i
                    else:
                        trimmed_name += i
                name = trimmed_name
            msg = name
            counter = 0
            if self.only_show_names == False:
                pass
            else:
                for i,amount in enumerate(self.data.original_data[highlight]):
                    if amount != 0.0:
                        counter += 1
                        msg += "\n  %.2f - %s" %(amount, self.data.attribute_names[i])
                        if counter <= 10:
                            annotation_fontsize = 14
                        elif 10 < counter <= 50:
                            annotation_fontsize = int(16 - 0.2*counter)
                        else:
                            annotation_fontsize = 6
            self.axes.annotate(msg, self.embedding.T[highlight], zorder=10,
                               bbox=box_style, fontsize=annotation_fontsize, 
                               xytext=(10,20), 
                               textcoords='offset points', 
                               arrowprops=dict(arrowstyle='->', 
                                               connectionstyle='angle,angleA=90,angleB=0,rad=10',
                                               color='g'
                                              )
                               )


    def add_must_link(self, ind):
        self.ml_cl_index = ind
        if self.pairwise_link_marker == 0:
            self.ml_cl_pending = True
            self.must_link.append([ind])
            self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml1.png")))
        else:
            if ind != self.must_link[-1][0]:
                self.must_link[-1].append(ind)
                self.must_link[-1] = set(self.must_link[-1])
                if self.must_link[-1] in self.must_link[:-1]:
                    #print "found double"
                    self.must_link = self.must_link[:-1]
                elif self.must_link[-1] in self.cannot_link:
                    #print "killing link"
                    self.cannot_link.remove(self.must_link[-1])
                    self.must_link = self.must_link[:-1]
            else:
                #print "same point"
                self.must_link = self.must_link[:-1]
            self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml.png")))
            self.ml_cl_pending = False
        self.pairwise_link_marker += 1
        self.pairwise_link_marker %= 2
        self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
        self.embedding_algorithm.update_control_points(self.control_points)
        self.update()


    def add_cannot_link(self, ind):
        self.ml_cl_index = ind
        if self.pairwise_link_marker == 0:
            self.ml_cl_pending = True
            self.cannot_link.append([ind])
            self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"cl1.png")))
        else:
            if ind != self.cannot_link[-1][0]:
                self.cannot_link[-1].append(ind)
                self.cannot_link[-1] = set(self.cannot_link[-1])
                if self.cannot_link[-1] in self.cannot_link[:-1]:
                    #print "found double"
                    self.cannot_link = self.cannot_link[:-1]
                elif self.cannot_link[-1] in self.must_link:
                    #print "killing link"
                    self.must_link.remove(self.cannot_link[-1])
                    self.cannot_link = self.cannot_link[:-1]
            else:
                #print "same point"
                self.cannot_link = self.cannot_link[:-1]
            self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"cl.png")))
            self.ml_cl_pending = False
        self.pairwise_link_marker += 1
        self.pairwise_link_marker %= 2
        self.embedding_algorithm.update_must_and_cannot_link(self.must_link, self.cannot_link)
        self.embedding_algorithm.update_control_points(self.control_points)
        self.update()


    def on_point_pick(self, event):
        """ Triggers different actions on picking a point with the mouse (for left/middle/right click) """
        interesting_points = []
        interesting_points += self.searched_results
        interesting_points += self.info_requests
        # TODO check if .keys() has an issue here
        interesting_points += self.control_points.keys()
        ind = event.ind[0]
        for p in interesting_points:
            if p in event.ind:
                ind = p
                break
        if event.mouseevent.button == 1:
            self.path = None
            for patch in self.axes.patches:
                patch.remove()
            if self.info_request:
                if ind not in self.info_requests:
                    self.info_requests.append(ind)
            elif self.cp_select_request:
                self.toggle([ind])
                self.embedding_algorithm.update_control_points(self.control_points)
                if self.uncertainty_coloring_flag == True:
                    self.color_by_uncertainty()
            elif self.shift_click:
                self.toggle([ind])
                self.embedding_algorithm.update_control_points(self.control_points)
                if self.uncertainty_coloring_flag == True:
                    self.color_by_uncertainty()
            elif self.ml_cl_available and (self.ml_cl_state == 'ML'):
                self.add_must_link(ind)
            elif self.ml_cl_available and (self.ml_cl_state == 'CL'):
                self.add_cannot_link(ind)
            # TODO check if .keys() has an issue here
            if ind in self.control_points.keys():
                self.selected_point = ind
        elif event.mouseevent.button == 2:
            self.toggle([ind])
            self.embedding_algorithm.update_control_points(self.control_points)
            if self.uncertainty_coloring_flag == True:
                self.color_by_uncertainty()
        elif event.mouseevent.button == 3:
            if ind not in self.info_requests:
                self.info_requests.append(ind)


    def on_press(self, event):
        """ On mouse key press """
        if event.inaxes is None: return
        if self.axes.figure.canvas.widgetlock.locked(): return
        if self.control_click or self.lasso_request:
            self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.lasso_callback)
            self.axes.figure.canvas.widgetlock(self.lasso)
            self.lassoLock = True


    def on_release(self, event):
        """ On mouse key release """
        self.lasso_request = False
        if self.lassoLock:
            self.axes.figure.canvas.widgetlock.release(self.lasso)
            self.lassoLock = False
            self.lasso_button.setIcon(QIcon(os.path.join(self.cwd,"lasso.png")))
        if self.selected_point != None:
            self.embedding_algorithm.finished_relocating()
            self.selected_point = None
            self.update()
        self.update()
        

    def lasso_callback(self, verts):
        """ Finds all points inside the lasso-selected area """
        self.path = matplotlib.path.Path(verts)
        ind = np.nonzero([self.path.contains_point(xy) for xy in self.embedding.T])[0]
        self.lassoed_points = ind
        if len(self.lassoed_points) > 1:
            self.image_displayer()
            self.center_ind = self.get_geometric_median_index_of_index_set(self.lassoed_points)
        self.axes.figure.canvas.widgetlock.release(self.lasso)
        del self.lasso
        self.update()

    def on_key_press(self, event):
        """ On key press """
        if event.inaxes:
            if event.key == 'control':
                self.control_click = True
                self.lasso_button.setIcon(QIcon(os.path.join(self.cwd,"lasso_active.png")))
            elif event.key == 'shift':
                self.shift_click = True
            elif event.key == '+':
                self.zoom('up', event.xdata, event.ydata)
            elif event.key == '-':
                self.zoom('down', event.xdata, event.ydata)


    def on_key_release(self, event):
        """ On key release """
        self.control_click = False 
        self.lasso_button.setIcon(QIcon(os.path.join(self.cwd,"lasso.png")))
        self.shift_click = False


    def on_motion(self, event):
        """ On mouse motion """
        if event.inaxes:
            if self.selected_point != None:
                if self.ml_cl_pending == True:
                    if self.ml_cl_state == 'ML':
                        try:
                            if len(self.must_link[-1]) == 1:
                                self.pairwise_link_marker += 1
                                self.pairwise_link_marker %= 2
                                self.must_link = self.must_link[:-1]
                                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"ml.png")))
                                self.ml_cl_pending = False
                        except:
                            pass
                    if self.ml_cl_state == 'CL':
                        try:
                            if len(self.cannot_link[-1]) == 1:
                                self.pairwise_link_marker += 1
                                self.pairwise_link_marker %= 2
                                self.cannot_link = self.cannot_link[:-1]
                                self.ml_cl_button.setIcon(QIcon(os.path.join(self.cwd,"cl.png")))
                                self.ml_cl_pending = False
                        except:
                            pass
                if self.embedding_algorithm.is_dynamic:
                    self.control_points[self.selected_point] = [event.xdata, event.ydata]
                    self.embedding_algorithm.update_control_points(self.control_points)
                    self.update()
    

    def zoom(self, direction, xdata, ydata):
        scale = 1.2
        if direction == 'up':
            factor = 1.0/scale
        elif direction == 'down':
            factor = scale
        curr_xlim  = self.axes.get_xlim()
        curr_ylim  = self.axes.get_ylim()
        new_width  = (curr_xlim[1]-curr_xlim[0])*factor
        new_height = (curr_ylim[1]-curr_ylim[0])*factor
        relx = (curr_xlim[1]-xdata)/(curr_xlim[1]-curr_xlim[0])
        rely = (curr_ylim[1]-ydata)/(curr_ylim[1]-curr_ylim[0])
        self.xlim = [xdata-new_width*(1-relx), xdata+new_width*(relx)]
        self.ylim = [ydata-new_height*(1-rely), ydata+new_height*(rely)]
        self.update()


    def on_scroll(self, event):
        """ On mouse scroll """
        if self.data != None:
            if event.inaxes:
                self.zoom(event.button, event.xdata, event.ydata)

    
    def clear(self):
        """ Clear the canvas from searches and annotated infos """
        self.fig.images = []
        self.lassoed_points = []
        self.path = None
        self.center_ind = None
        for patch in self.axes.patches:
            patch.remove()
        self.searched_results = []
        self.info_requests = []
        self.highlighted = []
        self.search_text_field.setText("")
        if isinstance(self.data.data, list) and self.data.data:
            self.update()
        elif isinstance(self.data.data, np.ndarray) and self.data.data.size != 0:
            self.update()



    def update(self):
        """ Renders the embedding """
        self.axes.clear()
        self.embedding = self.embedding_algorithm.get_embedding()
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylim(self.ylim)
        self.axes.set_aspect('auto')
        for text in self.axes.texts:
            text.remove()
        if self.show_origin:
            self.axes.plot([0,0], [self.ylim[0], self.ylim[1]], color='k', alpha=0.08, zorder=0)
            self.axes.plot([self.xlim[0], self.xlim[1]], [0,0], color='k', alpha=0.08, zorder=0)
        if self.point_representation:
            if self.show_search_as_color:
                self.colors = np.zeros(len(self.data.data))
                for highlight in self.searched_results:
                    self.colors[highlight] = 1.0
            # this [0] should circumvent the problem that vectors need to be equal length, but could later cause problems if for some reason the point_size is not a list
            self.scatter_plot = self.axes.scatter(self.embedding[0], self.embedding[1], color=pl.get_cmap(self.color_scheme)(self.colors), picker=self.pick_sensitivity, edgecolor=(0.3,0.3,0.3,0.2), s=self.point_size[0], zorder=2, alpha=self.opacity)
        else:
            self.scatter_plot = self.axes.scatter(self.embedding[0], self.embedding[1], facecolor='none', picker=self.pick_sensitivity, edgecolor=(0.3,0.3,0.3,0.2), s=self.point_size[0], zorder=2, alpha=self.opacity)
            for i, txt in enumerate(self.data.instance_names):
                self.axes.annotate(txt, (self.embedding[0][i], self.embedding[1][i]), alpha=0.5)
        if self.show_singletons:
            if self.embedding_algorithm.name in ['MLE', 'LSP']:
                dummy_embedding = self.embedding_algorithm.get_embedding(X=self.dummies)
                norms = np.linalg.norm(dummy_embedding, axis=0)
                sorting = np.argsort(norms)[::-1]
                attr_names = list(self.data.attribute_names)
                for n in self.data.ignored_attributes:
                    attr_names.remove(n)
                for s in sorting[:5]:
                    x,y = dummy_embedding.T[s]
                    self.axes.annotate(attr_names[s], (x, y), alpha=0.6, fontsize=14,color='k')
        if len(self.control_points.keys()) > 0:
            control_point_indices = list(self.control_points.keys())
            self.axes.scatter(self.embedding[0][control_point_indices], self.embedding[1][control_point_indices], color='k', s=self.point_size[0]+40, facecolor='none', edgecolor=self.control_point_color[self.color_scheme], linewidth=4, zorder=100)
        if len(self.lassoed_points) > 0:
            self.axes.scatter(self.embedding[0][self.lassoed_points], self.embedding[1][self.lassoed_points], color='k', s=self.point_size[0], facecolor='none', edgecolor='k', linewidth=2, zorder=10, alpha=0.3)
            if self.path != None:
                self.axes.add_patch(patches.PathPatch(self.path, color='k', lw=0, alpha=0.2, zorder=0))
        if self.show_links:
            for link in self.must_link:
                self.axes.plot(self.embedding[0][list(link)], self.embedding[1][list(link)], 'g-', alpha=0.5)
            for link in self.cannot_link:
                self.axes.plot(self.embedding[0][list(link)], self.embedding[1][list(link)], 'r:', alpha=0.5)
        if self.ml_cl_pending:
            self.axes.scatter(self.embedding[0][self.ml_cl_index], self.embedding[1][self.ml_cl_index], color='k', s=self.point_size[0], facecolor='none', edgecolor='k', linewidth=5, zorder=200, alpha=0.3)
        self.draw_search_annotations()
        self.draw_info_annotations()
        self.highlight_center_data_record()

        self.canvas.draw()
    




