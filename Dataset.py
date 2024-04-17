#!/usr/bin/python
import numpy as np
import pandas as pd
from copy import copy
from sklearn.datasets import load_svmlight_file
#import arff
try:
    from sklearn.utils import arraybuilder # needed as explicit import for generating one binary executable file
except:
        pass
from unidecode import unidecode

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from statsmodels.tsa.arima_model import ARMA

class PopupAutocorrellationSlider(QDialog):
    def __init__(self, label_text, default=0, minimum=0, maximum=50):
        QWidget.__init__(self)
        self.slider_value = 1

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
        layout.addWidget(name_label      , 1, 1, 1, 3, Qt.AlignLeft)
        layout.addWidget(self.slider     , 2, 1, 2, 1, Qt.AlignLeft)
        layout.addWidget(self.value_label, 2, 2, 2, 2, Qt.AlignCenter)
        layout.addWidget(self.button     , 2, 3, 2, 3, Qt.AlignRight)

        self.setWindowTitle('Parameter choice')


    def slider_changed(self):
        val = self.slider.value()
        self.value_label.setText('%d' %val)
        self.slider_value = val
 

    def handleButton(self):
        self.hide()




class Dataset():
    def __init__(self):
        self.database_type = None
        self.original_data = []
        self.data = []
        self.instance_names = []
        self.attribute_names = []
        self.ignored_attributes = []
        self.considered_attributes = []
        self.label_name = ''
        self.label_index = -1
        self.dataset_name = ''
        self.masked_names = None
        self.pictures = []
        self.raw_data = []
        self.normalize = self.normalize_max


    def read_in_data(self, path):
        extension = path.split('.')[-1]
        if extension == 'csv':
            self.read_in_csv(path)
        elif extension == 'arff':
            try:
                self.read_in_arff(path)
            except Exception as e:
                print(e)
                print("Need package liac-arff installed")
                print("Try \"pip install liac-arff\"")
        elif extension == 'ts':
            self.read_in_timeseries(path)
        else: # assume it is libsvm format
            self.read_in_libsvm(path)
        self.mask = []
        for i, name in enumerate(self.attribute_names):
            if name not in self.ignored_attributes:
                self.mask.append(i)
        self.attribute_names = np.array([unidecode(n) for n in self.attribute_names])
        self.instance_names = [unidecode(n) for n in self.instance_names]
        #self.normalize()


    def load_pandas_dataframe(self, data):
        self.dataset_name = "Pandas DataFrame"  
        self.database_type = 'DATA'
        self.attribute_names = np.array([str(i) for i in data.columns])
        if data.index.dtype.kind in 'biufc': # is_numeric()
            self.instance_names = ["Instance: " + str(s) for s in data.index]
        else:
            self.instance_names = list(data.index.astype(str))
        self.considered_attributes = np.ones(len(self.attribute_names)).astype(int).tolist()
        self.original_data = np.array(data.values).astype(float)

        stds = np.std(self.original_data, axis=0)
        keeper = list(np.nonzero(stds)[0])
        self.attribute_names = self.attribute_names[keeper]
        self.original_data = self.original_data.T[keeper].T

        self.data = copy(self.original_data)
        self.label_name = self.attribute_names[self.label_index]
        self.ignored_attributes.append(self.label_name)
        self.mask = []
        for i, name in enumerate(self.attribute_names):
            if name not in self.ignored_attributes:
                self.mask.append(i)
        self.update_data()
        self.normalize()
        #self.update_data()
        



    def normalize_z(self, X=[]):
        if X == []:
            self.mask = []
            for i, name in enumerate(self.attribute_names):
                if name not in self.ignored_attributes:
                    self.mask.append(i)
            self.masked_names = [self.attribute_names[i] for i in self.mask]
            self.data = copy(self.original_data.T[self.mask].T)
            means = np.mean(self.data, axis=0)
            shifted_data = self.data - means
            variances = np.std(shifted_data, axis=0)
            variances[variances==0.0] = 1.0
            self.data = shifted_data/variances
        else:
            means = np.mean(X, axis=0)
            shifted_data = X - means
            variances = np.std(shifted_data, axis=0)
            variances[variances==0.0] = 1.0
            X = shifted_data/variances
            return X
        


    def normalize_max(self, X=[]):
        if X == []:
            self.mask = []
            for i, name in enumerate(self.attribute_names):
                if name not in self.ignored_attributes:
                    self.mask.append(i)
            self.masked_names = [self.attribute_names[i] for i in self.mask]
            self.data = copy(self.original_data.T[self.mask].T)
            means = np.mean(self.data, axis=0)
            shifted_data = self.data - means
            maxval = np.max(shifted_data, axis=0)
            self.data = shifted_data/maxval
        else:
            means = np.mean(X, axis=0)
            shifted_data = X - means
            maxval = np.max(shifted_data, axis=0)
            X = shifted_data/maxval
            return X
        
        
    
    def normalize_nonzero_median(self, X=[]):
        if X == []:
            self.mask = []
            for i, name in enumerate(self.attribute_names):
                if name not in self.ignored_attributes:
                    self.mask.append(i)
            self.masked_names = [self.attribute_names[i] for i in self.mask]
            self.data = copy(self.original_data.T[self.mask].T)
            nonzero_medians = np.array([np.median(col[col.nonzero()]) for col in self.data.T])
            nonzero_medians[np.isnan(nonzero_medians)] = 1.0
            nonzero_medians[nonzero_medians == 0.0] = 1.0
            self.data /= nonzero_medians
            self.data = self.data - np.mean(self.data, axis=0)
        else:
            nonzero_medians = np.array([np.median(col[col.nonzero()]) for col in X.T])
            nonzero_medians[np.isnan(nonzero_medians)] = 1.0
            nonzero_medians[nonzero_medians == 0.0] = 1.0
            X /= nonzero_medians
            X = X - np.mean(X, axis=0)
            return X
        
        


    def read_in_arff(self, path):
        path = str(path)
        self.dataset_name = path.split('/')[-1][:-5]
        data = arff.load(open(path))
        self.database_type = 'DATA'
        self.attribute_names = np.array([str(x[0]) for x in data['attributes']])
        considered = []
        for i,t in enumerate(data['attributes']):
            if str(t[1]).lower() in ['real', 'integer']:
                considered.append(i)
        self.attribute_names = self.attribute_names[considered]
        self.considered_attributes = np.ones(len(self.attribute_names)).astype(int).tolist()
        for i, line in enumerate(data['data']):
            self.instance_names.append("Data record %d" %i)
            row = [line[j] for j in considered]
            self.original_data.append(row)
        self.original_data = np.array(self.original_data)
        self.data = copy(self.original_data)
        self.label_name = self.attribute_names[self.label_index]
        self.ignored_attributes.append(self.label_name)
        self.update_data()


    def read_in_libsvm(self, path):
        path = str(path)
        self.dataset_name = path.split('/')[-1][:-5]
        x, y = load_svmlight_file(path)
        data = np.array(np.concatenate((x.todense(), y.reshape(len(y),1)), axis=1))
        self.database_type = 'DATA'
        self.attribute_names = np.array(["attr_%d"%i for i in range(len(data[0]))])
        self.attribute_names[-1] = 'label'
        self.considered_attributes = np.ones(len(self.attribute_names)).astype(int).tolist()
        for i, line in enumerate(data):
            self.instance_names.append("Data record %d" %i)
        self.original_data = np.array(data)
        self.data = copy(self.original_data)
        self.label_name = self.attribute_names[self.label_index]
        self.ignored_attributes.append(self.label_name)
        self.update_data()


    def read_in_timeseries(self, path, parent):
        try:
            self.w = PopupAutocorrellationSlider('How many Autocorrellation lags do you want to consider?', 2, 1, 50)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 0
        except:
            msg = "WTF man??!"
            # QMessageBox.about(parent, "Big error", msg)
        autoCorrelLags = num
        try:
            self.w = PopupAutocorrellationSlider('How much lag for AR in the ARMA process do you want to consider?', 1, 0, 5)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 0
        except:
            msg = "WTF man??!"
            # QMessageBox.about(parent, "Big error", msg)
        arProcessLags = num
        try:
            min = 0
            if arProcessLags == 0:
                min = 1
            self.w = PopupAutocorrellationSlider('How much lag for MA in the ARMA process do you want to consider?', 1, min, 5)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 0
        except:
            msg = "WTF man??!"
            # QMessageBox.about(parent, "Big error", msg)
        maProcessLags = num
        try:
            self.w = PopupAutocorrellationSlider('How many top frequencies of a DTF do you want to consider?', 1, 0, 10)
            self.w.exec_()
            num = int(self.w.slider_value)
            if num == '':
                num = 0
        except:
            msg = "WTF man??!"
            # QMessageBox.about(parent, "Big error", msg)
        kFrequencies = num
        path = str(path)
        self.dataset_name = path.split('/')[-1][:-4]
        rawdata = open(path).readlines()
        self.database_type = 'TIMESERIES'
        attributeNames = [  'Max value',
                            'Min value',
                            'Length',
                            'Average', 
                            'Std', 
                            'Global trend',
                            'Smoothness', # aka Lag 1
                            ]        
        for i in range(0, arProcessLags):
            attributeNames.append('ARIMA AR Param for t - '+str(i))
        for i in range(0, maProcessLags):
            attributeNames.append('ARIMA MA Param for t - '+str(i))
        for i in range(1, autoCorrelLags):
            attributeNames.append('Autocorrellation Lag '+str(i+1))
        for i in range(0, kFrequencies):
            attributeNames.append("Freq. of DFT no "+str(i+1))
        attributeNames.append('Label')                                         
        self.attribute_names = np.array(attributeNames)
        self.original_data = np.random.random((len(rawdata), len(self.attribute_names)))
        for i, line in enumerate(rawdata):
            parts = line.rstrip().split(',')
            self.instance_names.append(parts.pop(0)) # Name
            self.original_data[i,-1] = parts.pop(0) # Label
            row = [float(x) for x in parts]
            self.original_data[i,0] = np.max(row)
            self.original_data[i,1] = np.min(row)
            self.original_data[i,2] = float(len(row))
            self.original_data[i,3] = np.average(row)
            self.original_data[i,4] = np.std(row)
            # trend
            m,b = np.polyfit(range(len(row)), row, 1)
            self.original_data[i,5] = m
            #statsmodels ARIMA parameters           
            iFail = 1
            arparams = [0.0]*arProcessLags
            maparams = [0.0]*maProcessLags
            #with suppress_stdout_stderr():
            while iFail > 0 and iFail < 3:                
                try:
                    armaModel = ARMA(row, (arProcessLags,maProcessLags)).fit() #ARMA_Model.fit().params
                    arparams = armaModel.arparams
                    maparams = armaModel.maparams
                    iFail = 0   
                except:
                    iFail += 1
            for s in range(arProcessLags):
                self.original_data[i,s+6] = arparams[s]
            for s in range(maProcessLags):
                self.original_data[i,s+6+arProcessLags] = maparams[s]
            # lagged autocorrellation
            for s in range(autoCorrelLags):
                ringshifted_seq = np.roll(row, -(s+1))
                self.original_data[i,arProcessLags+maProcessLags+s+6] = np.corrcoef(row, ringshifted_seq)[0,1]
            self.raw_data.append(row)
            #k-frequencies of a DFT
            dft = np.fft.fft(row)
            idxs = np.argpartition(dft, -1*kFrequencies)[-1*kFrequencies:]        
            freqs = np.fft.fftfreq(len(row))
            freqs = freqs[idxs[np.argsort(dft[idxs])]]
            for s in range(kFrequencies):
                self.original_data[i,arProcessLags+maProcessLags+autoCorrelLags+s+6] = freqs[s]
            if np.isnan(self.original_data[i]).any():
                print("Nan in line ",i,":")
                print(self.original_data[i])
        self.data = copy(self.original_data)
        self.label_name = 'Label'
        self.ignored_attributes.append('Label')        
        self.update_data()



    def read_in_csv(self, path):
        path = str(path)
        self.dataset_name = path.split('/')[-1][:-4]
        rawdata = open(path).readlines()
        if rawdata[0].rstrip() == 'LINK':
            self.database_type = 'IMAGE'
            from PIL import Image
            rawdata.pop(0)
            self.attribute_names = []
            self.considered_attributes = np.array([])
            for line in rawdata:
                name, pic_path = line.split(',')
                self.instance_names.append(name)
                pic = "/".join(path.split('/')[:-1]) + "/" + pic_path.rstrip()
                self.pictures.append(Image.open(pic).convert("L"))
                im = Image.open(pic).convert("L").resize((40,40), Image.ANTIALIAS)
                data = list(np.array(im.getdata()).T.astype(float))
                self.original_data.append(data)
            self.original_data = np.array(self.original_data)
            self.data = copy(self.original_data)
            self.label_name = None
        else:
            self.database_type = 'DATA'
            df = pd.read_csv(path, sep=None, engine="python").dropna(axis=0)

            # Get the values of the first column for instance names
            self.instance_names = list(df.iloc[:, 0].astype(str))  # assuming the first column of df holds instance names

            # Drop the first column, regardless of its content
            df = df.iloc[:, 1:]._get_numeric_data().astype(float)  # Select all rows and start from the second column
            self.attribute_names = np.array(df.columns)  # get column names
            self.considered_attributes = np.ones(len(self.attribute_names)).astype(int).tolist()
            self.original_data = df.values  # store the DataFrame values in original_data
        
            stds = np.std(self.original_data, axis=0)
            keeper = list(np.nonzero(stds)[0])
            self.attribute_names = self.attribute_names[keeper]
            self.original_data = self.original_data.T[keeper].T

            self.data = copy(self.original_data)
            self.label_name = self.attribute_names[self.label_index]
            self.ignored_attributes.append(self.label_name)
            self.update_data()


    def get_mask_by_constraint(self, name, constraint, value):
        try:
            value = float(value)
            index = self.attribute_names.tolist().index(name)
            values = self.original_data.T[index]
            if constraint == '>':
                return values>value
            elif constraint == '<':
                return values<value
            elif constraint == '=':
                return values==value
        except:
            print("That constraint is unknown. Try >,<,=")
            return np.ones(len(values)).astype(bool)


    def set_attribute_as_label(self, name):
        self.label_index = self.attribute_names.tolist().index(name)
        self.label_name = name
        self.update_data()


    def get_next_label_name(self):
        cur_name = self.attribute_names[self.label_index]
        sorted_names = list(sorted(self.attribute_names[:-1])) + [self.attribute_names[-1]]
        index = sorted_names.index(cur_name)
        name = ""
        if index == len(sorted_names) - 1:
            index = list(self.attribute_names).index(sorted_names[0])
            name = self.attribute_names[index]
        else:
            index += 1
            name = sorted_names[index]
        return name


    def ignore_attributes(self, names):
        for name in names:
            self.ignored_attributes.append(name)
        self.ignored_attributes = list(set(self.ignored_attributes))
        self.update_data()


    def unignore_attributes(self, names):
        for name in names:
            if name in self.ignored_attributes:
                self.ignored_attributes.pop(self.ignored_attributes.index(name))
        self.update_data()


    def update_data(self):
        self.mask = []
        for i, name in enumerate(self.attribute_names):
            if name not in self.ignored_attributes:
                self.mask.append(i)
        self.masked_names = [self.attribute_names[i] for i in self.mask]
        self.data = copy(self.original_data.T[self.mask].T)
        self.normalize()
        if np.isnan(self.data).any():
            print("Nan in copied data after normalizing:")
            print(self.original_data)


    def get_labels(self):
        return copy(self.original_data.T[self.label_index])


    def get_range(self, name):
        index = self.attribute_names.tolist().index(name)
        return min(self.original_data.T[index]), max(self.original_data.T[index])


