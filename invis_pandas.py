from sklearn import datasets
import pandas as pd
import sys
sys.path.append('/home/daniel/InVis')
from Main import start_InVis

iris = datasets.load_iris()
data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
data['Species'] = iris['target']

results = start_InVis(data)