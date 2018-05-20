import numpy as np
from sklearn.decomposition import PCA
import csv
from sklearn.externals import joblib


pca = PCA(n_components=256)

total_list = []

for x in xrange(1,2):
    print x
    seg = []
    with open('all_data_pca/all_features_pca_256_%d.csv'%(x), 'rU') as f:
        reader = csv.reader(f)
        reader = csv.reader(f, delimiter=',')
        seg = list(reader)
    total_list += seg

print total_list

data = np.asarray(total_list)

print data.shape

pca_model = pca.fit(data)
joblib.dump(pca_model, 'pca.pkl')

