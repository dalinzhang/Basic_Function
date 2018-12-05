'''
import self-defined module
'''
from multiclass_roc_auc import multiclass_roc_auc

'''
import public module
'''
import numpy as np
import scipy.io as sio
from scipy import interp

from sklearn.svm import SVC
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, auc, roc_curve

from mne.decoding import CSP


subject_id = 1
data_folder = "/home/dalinzhang/scratch/datasets/BCICIV_2a_gdf/cross_sub/"
data = sio.loadmat(data_folder+"cross_subject_data_"+str(subject_id)+".mat")

'''
train/test_x has shape [n_samples, n_channels, n_timelength]
train/test_y has shape [n_samples]
'''
train_x = data["train_x"]
test_x = data["test_x"]

train_y = data["train_y"].ravel()
test_y = data["test_y"].ravel()

cv = ShuffleSplit(10, test_size=0.2, random_state=33)

# Assemble a classifier
'''
define csp
'''
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

'''
define classifier
'''
lda = LinearDiscriminantAnalysis()
# svc = SVC(kernel='rbf', C=0.1)#, gamma=0.001)
# dt = tree.DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=5)
# rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=20, random_state=33)
# knn = KNeighborsClassifier(n_neighbors=5)

'''
define pipeline
'''
# clf = Pipeline([('CSP', csp), ('SVC', svc)])
# clf = Pipeline([('CSP', csp), ('KNN', knn)])
clf = Pipeline([('CSP', csp), ('LDA', lda)])

'''
fit and predict
'''
clf.fit(train_x, train_y)

cv_score = cross_val_score(clf, train_x, train_y, cv=cv, n_jobs=1)
test_score = clf.score(test_x, test_y)

test_pred = clf.predict(test_x)
f1_micro = f1_score(y_true = test_y, y_pred = test_pred, average='micro')
f1_macro = f1_score(y_true = test_y, y_pred = test_pred, average='macro')

'''
predict decision score
'''
pred_posi = clf.decision_function(test_x)
lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)
roc_auc = multiclass_roc_auc_score(y_true = test_y, y_score = pred_posi)

# Printing the results
print("#######################################################################################")
print("subject # ", subject_id)
print("test Classification accuracy:", np.mean(test_score))
print("cv Classification accuracy:", np.mean(cv_score))
print("test micro f1:", f1_micro)
print("test macro f1:", f1_macro)
print("test micro auc_roc:", roc_auc["micro"])
print("test macro auc_roc:", roc_auc["macro"])
print("#######################################################################################")
