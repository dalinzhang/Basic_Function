from butter_filter import butter_bandpass_filter
import numpy as np
import scipy.io as sio
from scipy import interp

from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, auc, roc_curve
from sklearn.feature_selection import mutual_info_classif, SelectPercentile


class FBCSP(object):
	def __init__(self,
				 sample_rate,
				 feat_sel_proportion=0.8,
				 low_cut_hz = 4,
				 high_cut_hz = 36,
				 step = 4,
				 csp_components = 4
				 ):

		self.low_cut_hz = low_cut_hz
		self.high_cut_hz = high_cut_hz
		self.step = step
		self.sample_rate = sample_rate
		self.csp_component = csp_components
		self.feat_proportion = feat_sel_proportion
		self.csp_bank = dict()
		self.low = dict()
		self.high = dict()
		self.n_bank = (self.high_cut_hz - self.low_cut_hz)//self.step
		self.n_feat = int(self.n_bank*self.csp_component*self.feat_proportion)

		for i in range(self.n_bank):
			self.low[i]  = self.low_cut_hz+i*self.step
			self.high[i] = self.low_cut_hz+i*self.step+self.step
			if (self.high_cut_hz - self.high[i]) < self.step:
				self.high[i] = self.high_cut_hz

				
	def fit(self, data, label):
		data_bank = dict()
		for i in range(self.n_bank):
			# get each freq filter bank 
			data_bank[i] = self.bank_filter(data, self.low[i], self.high[i], self.sample_rate)
			# extract csp feature for each bank 
			self.csp_bank[i] = CSP(n_components=self.csp_component, reg=None, log=True, norm_trace=False)
			self.csp_bank[i].fit(data_bank[i], label)


	def transform(self, data):
		data_bank = dict()
		csp_feat = dict()
		for i in range(self.n_bank):
			# get each freq filter bank 
			data_bank[i] = self.bank_filter(data, self.low[i], self.high[i], self.sample_rate)
			# extract csp feature for each bank 
			csp_feat[i] = self.csp_bank[i].transform(data_bank[i])
			try:
				feature
			except NameError:
				feature = csp_feat[i]
			else:
				feature = np.hstack([feature, csp_feat[i]])
		return feature
	
	
	def fit_transform(self, data, label):
		data_bank = dict()
		csp_feat = dict()
		for i in range(self.n_bank):
			# get each freq filter bank 
			data_bank[i] = self.bank_filter(data, self.low[i], self.high[i], self.sample_rate)
			# extract csp feature for each bank 
			self.csp_bank[i] = CSP(n_components=4, reg=None, log=True, norm_trace=False)
			self.csp_bank[i].fit(data_bank[i], label)
			csp_feat[i] = self.csp_bank[i].transform(data_bank[i])
			try:
				feature
			except NameError:
				feature = csp_feat[i]
			else:
				feature = np.hstack([feature, csp_feat[i]])
		return feature


	def bank_filter(self, data, low_cut_hz, high_cut_hz, sample_rate):
		n_trial		= data.shape[0]
		n_channel	= data.shape[1]
		n_length	= data.shape[2]
		data_bank	= []
		for i in range(n_trial):
			data_bank += [np.array([butter_bandpass_filter(data[i, j, :], low_cut_hz, high_cut_hz, sample_rate, pass_type = 'band', order=6) 
							for j in range(n_channel)])]
		return np.array(data_bank)


	def classifier_fit(self, feature, label):
		# feature selection
		self.MI_sel = SelectPercentile(mutual_info_classif, percentile=self.feat_proportion*100)
		self.MI_sel.fit(feature, label)
		new_feat = self.MI_sel.transform(feature)
		# classification
		self.clf = LinearDiscriminantAnalysis()
		self.clf.fit(new_feat, label)


	def classifier_transform(self, feature):
		# feature selection
		new_feat = self.MI_sel.transform(feature)
		# classification
		return self.clf.transform(new_feat)


	def evaluation(self, feature, label):
		# feature selection
		new_feat = self.MI_sel.transform(feature)
		# accuracy
		accuracy = self.clf.score(new_feat, label)
		# f1
		f1 = dict()
		pred = self.clf.predict(new_feat)
		f1["micro"] = f1_score(y_true = label, y_pred = pred, average='micro')
		f1["macro"] = f1_score(y_true = label, y_pred = pred, average='macro')
		# auc
		pred_posi = self.clf.decision_function(new_feat)
		lb = LabelBinarizer()
		test_y = lb.fit_transform(label)
		roc_auc = self.multiclass_roc_auc_score(y_true = test_y, y_score = pred_posi)
		return accuracy, f1, roc_auc


	def multiclass_roc_auc_score(self, y_true, y_score):
		assert y_true.shape == y_score.shape
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		n_classes = y_true.shape[1]
		# compute ROC curve and ROC area for each class
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])
		# compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	
		# compute macro-average ROC curve and ROC area
		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		# Finally average it and compute AUC
		mean_tpr /= n_classes
		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		return roc_auc


if __name__ == "__main__":
	subject_id = 1
	data_folder = "/home/dadafly/program/19AAAI/bci_data/data_folder/cross_sub/"
	data = sio.loadmat(data_folder+"cross_subject_data_"+str(subject_id)+".mat")
	
	train_x = data["train_x"]
	test_x = data["test_x"]
	
	train_y = data["train_y"].ravel()
	test_y = data["test_y"].ravel()

	model = FBCSP(sample_rate = 250,
				  feat_sel_proportion=0.8,
				  low_cut_hz = 4,
                  high_cut_hz = 36,
                  step = 4,
                  csp_components = 4
				  )

	model.fit(train_x, train_y)
	train_feat = model.transform(train_x)
	test_feat = model.transform(test_x)

	model.classifier_fit(train_feat, train_y)
	train_acc, train_f1, train_auc = model.evaluation(train_feat, train_y)
	test_acc, test_f1, test_auc = model.evaluation(test_feat, test_y)
	print("#########################################################")
	print("train accuracy: ", train_acc)
	print("train f1 micro: ", train_f1["micro"])
	print("train f1 macro: ", train_f1["macro"])
	print("train auc micro: ", train_auc["micro"])
	print("train auc macro: ", train_auc["macro"])
	print("#########################################################")
	print("test accuracy: ", test_acc)
	print("test f1 micro: ", test_f1["micro"])
	print("test f1 macro: ", test_f1["macro"])
	print("test auc micro: ", test_auc["micro"])
	print("test auc macro: ", test_auc["macro"])
	print("#########################################################")




