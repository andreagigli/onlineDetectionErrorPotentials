#!/usr/bin/env python

from __future__ import division

from scipy.io import loadmat, savemat

import numpy as np

import matplotlib.pyplot as plt

from scipy import signal
from spectrum import pburg

from sklearn.linear_model import Lasso, LassoCV, SGDClassifier, RidgeClassifier, RidgeClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, FastICA
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_class_weight

from joblib import Parallel, delayed
import multiprocessing

import pywt

import time




###############################################################################
###################################  FUNCTIONS ################################
###############################################################################



def segment_eeg(execution_error, outcome_error, w_length):
    # Segment the raw data in trials of length w_length. 2 kinds of trials are to be detected: err_trials associated to
    # the beginning of an execution error (1 in execution_error), and noErr_trials of length w_length placed in between
    # 2 consecutive errors (execution OR outcome) that are more than 3*w_length apart from each other.

    idx_X = []
    y = []

    # Detect execution_error trials
    idx_error = np.where(np.concatenate(([1],np.diff(execution_error))))[0]
    idx_error = idx_error[execution_error[idx_error]!=0]
    idx_X = np.concatenate((idx_error[:,None],(idx_error+w_length)[:,None]), axis=1)
    idx_X[-1,1] = np.clip(idx_X[-1,1],0,len(execution_error))
    y = np.ones(len(idx_error))

    # Detect noErr trials
    error = np.clip(execution_error + outcome_error, 0, 1)
    # get the 0-sequences in error that are longer than 3*w_length.
    idx_error = np.concatenate((np.where(np.concatenate(([1],np.diff(error)))!=0)[0], [len(error)]))
    if error[0]==0: # we're interested only in the lengths of the zero sequences
        tmp0 = idx_error[::2]
        tmp1 = idx_error[1::2]
    else:
        tmp0 = idx_error[1::2]
        tmp1 = idx_error[2::2]
    if len(tmp0) > len(tmp1): # check if error starts with zeros and end with ones (or viceversa)
        tmp0 = tmp0[:-1]
    idx_long_zero_seq = np.concatenate((tmp0[:,None],tmp1[:,None]),axis=1)
    idx_long_zero_seq = idx_long_zero_seq[idx_long_zero_seq[:,1]-idx_long_zero_seq[:,0]>=3*w_length]
    tmp = np.floor((idx_long_zero_seq[:,1]-idx_long_zero_seq[:,0])/3).astype(np.int)
    idx_long_zero_seq[:,0] += tmp
    idx_long_zero_seq[:,1] = idx_long_zero_seq[:,0]+w_length


    idx_X = np.concatenate((idx_X, idx_long_zero_seq), axis=0)
    y = np.concatenate((y, np.zeros(len(idx_long_zero_seq[:,0]))))

    # Sort idx_X and y on wrt the first column of idx_X
    tmp = np.argsort(idx_X[:,0], axis=0)
    idx_X = (idx_X[tmp,:]).astype(np.int)
    y = y[tmp]

    # # Debug: plot execution_error, outcome_error, err_trials, noErr_trials
    # plt.figure()
    # ax = plt.subplot(111)
    # plt.plot(execution_error, 'b')
    # plt.plot(outcome_error, 'g')
    # for k,(i,j) in enumerate(idx_X):
    #     if y[k]==0:
    #         color = 'green' # noErr trial
    #     else:
    #         color = 'red' # err trial
    #     ax.axvspan(i,j,alpha=0.5,color=color)
    # plt.show()

    return idx_X, y



def segment_eeg_asynchronous(execution_error, w_length, stride):
    
    # idx_error = np.where(np.concatenate(( [1], np.diff(execution_error) ))!=0)[0]
    # idx_error = idx_error[execution_error[idx_error]!=0]
    # execution_error = np.zeros(len(execution_error))
    # execution_error[idx_error] = 1 # error = bool vector with ones in correspondance of the beginnings of execution errors

    # segment data
    n_samples = 1 + np.floor((len(execution_error)-w_length)/stride).astype(np.int)
    assert n_samples>0
    idx_X = np.empty((n_samples ,2), dtype=np.int)
    idx_X[:,0] = range(0, (n_samples-1)*stride+1, stride)
    idx_X[:,1] = idx_X[:,0] + w_length
    y = np.zeros(n_samples, dtype=np.int8)

    # Set label=1 for the error trials (trials that contain errors)
    idx_error_trials = [k for k,(i,j) in enumerate(idx_X) if np.sum(execution_error[i:j])>0]
    y[idx_error_trials] = 1

    # noError trials: if the trial does not contain an error
    # y[..complementary to the error trials..] = 0 , but they are already zero by creation

    return idx_X, y



def segment_eeg_asynchronous_new(execution_error, fs, w_length, stride):

    # Correct true labels s.t. execution_error==1 only between 100ms and 600ms after the onset of an error
    idx_error = np.where(np.concatenate(( [1], np.diff(execution_error) ))!=0)[0]
    idx_error = idx_error[execution_error[idx_error]!=0]
    idx_error = np.concatenate((idx_error[:,None]+int(0.1*fs),idx_error[:,None]+int(0.6*fs)), axis=1)
    execution_error_corrected = np.zeros(len(execution_error))
    for (i,j) in idx_error:
        execution_error_corrected[i:j] = 1

    # segment data
    n_samples = 1 + np.floor((len(execution_error_corrected)-w_length)/stride).astype(np.int)
    assert n_samples>0
    idx_X = np.empty((n_samples ,2), dtype=np.int)
    idx_X[:,0] = range(0, (n_samples-1)*stride+1, stride)
    idx_X[:,1] = idx_X[:,0] + w_length
    y = np.ones(n_samples, dtype=np.int8)*(-1)

    # Set label=1 for the error trials (trials that contain whole batches of error trials)
    idx_error_trials = [k for k,(i,j) in enumerate(idx_X) if execution_error_corrected[i]==0 and execution_error_corrected[j-1]==0 and np.sum(execution_error_corrected[i:j])>0]
    y[idx_error_trials] = 1

    # Set label=0 for the no error trials (trials that do not contain any part of an error)
    idx_noerror_trials = [k for k,(i,j) in enumerate(idx_X) if np.sum(execution_error_corrected[i:j])==0]
    y[idx_noerror_trials] = 0

    # Eliminate ambiguous trials (trials containing only part of errors)
    idx_X = idx_X[y!=-1,:]
    y = y[y!=-1]

    return idx_X, y



def split_instances(n_samples, trainsplit, testsplit):
    n_splits = len(trainsplit)+len(testsplit)
    idx_split = np.arange(0, np.floor(n_samples/n_splits)*n_splits, np.floor(n_samples/n_splits))
    idx_split = np.concatenate((idx_split, [n_samples])).astype(np.int)

    idx_train = []
    idx_test = []
    for i in trainsplit:
        idx_train = np.concatenate((idx_train, range(idx_split[i], idx_split[i+1])), axis=0).astype(np.int)
    for i in testsplit:
        idx_test = np.concatenate((idx_test, range(idx_split[i], idx_split[i+1])), axis=0).astype(np.int)

    return idx_train, idx_test



def burgSpectrum(input, subwind_length, fs): # for parallel burg
    return pburg(input, subwind_length, criteria='AKICc', NFFT=int(fs), sampling=int(fs)).psd



def mdwt(input, wavelet, n_dec_lev):
    coeffs = pywt.wavedec(input, wavelet, level=n_dec_lev)
    marginals = np.empty(n_dec_lev+1)
    for l in range(n_dec_lev+1):
        marginals[l] = np.sum(np.abs(coeffs[l])) # sum
        # marginals[l] = np.max(coeffs[l]) # max
    return marginals



def multichannel_mdwt(inputs, n_channels, wavelet, n_dec_lev):
    marginals = np.empty((n_channels, n_dec_lev+1))
    for c in range(n_channels):
        marginals[c] = mdwt(inputs[:,c], wavelet, n_dec_lev)
    return marginals.flatten() # all marginals for all channels flattened (n_dec_lev+1)*n_channel



def customflatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(customflatten(el))
        else:
            result.append(el)
    return result



def multichannel_dwt_new(inputs, n_channels, wavelet, n_dec_lev, marginalize=True):
    # extraction of the dwt coeffs for each channel
    coeffs = []
    for c in range(n_channels):
        coeffs.append(pywt.wavedec(inputs[:,c], wavelet, level=n_dec_lev)) # coeffs is a list of lists coeffs[n_channels][n_dec_lev]
    # marginalization
    for c in range(n_channels):
        for d in range(n_dec_lev+1):
            coeffs[c][d] = np.sum(np.abs(coeffs[c][d])) # sum of the absolute values of the coeffs at decomposition level d
            # coeffs[c][d] = np.max(coeffs[c][d]) # maximum of the coeffs at decomposition level d
    # concatenate the marginals/coeffs of the various channels and return them
    if marginalize==True:
        return np.concatenate(coeffs).ravel()
    else:
        return np.array(customflatten(coeffs))



def extract_features(data, idx_samples, feature_type,
                     w_lower=None, w_upper=None, temporal_subsample_step=None,
                     fs=None, psd_method=None,max_relevant_freq=None):

    # checks
    assert 'temporal' in feature_type or 'spectral' in feature_type or 'mdwt' in feature_type
    if 'temporal' in feature_type:
        assert w_lower is not None and w_upper is not None and temporal_subsample_step is not None
    if 'spectral' in feature_type:
        assert fs is not None and (psd_method=='welch' or psd_method=='burg') and max_relevant_freq is not None

    # Define output structures
    n_channels = data.shape[-1]
    X = np.empty(0)
    subwind_length = w_upper - w_lower

    if 'temporal' in feature_type:
        tmp = np.empty((idx_samples.shape[0], n_channels*(1+np.floor((w_upper-w_lower-1)/temporal_subsample_step).astype(np.int))))
        X = np.hstack((X,tmp)) if X.size else tmp
        tmp_idx_samples = np.empty((idx_samples.shape[0], 2)) # adjust the window: [w_lower, w_upper] instead of [0,w_length]
        tmp_idx_samples[:,0] = idx_samples[:,0] + w_lower
        tmp_idx_samples[:,1] = idx_samples[:,0] + w_upper # [:,0] is correct!
        tmp_idx_samples = tmp_idx_samples.astype(np.int)
        for k,(i,j) in enumerate(tmp_idx_samples):
            X[k,:] = data[i:j:temporal_subsample_step, :].ravel()
    if 'spectral' in feature_type:
        init_dim_X = X.shape[-1]
        tmp = np.empty(( idx_samples.shape[0], n_channels*max_relevant_freq ))
        X = np.hstack(( X, tmp )) if X.size else tmp
        tmp_idx_samples = np.empty((idx_samples.shape[0], 2)) # adjust the window: [w_lower, w_upper] instead of [0,w_length]
        tmp_idx_samples[:,0] = idx_samples[:,0] + w_lower
        tmp_idx_samples[:,1] = idx_samples[:,0] + w_upper # [:,0] is correct!
        tmp_idx_samples = tmp_idx_samples.astype(np.int)
        if psd_method == 'welch': # welch PSD
            nperseg = np.minimum( 64,2**(np.floor(np.log2(subwind_length)).astype(np.int)-1)) # length of the welch segments. 64 samples or less, depending on the length of the instance.
            noverlap = int(nperseg/2) # overlap = 50%
            for k,(i,j) in enumerate(tmp_idx_samples):
                _, tmp = signal.welch(data[i:j,:], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=fs, return_onesided=True, axis=0) # fs=fs and nfft=fs implies freq_resolution=1
                X[k,init_dim_X:] = tmp[1:max_relevant_freq+1,:].ravel(order='F') # being freq_resolution=1, tmp[1:max_relevant_freq+1] takes the power of exactly the first max_relevant_freq frequences
        else: # burg psd
            n_cores = np.maximum(1, multiprocessing.cpu_count()) # multicore burg
            for k,(i,j) in enumerate(tmp_idx_samples):
                X[k,init_dim_X:] = np.array(Parallel(n_jobs=n_cores)(delayed(burgSpectrum)(sig, subwind_length, fs) for sig in list(np.transpose(data[i:j,:]))))[:,0:max_relevant_freq].flatten()
                # print(k) # debug
    if 'mdwt' in feature_type:
        n_cores = np.maximum(1, multiprocessing.cpu_count() ).astype(int) # multicore mdwt
        wavelet = 'db5' #sym5, db5
        n_dec_lev = 3
        # n_dec_lev = pywt.dwt_max_level(idx_samples[0,1]-idx_samples[0,0], pywt.Wavelet(wavelet)) # max decomposition level
        init_dim_X = X.shape[-1]
        tmp = np.empty(( idx_samples.shape[0], n_channels*(n_dec_lev+1) ))
        X = np.hstack(( X, tmp )) if X.size else tmp
        # parallelize over trials
        trials = []
        for (i,j) in idx_samples:
            trials.append(data[i:j,:])
        # start_time = time.time()
        X[:, init_dim_X:] = np.array( Parallel ( n_jobs=n_cores ) ( delayed(multichannel_dwt_new)(trial,n_channels,wavelet,n_dec_lev) for trial in trials ) )
        # print('Time for feature extraction: '+str(time.time()-start_time))
    return X


def select_relevant_features(X_train, X_test, bestk=20, method='PCA', label=None):

    if method=='mutualinfo':
        assert label is not None
        idx_relevant_features = (SelectKBest(mutual_info_classif, k=bestk).fit(X_train, label.ravel())).get_support(indices=True)
        X_train = X_train[:,idx_relevant_features]
        X_test = X_test[:,idx_relevant_features]

    if method=='PCA':
        tmp = PCA(n_components=bestk).fit(X_train)
        X_train = tmp.transform(X_train)
        X_test = tmp.transform(X_test)
        # print(tmp.explained_variance_ratio_*100.)

    elif method=='ICA':
        tmp = FastICA(n_components=bestk).fit(X_train)
        X_train = tmp.transform(X_train)
        X_test = tmp.transform(X_test)

    elif method=='R2':
        assert label is not None
        r2 = np.zeros(X_train.shape[1])
        idx0 = label==0
        idx1 = label==1
        m = X_train.shape[0]
        m0 = np.sum(idx0)
        m1 = np.sum(idx1)
        for j in range(X_train.shape[1]):
            r2[j] = ( (np.sum(X_train[idx1,j]))**2/m1 + (np.sum(X_train[idx0,j]))**2/m0 - (np.sum(X_train[:,j]))**2/m ) / \
                    ( np.sum(X_train[idx1,j]**2) + np.sum(X_train[idx0,j]**2) - (np.sum(X_train[:,j]))**2/m )
        idx_relevant_features = np.argsort(r2)[::-1]
        idx_relevant_features = idx_relevant_features[:bestk]
        X_train = X_train[:,idx_relevant_features]
        X_test = X_test[:,idx_relevant_features]

    return X_train, X_test



def trainmodel(X, y, model='linearSVM', hyp_opt=False, param_grid=None, validation_subsampling_rate=1):

    clf = None

    if model=='SVM':
        hyper_C = 1.0
        hyper_gamma = 'auto'
        # hyperparameter optimization
        if hyp_opt==True:
            assert param_grid!=None and param_grid.has_key('kernel') and param_grid.has_key('gamma') and param_grid.has_key('C'), \
                'ERROR: please provide the grid to optimize the hyperparameters gamma and C of the linearSVM. The grid must be a dictionary like {\'kernel\': [\'rbf\'], \'gamma\': (numlist), \'C\': (numlist)}'
            if len(param_grid)>3:
                param_grid = {'C':param_grid['C'], 'kernel':param_grid['kernel'], 'gamma':param_grid['gamma']}
            clf = GridSearchCV(SVC(), param_grid, cv=4, n_jobs=-1, verbose=1)
            clf.fit(X[::validation_subsampling_rate,:], y[::validation_subsampling_rate])
            hyper_C = clf.best_params_['C']
            hyper_gamma = clf.best_params_['gamma']
            print('best C: '+ str(hyper_C))
            print('best gamma: '+ str(hyper_gamma))
        # Training
        clf = SVC(kernel='rbf',C=hyper_C,gamma=hyper_gamma)
        clf.fit(X, y)

    elif model=='weightedSVM':
        hyper_C = 1.0
        hyper_gamma = 'auto'
        class_weights = np.array([1/(2*np.sum(y==0)/len(y)),1/(2*np.sum(y==1)/len(y))])
        # hyperparameter optimization
        if hyp_opt==True:
            assert param_grid!=None and param_grid.has_key('kernel') and param_grid.has_key('gamma') and param_grid.has_key('C'), \
                'ERROR: please provide the grid to optimize the hyperparameters gamma and C of the linearSVM. The grid must be a dictionary like {\'kernel\': [\'rbf\'], \'gamma\': (numlist), \'C\': (numlist)}'
            if len(param_grid)>3:
                param_grid = {'C':param_grid['C'], 'kernel':param_grid['kernel'], 'gamma':param_grid['gamma']}
            # clf = GridSearchCV(SVC(class_weight={0: class_weights[0], 1: class_weights[1]}), param_grid, cv=4, n_jobs=-1, verbose=1)
            clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=4, n_jobs=-1, verbose=1)
            clf.fit(X[::validation_subsampling_rate,:], y[::validation_subsampling_rate])
            hyper_C = clf.best_params_['C']
            hyper_gamma = clf.best_params_['gamma']
            print('best C: '+ str(hyper_C))
            print('best gamma: '+ str(hyper_gamma))
        # Training
        clf = SVC(kernel='rbf',C=hyper_C,gamma=hyper_gamma, class_weight='balanced',probability=True)
        clf.fit(X, y)

    elif model=='linearSVM':
        hyper_C = 1.0
        # hyperparameter optimization
        if hyp_opt==True:
            assert param_grid!=None and param_grid.has_key('C'), \
                'ERROR: please provide the grid to optimize the hyperparameter C of the linearSVM. The grid must be a dictionary like {\'C\': (numlist)}'
            if len(param_grid)>1:
                param_grid = {'C':param_grid['C']}
            clf = GridSearchCV(SVC(kernel='linear'), param_grid, cv=4, n_jobs=-1, verbose=1)
            clf.fit(X[::validation_subsampling_rate,:], y[::validation_subsampling_rate])
            hyper_C = clf.best_params_['C']
            print('best C: '+ str(hyper_C))
        # Training
        clf = SVC(kernel='linear', C=hyper_C, random_state=0, probability=True)
        clf.fit(X, y)

    elif model=='weightedLinearSVM':
        hyper_C = 1.0
        class_weights = np.array([1/(2*np.sum(y==0)/len(y)),1/(2*np.sum(y==1)/len(y))])
        # hyperparameter optimization
        if hyp_opt==True:
            assert param_grid!=None and param_grid.has_key('C'), \
                'ERROR: please provide the grid to optimize the hyperparameter C of the linearSVM. The grid must be a dictionary like {\'C\': (numlist)}'
            if len(param_grid)>1:
                param_grid = {'C':param_grid['C']}
            # clf = GridSearchCV(LinearSVC(class_weight={0: class_weights[0], 1: class_weights[1]}), param_grid, cv=4, n_jobs=-1, verbose=1)
            clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid, cv=4, n_jobs=-1, verbose=1)
            clf.fit(X[::validation_subsampling_rate,:], y[::validation_subsampling_rate])
            hyper_C = clf.best_params_['C']
            print('best C: '+ str(hyper_C))
        # Training
        clf = SVC(kernel='linear', C=hyper_C, class_weight='balanced', probability=True)
        clf.fit(X, y)

    elif model == 'rls':
        hyper_alpha = 1.0
        if hyp_opt==True:
            assert param_grid!=None and param_grid.has_key('C'), \
                'ERROR: please provide a grid of C values to optimize the hyperparameter alpha=C^-1 of the Ridge Classifier. The grid must be a dictionary like {\'C\': (numlist)}'
            param_grid = {'alphas':np.reciprocal(param_grid['C'],dtype=np.float)} # overwrite param_grid
            clf = RidgeClassifierCV(alphas=param_grid['alphas'])
            clf.fit(X[::validation_subsampling_rate,:], y[::validation_subsampling_rate])
            hyper_alpha = clf.alpha_
            print('best alpha: '+ str(hyper_alpha))
        # Training
        clf = RidgeClassifier(alpha=hyper_alpha)
        clf.fit(X, y)

    elif model == 'weightedrls':
        hyper_alpha = 1.0
        if hyp_opt==True:
            assert param_grid!=None and param_grid.has_key('C'), \
                'ERROR: please provide a grid of C values to optimize the hyperparameter alpha=C^-1 of the Ridge Classifier. The grid must be a dictionary like {\'C\': (numlist)}'
            param_grid = {'alphas':np.reciprocal(param_grid['C'],dtype=np.float)} # overwrite param_grid
            clf = RidgeClassifierCV(alphas=param_grid['alphas'], class_weight='balanced')
            clf.fit(X[::validation_subsampling_rate,:], y[::validation_subsampling_rate])
            hyper_alpha = clf.alpha_
            print('best alpha: '+ str(hyper_alpha))
        # Training
        clf = RidgeClassifier(alpha=hyper_alpha, class_weight='balanced')
        clf.fit(X, y)

    elif model == 'cart':
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

    elif model=='sgdminibatch':
        clf = SGDClassifier(n_jobs=-1) # class_weight=compute_class_weight('balanced', np.unique(y), y) # !! non funziona perche vaffanculo
        for i in np.concatenate((np.arange(1,11)*np.floor(len(y)/10),[len(y)])).astype(np.int): # split the training set into 11 minibatches
            clf.partial_fit(X[i-1:i,:], y[i-1:i],classes=[0,1])

    return clf



def make_avg_statistics(y_true_vec, y_pred_vec, y_pred_conf_vec, n_splits, print_results=False):
    # average baseline accuracy
    bsl_acc = 0
    for y_true in y_true_vec:
        if sum(y_true==0) >= sum(y_true==1):
            bsl_acc = bsl_acc + sum(y_true==0).astype(np.float)/len(y_true)
        else:
            bsl_acc = bsl_acc + sum(y_true==1).astype(np.float)/len(y_true)
    bsl_acc = bsl_acc/n_splits*100
    if print_results==True:
        print('AVERAGE BASELINE ACCURACY (accuracy of predicting always the most numerous class, averaged over '+str(n_splits)+' splits): '+ str(bsl_acc) +'%')

    # average accuracy
    avg_acc = 0
    for y_true, y_pred in zip(y_true_vec, y_pred_vec):
        avg_acc += accuracy_score(y_true, y_pred)*100
    avg_acc /= n_splits
    if print_results==True:
        print('AVERAGE CLASSIFICATION ACCURACY (over '+str(n_splits)+' splits): ' + str(avg_acc) + '%')

    # confusion matrix
    conf_mat = np.zeros((2,2))
    for y_true, y_pred in zip(y_true_vec, y_pred_vec):
        conf_mat += confusion_matrix(y_true=y_true, y_pred=y_pred)
    conf_mat /= n_splits
    if print_results==True:
        print('AVERAGE CONFUSION MATRIX (over '+str(n_splits)+' splits)')
        print(conf_mat/np.sum(np.sum(conf_mat)))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Confusion Matrix')
        plt.imshow((conf_mat/np.sum(np.sum(conf_mat))).astype(np.float),origin='upper',aspect='auto', interpolation='None', vmin=0, vmax=1, cmap='winter')
        # add colorbar
        plt.colorbar(orientation='vertical')
        plt.show()
    # sensitivity and specificity
    sensitivity = conf_mat[1,1]/np.sum(conf_mat[1,:])
    specificity = conf_mat[0,0]/np.sum(conf_mat[0,:])

    # balanced accuracy
    bal_acc = (sensitivity + specificity)/2

    # # AUC of the ROC (TPR vs FPR)
    # avg_auc = 0
    # for y_true, y_pred_conf in zip(y_true_vec, y_pred_conf_vec):
    #     avg_auc += roc_auc_score(y_true, y_pred_conf)
    # avg_auc /= n_splits

    # AUC of the ROC (TNR Vs FNR)
    thresholds = np.arange(0,1.0001,0.0001)
    avg_roc = np.zeros((len(thresholds),2))
    avg_auc = 0
    for y_true, y_pred_conf in zip(y_true_vec, y_pred_conf_vec):
        y_pred_conf = np.exp(y_pred_conf)/np.sum(np.exp(y_pred_conf)) # comment this line if you are using SVM
        y_pred_conf /= np.max(y_pred_conf) # technically erroneous, because y_pred_conf will not sum up to 1 anymore # comment this line if you are using SVM
        tnr = np.zeros(len(thresholds))
        fnr = np.zeros(len(thresholds))
        for i in range(len(thresholds)):
            tmp_y_pred = np.zeros(len(y_pred_conf))
            tmp_y_pred[y_pred_conf>=thresholds[i]] = 1
            tnr[i] = np.count_nonzero(np.logical_and(y_true==0, tmp_y_pred==0))/np.count_nonzero(y_true==0)
            fnr[i] = np.count_nonzero(np.logical_and(y_true==1, tmp_y_pred==0))/np.count_nonzero(y_true==1)
        avg_auc += auc(fnr, tnr)
        avg_roc += np.concatenate((fnr[:,None],tnr[:,None]),axis=1)
    avg_auc /= n_splits
    avg_roc /= n_splits
    return bsl_acc, avg_acc, conf_mat, sensitivity, specificity, bal_acc, avg_auc, avg_roc








# def get_split_indices(idx_split, splits, n_splits, labels):
#     idx_corrected = np.zeros((len(splits),2),dtype=np.int64)
#     for n,idx in enumerate(splits):
#         idx_corrected[n,0] = idx_split[idx-1]
#         if idx!=n_splits:
#             idx_corrected[n,1] = idx_split[idx]-1
#         else:
#             idx_corrected[n,1] = len(labels)-1
#     return idx_corrected
# def splitdb(data, labels, n_splits, trainsplit, testsplit):
#     # Partition the data into n_splits splits ALONG THE FIRST AXIS. The idea is that the splits should have "comparable"
#     # lengths and that the split is made after a +1 sequence in labels.
#     idx_split = (np.arange(0,np.floor(data.shape[0]/n_splits)*n_splits,np.floor(data.shape[0]/n_splits))).astype(np.int64) # splits of the same length (except the last one)
#     idx_tmp = np.concatenate(([1],np.diff(labels,axis=0))).nonzero()[0]
#     if labels[0] == 0:
#         idx_tmp = idx_tmp[::2]
#     else:
#         idx_tmp = idx_tmp[1::2]
#     assert idx_split[0]>=idx_tmp[0] and idx_split[-1]<=idx_tmp[-1], 'ERROR: please choose a smaller number of splits'
#     idx_split=idx_tmp[np.searchsorted(idx_tmp,idx_split)]
#
#     # # debug
#     # a=idx_tmp[np.searchsorted(idx_tmp,idx_split)] # instead of idx_split=idx_tmp[np.searchsorted...]
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.hold(True)
#     # ax.plot(labels, label='label')
#     # tmp = np.zeros(labels.shape)
#     # tmp[idx_split] = 1
#     # ax.plot(tmp, label='split original')
#     # tmp = np.zeros(labels.shape)
#     # tmp[a] = 1
#     # ax.plot(tmp, label='split corrected')
#     # # ax.legend(loc='best')
#     # plt.show()
#
#     idx_train = get_split_indices(idx_split, trainsplit, n_splits, labels)
#     idx_test = get_split_indices(idx_split, testsplit, n_splits, labels)
#
#     return idx_train, idx_test
# def extract_time_features(data, idx_splits, fs, w_lower, w_upper, w_length, stride, subsample_step, labels):
#     # fs is given in Hz. w_upper, w_lower, stride are given in # of samples.
#
#     # # convert w_upper, w_lower, stride in number of samples
#     # assert w_lower>=0 and w_upper>w_lower and stride>0
#     # w_lower = np.floor(w_lower*fs).astype(np.int)
#     # w_upper = np.floor(w_upper*fs).astype(np.int)
#     # stride = np.floor(stride*fs).astype(np.int)
#
#     n_channels = data.shape[-1]
#     X = np.empty((0,n_channels*(1+np.floor((w_upper-w_lower-1)/subsample_step).astype(np.int))))
#     y = []
#
#     for i,j in idx_splits:
#
#         tmp_data = data[i:j+1,:] # portion of the data relative to the current split
#         tmp_labels = labels[i:j+1] # portion of the labels relative to the current split
#         length_tmp_labels = len(tmp_labels) # length of the current split
#         # pad data at the end
#         tmp_data = np.lib.pad(tmp_data, ((0,w_length-1),(0,0)), 'edge')
#         tmp_labels = np.lib.pad(tmp_labels, (0,w_length-1), 'edge')
#
#         # Create samples based on the sliding window.
#         # Relevant vectors for this operation. tmp_data and tmp_label = portions of the raw data vectors from which to read.
#         # X, y: vectors where to save the samples (to be expanded).
#         n_new_samples = 1+np.floor((length_tmp_labels-1)/stride).astype(np.int)
#         l = len(y) # from where to start storing new samples in X and y
#         X = np.concatenate((X,np.zeros((n_new_samples,X.shape[1])))) # allocate space for the new samples
#         y = np.concatenate((y,np.zeros(n_new_samples)))
#         for h, k in enumerate(range(0, length_tmp_labels, stride)): # slide the window along the raw data and create the samples
#             X[l+h,:] = tmp_data[k+w_lower:k+w_upper:subsample_step,:].ravel() # k = index of the inputs tmp_.. . h = progressive index of the outputs X and y.
#             y[l+h] = tmp_labels[k]
#
#     return X, y
# def extract_time_features2(data, idx_splits, fs, w_lower, w_upper, w_length, stride, subsample_step):
#     # fs is given in Hz. w_upper, w_lower, stride are given in # of samples.
#
#     # # convert w_upper, w_lower, stride in number of samples
#     # assert w_lower>=0 and w_upper>w_lower and stride>0
#     # w_lower = np.floor(w_lower*fs).astype(np.int)
#     # w_upper = np.floor(w_upper*fs).astype(np.int)
#     # stride = np.floor(stride*fs).astype(np.int)
#
#
#     n_channels = data.shape[-1]
#     X = np.empty((0,n_channels*2))
#     y = []
#
#
#     for i,j in idx_splits:
#
#         tmp_data = data[i:j+1,:] # portion of the data relative to the current split
#         tmp_labels = labels[i:j+1] # portion of the labels relative to the current split
#         length_tmp_labels = len(tmp_labels) # length of the current split
#         # pad data at the end
#         tmp_data = np.lib.pad(tmp_data, ((0,w_length-1),(0,0)), 'edge')
#         tmp_labels = np.lib.pad(tmp_labels, (0,w_length-1), 'edge')
#
#         # Create samples based on the sliding window.
#         # Relevant vectors for this operation. tmp_data and tmp_label = portions of the raw data vectors from which to read.
#         # X, y: vectors where to save the samples (to be expanded).
#         n_new_samples = 1+np.floor((length_tmp_labels-1)/stride).astype(np.int)
#         l = len(y) # from where to start storing new samples in X and y
#         X = np.concatenate((X,np.zeros((n_new_samples,X.shape[1])))) # allocate space for the new samples
#         y = np.concatenate((y,np.zeros(n_new_samples)))
#         for h, k in enumerate(range(0, length_tmp_labels, stride)): # slide the window along the raw data and create the samples
#             X[l+h,:n_channels] = np.sqrt(np.mean(np.square(tmp_data[k+w_lower:k+w_upper:subsample_step,:]), axis=0)) # RMS
#             X[l+h,n_channels:] = np.mean(tmp_data[k+w_lower:k+w_upper:subsample_step,:], axis=0) # MEAN # k = index of the inputs tmp_.. . h = progressive index of the outputs X and y.
#             y[l+h] = tmp_labels[k]
#     return X, y
# def extract_time_features3(data, idx_splits, fs, w_lower, w_upper, w_length, stride, subsample_step, labels):
#     # fs is given in Hz. w_upper, w_lower, stride are given in # of samples.
#
#     # # convert w_upper, w_lower, stride in number of samples
#     # assert w_lower>=0 and w_upper>w_lower and stride>0
#     # w_lower = np.floor(w_lower*fs).astype(np.int)
#     # w_upper = np.floor(w_upper*fs).astype(np.int)
#     # stride = np.floor(stride*fs).astype(np.int)
#
#
#     n_channels = data.shape[-1]
#     X = np.empty((0,1+np.floor((w_upper-w_lower-1)/subsample_step).astype(np.int)))
#     y = []
#
#
#     for i,j in idx_splits:
#
#         tmp_data = data[i:j+1,:] # portion of the data relative to the current split
#         tmp_labels = labels[i:j+1] # portion of the labels relative to the current split
#         length_tmp_labels = len(tmp_labels) # length of the current split
#         # pad data at the end
#         tmp_data = np.lib.pad(tmp_data, ((0,w_length-1),(0,0)), 'edge')
#         tmp_labels = np.lib.pad(tmp_labels, (0,w_length-1), 'edge')
#
#         # Create samples based on the sliding window.
#         # Relevant vectors for this operation. tmp_data and tmp_label = portions of the raw data vectors from which to read.
#         # X, y: vectors where to save the samples (to be expanded).
#         n_new_samples = 1+np.floor((length_tmp_labels-1)/stride).astype(np.int)
#         l = len(y) # from where to start storing new samples in X and y
#         X = np.concatenate((X,np.zeros((n_new_samples,X.shape[1])))) # allocate space for the new samples
#         y = np.concatenate((y,np.zeros(n_new_samples)))
#         for h, k in enumerate(range(0, length_tmp_labels, stride)): # slide the window along the raw data and create the samples
#             X[l+h,:] = np.mean(tmp_data[k+w_lower:k+w_upper:subsample_step,:], axis=1) # MEAN OVER CHANNELS # k = index of the inputs tmp_.. . h = progressive index of the outputs X and y.
#             y[l+h] = tmp_labels[k]
#     return X, y
# def extract_spectrogram_features(data, idx_splits, fs, w_length, stride, max_interest_freq, label_data):
#     # w_length and stride in # of samples. w_percent_up in [0,1]
#
#     # # fs is given in Hz. w_upper, w_lower, stride are given in seconds.
#     #
#     # assert max_interest_freq <= fs/2
#     #
#     # # convert w_upper, w_lower, stride in number of samples
#     # assert w_lower>=0 and w_upper>w_lower and stride>0
#     # w_lower = np.floor(w_lower*fs).astype(np.int)
#     # w_upper = np.floor(w_upper*fs).astype(np.int)
#     # stride = np.floor(stride*fs).astype(np.int)
#
#
#     # # create the window
#     # n_up_values = np.floor(w_length * w_percent_up)
#     # window = np.zeros(w_length)
#     # if w_length%2==0:
#     #     if n_up_values%2!=0:
#     #         n_up_values-=1
#     #     window[w_length/2:n_up_values/2:w_length/2+n_up_values/2]=1
#     # if w_length%2!=0:
#     #     if n_up_values%2==0:
#     #         n_up_values-=1
#     #     window[(w_length-1)/2:(n_up_values-1)/2:(w_length-1)/2+(n_up_values+1)/2]=1
#
#
#     n_channels = data.shape[-1]
#     X = np.empty((0,n_channels*max_interest_freq))
#     y = []
#
#     # iterate the process over the signal splits
#     for i,j in idx_splits:
#         # Compute the spectrogram of the signal with fft. Desired frequency resolution = 1Hz (-> nfft=fs). The spectrogram
#         # at instant n is computed over the window [n+w_lower,n+w_upper). The signal has to be 0-padded at the end to
#         # compute the spectrogram with mode=psd, i.e. matlab's 'valid'.
#         # tmp = np.transpose(np.concatenate((np.zeros((w_length-1,n_channels)),data[i:j+1,:]))) # manual left 0-padding for causal stft
#         f, t, Zxx = signal.spectrogram(np.transpose(np.concatenate((data[i:j+1,:],np.zeros((w_length-1,n_channels))))),
#                                        fs=fs, window='hanning', nperseg=w_length, noverlap=(w_length-stride),
#                                        nfft=fs, # nfft=fs to have freq resolution f=fs/nfft=1. But to be faster it would be better to impose nfft=np.max(256,2**(np.ceil(np.log2(fs))))
#                                        return_onesided=True, axis=-1, mode='psd') # psd = power spectral density obtained by stft with zero padding and valid window
#         # t -= t[0] # correct the timeshift introduced by the manual 0-padding
#
#         # Select only the power values associated to freq in [1-max_interest_freqHz]. Remember: freq resolution=1, then Zxx[:,1:max_interest_freq+1]
#         Zxx = Zxx[:,1:max_interest_freq+1,:]
#         f = f[1:max_interest_freq+1]
#
#         # # plot the spectrogram of the 1st channel, for debugging
#         # fig = plt.figure()
#         # ax= fig.add_subplot(111)
#         # ax.set_title('ColorMap')
#         # plt.imshow(Zxx[0,:,:],origin='lower',aspect='auto',extent=[t[0],t[-1],f[0],f[-1]], interpolation='None')
#         # # plt.xticks(t)
#         # # plt.yticks(f)
#         # # # add colorbar
#         # plt.colorbar(orientation='vertical')
#         # plt.show()
#
#         # Form the instances xi by entailing the relevant psd values of all the channels at time i
#         k_in = X.shape[0]
#         X = np.concatenate((X,np.zeros((Zxx.shape[-1],Zxx.shape[0]*max_interest_freq))),axis=0)
#         for k in range(Zxx.shape[-1]):
#             X[k_in+k,:] = Zxx[:,:,k].flatten()
#
#         # plot the features for the current split
#         # fig = plt.figure()
#         # ax= fig.add_subplot(111)
#         # ax.set_title('ColorMap')
#         # plt.imshow(X[k_in:k_in+k,:],origin='lower',aspect='auto',extent=[t[0],t[-1],f[0],f[-1]*31], interpolation='None') # the yticklabels should be 1,..,12,1,..,12,1,..,12,.. (31 times)
#         # # plt.xticks(t)
#         # # plt.yticks(f)
#         # # # add colorbar
#         # plt.colorbar(orientation='vertical')
#         # plt.show()
#
#
#         # Create label vector on the base of the "raw" labels provided
#         y = np.concatenate((y,label_data[i:j+1:stride]))
#
#
#     return X, y
#
# def make_statistics(y_true, y_pred):
#     # average accuracy
#     print('CLASSIFICATION ACCURACY: ' + str(accuracy_score(y_true, y_pred)*100) + '%')
#     # precision, recall, f1
#     print('CLASSIFICATION REPORT:')
#     print(classification_report(y_true, y_pred))
#     # confusion matrix
#     print('CONFUSION MATRIX')
#     M = confusion_matrix(y_true=y_true, y_pred=y_pred)
#     print(M)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_title('Confusion Matrix')
#     plt.imshow(M/np.max(np.max(M)).astype(np.float64),origin='upper',aspect='auto', interpolation='None', vmin=0, vmax=1)
#     # add colorbar
#     plt.colorbar(orientation='vertical')
#     plt.show()
#     return
# def plot_spectral_set(X, freqaxis=1):
#     # plot the samples in a spectrogram fashion (time on the x, frequency on the y axis)
#     if freqaxis==1:
#         X = np.transpose(X)
#     fig = plt.figure()
#     ax= fig.add_subplot(111)
#     ax.set_title('ColorMap')
#     plt.imshow(X,origin='lower',aspect='auto', interpolation='None')
#     # plt.xticks(t)
#     # plt.yticks(f)
#     # # add colorbar
#     plt.colorbar(orientation='vertical')
#     plt.show()
#     return
# def plot_eeg(eeg, labels, fs, channel):
#     # plot eeg
#     # channel has to stay in [0, n_channels-1]
#     # fs = sampling frequency in Hz
#     fig = plt.figure()
#     fig.suptitle('EEG RECORDING, 1 ELECTRODE')
#     plt.hold(True) #it is true by default!
#     ax1 = plt.subplot(211)
#     ax1.plot(np.arange(0,float(len(eeg[:,channel]))/float(fs),1./fs),eeg[:,channel],'b')
#     ax1.set_title('EEG, one channel')
#     ax1.set_xlabel('time')
#     ax1.set_ylabel('eeg')
#     ax2 = plt.subplot(212, sharex=ax1)
#     ax2.plot(np.arange(0,float(len(labels))/float(fs),1./fs),labels,'b')
#     ax2.set_ylim(-0.2,1.2)
#     ax2.set_title('EXECUTION ERROR MASK')
#     ax2.set_xlabel('time')
#     ax2.set_ylabel('execution error')
#     plt.show()
#     return
# def compute_average_eeg_allchannels(eeg, labels, fs, seconds, plot=True):
#     n_samples = np.floor(seconds*fs).astype(np.int)
#     t_avg_eeg = np.arange(-seconds,seconds,2.*seconds/(2.*n_samples))
#     avg_eeg = np.zeros((np.floor(2.*seconds*fs).astype(np.int),eeg.shape[-1]))
#     idx_tmp = np.concatenate(([1],np.diff(labels,axis=0))).nonzero()[0]
#     if labels[0] == 1:
#         idx_tmp = idx_tmp[::2]
#     else:
#         idx_tmp = idx_tmp[1::2]
#     idx_tmp = idx_tmp[1:-1]
#     for i in range(len(idx_tmp)):
#         avg_eeg[:n_samples,:] = np.add(avg_eeg[:n_samples,:], eeg[idx_tmp[i]-n_samples:idx_tmp[i],:]) # you should control that idx_tmp-fs exists..
#         avg_eeg[n_samples:,:] = np.add(avg_eeg[n_samples:,:], eeg[idx_tmp[i]:idx_tmp[i]+n_samples,:])
#     avg_eeg = np.mean(avg_eeg,axis=1)
#     if plot==True:
#         plt.figure()
#         ax = plt.subplot(111)
#         ax.plot(t_avg_eeg, avg_eeg)
#         ax.set_xlabel('time [s]')
#         plt.show()
#     return avg_eeg
# def compute_average_eeg(eeg, labels, fs, seconds, channel, plot=True):
#     # plot average eeg n seconds before and after an error event
#     # channel has to stay in [0, n_channels-1].
#     # fs = sampling frequency in Hz
#
#     # do the average of the eeg in 1s windows right before and right after an error event. Map them, respectively in
#     # [-1,0) and [0,1]. This allows to display the "trend" of the eeg after an error event.
#     n_samples = np.floor(seconds*fs).astype(np.int)
#     t_avg_eeg = np.arange(-seconds,seconds,2.*seconds/(2.*n_samples))
#     avg_eeg = np.zeros(np.floor(2.*seconds*fs).astype(np.int))
#     idx_tmp = np.concatenate(([1],np.diff(labels,axis=0))).nonzero()[0]
#     if labels[0] == 1:
#         idx_tmp = idx_tmp[::2]
#     else:
#         idx_tmp = idx_tmp[1::2]
#     idx_tmp = idx_tmp[1:-1]
#     for i in range(len(idx_tmp)):
#         avg_eeg[:n_samples] = np.add(avg_eeg[:n_samples], eeg[idx_tmp[i]-n_samples:idx_tmp[i],channel]) # you should control that idx_tmp-fs exists..
#         avg_eeg[n_samples:] = np.add(avg_eeg[n_samples:], eeg[idx_tmp[i]:idx_tmp[i]+n_samples,channel])
#     avg_eeg = np.divide(avg_eeg,float(len(idx_tmp)))
#     if plot==True:
#         plt.figure()
#         ax = plt.subplot(111)
#         ax.plot(t_avg_eeg, avg_eeg)
#         ax.set_xlabel('time [s]')
#         plt.show()
#     return avg_eeg











######################################################################################
######################################## MAIN ########################################
######################################################################################

def main():

    n_sbjs = 10

    errp_detection_modality = 'synchronous' # synchronous, asynchronous

    feature_type = ('spectral') # 'spectral', 'temporal', 'mdwt'
    temporal_subsample_step = 1
    psd_method = 'welch' # 'welch', 'burg'
    max_relevant_freq = 12 # 40, 12 # Hz
    spectral_feat_selection_criterion = 'no' # mutualinfo, PCA, ICA, R2, no (no feature selection)
    bestk_freq = 20 # spectral dimension after spectral feature selection (selects the best bestk spectral features from the ensemble max_relevant_freq*n_channels).

    fs = 512
    stride = 32 # stride between consecutive windows (in # of samples) in asynchronous classification. Stride in seconds = stride/fs seconds

    w_length = fs # duration (in # of samples) of the window used feature extraction. Window's duration = w_length/fs seconds. For 1s window -> w_length=fs
    w_lower_time = 0.0 # seconds. Used to define a sub-window for temporal features.
    w_upper_time = 1.0 # seconds

    classifier_type = 'weightedrls' # linearSVM, SVM, weightedLinearSVM, weightedSVM, cart, sgdminibatch, rls, weightedrls
    optimize_hyperparameters = True
    params = {'kernel': ['rbf'], 'gamma': (2.**np.arange(-12,-5,2)).tolist(), 'C': (2.**np.arange(-15,10,6)).tolist()}
    majority_vote = False

    label_criterion = 'new' # old, new

    n_splits = 10

    w_lower = np.floor(w_lower_time*fs).astype(np.int)
    w_upper = np.floor(w_upper_time*fs).astype(np.int)


    print('********************************************************')
    print('********************************************************')
    print(errp_detection_modality +' detection of ErrPs in '+str(n_sbjs)+' subjects')
    # print('Train splits: '+ str(trainsplit) + ' Test splits: ' +str(testsplit) )
    print('Feature type: ' + str(feature_type))
    print('Window duration = ' + str(w_length/fs)+'s')
    print('Sub-window = [' + str(w_lower_time) + ', ' + str(w_upper_time) + ']s')
    if 'temporal' in feature_type:
        # print('Temporal sub-window = [' + str(w_lower_time) + ', ' + str(w_upper_time) + ']s')
        print('Temporal subsample step = '+ str(temporal_subsample_step))
    if 'spectral' in feature_type:
    #     print('Spectral feature selection criterion: '+ spectral_feat_selection_criterion) # no feature selection in the synchro case
        print('Interesting frequency range = [0,'+str(max_relevant_freq)+']')
        print('Spectral feature selection criterion: '+ spectral_feat_selection_criterion)
        if spectral_feat_selection_criterion != 'no':
            print('Required number of spectral features after feature selection = '+str(bestk_freq))
    if errp_detection_modality == 'asynchronous': # only spectral feats in this case
        # print('Spectral sub-window = [' + str(w_lower_time) + ', ' + str(w_upper_time) + ']s')
        print('Window stride = '+str(stride)+'samples')
        print('Sampling frequency = '+str(fs)+'Hz')
    print('Classifier type: '+ classifier_type)
    print('********************************************************')
    print('********************************************************')



    ### ITERATE FOR ALL SUBJECTS IN THE DB
    avg_clf_acc_list = []
    avg_bsl_acc_list = []
    avg_bal_acc_list = []
    avg_conf_mat_list = []
    avg_sensitivity_list = []
    avg_specificity_list = []
    avg_auc_list = []
    avg_roc_list = []

    for sbj in range(n_sbjs):
        print('SUBJECT '+str(sbj+1))

        ### READ PARAMETERS
        tmp = loadmat('dataset/S'+str(sbj+1).zfill(2)+'.mat', appendmat = True, variable_names = ['eeg_signal','marker','info'])
        eeg = tmp['eeg_signal']
        execution_error = tmp['marker']['execution_error'][0][0].flatten() # boolean indicating execution errors
        outcome_error = tmp['marker']['outcome_error'][0][0].flatten() # boolean indicating outcome errors
        fs = tmp['info']['samplingrate_Hz'][0][0][0][0] # sampling frequency (Hz)


        # check the inputs
        assert n_splits >= 2
        assert w_lower >= 0 and w_lower < w_upper and w_upper <= w_length
        assert max_relevant_freq <= fs/2
        assert bestk_freq <= max_relevant_freq*eeg.shape[0]
        assert not(errp_detection_modality=='asynchronous' and 'temporal' in feature_type and temporal_subsample_step<2), 'Warning: memory consumption will explode if you do not subsample the temporal features'

        ### Remove EOG data
        eeg = eeg[:,:-3]
        # todo: exploit EOG data to remove the artifacts produced by ocular movement on EEG



        ### DATA SEGMENTATION
        # print('** Data segmentation')
        if errp_detection_modality == 'synchronous':
            idx_X, y = segment_eeg(execution_error, outcome_error, w_length)
        else:
            if label_criterion=='old':
                idx_X, y = segment_eeg_asynchronous(execution_error, w_length, stride)
            else:
                idx_X, y = segment_eeg_asynchronous_new(execution_error, fs, w_length, stride)

        ### FEATURE EXTRACTION
        # print('** Feature extraction')
        X = extract_features(eeg, idx_X, feature_type,
                             w_lower=w_lower, w_upper=w_upper, temporal_subsample_step=temporal_subsample_step,
                             fs=fs, psd_method=psd_method, max_relevant_freq=max_relevant_freq)



        ### ITERATE FOR THE n_splits SPLITS:
        y_test = []
        y_pred = []
        y_pred_conf = []
        for i in range(n_splits):

            print('Split '+str(i+1))

            testsplit = [i]
            trainsplit = np.arange(0,n_splits)
            trainsplit = trainsplit[trainsplit!=i]


            ### SPLIT INSTANCES INTO TRAIN AND TEST SET
            # print('** Split instances')
            idx_train, idx_test = split_instances(X.shape[0], trainsplit, testsplit)
            y_test.append(y[idx_test])


            ### Feature selection..
            # Feature selection has to be based only on the instances of the training set
            if ('spectral' in feature_type or 'mdwt' in feature_type) and spectral_feat_selection_criterion is not 'no':
                if 'temporal' not in feature_type:
                    tmp_X = np.empty((X.shape[0],bestk_freq))
                    tmp_X[idx_train,:], tmp_X[idx_test,:] = select_relevant_features(X[idx_train,:], X[idx_test,:], bestk=bestk_freq, method=spectral_feat_selection_criterion, label=y[idx_train])
                    X = tmp_X
                    del tmp_X
                else:
                    n_channels = eeg.shape[-1]
                    n_temporal_feat = n_channels*(1+np.floor((w_upper-w_lower-1)/temporal_subsample_step).astype(np.int))
                    tmp_X = np.empty((X.shape[0],bestk_freq))
                    tmp_X[idx_train,:], tmp_X[idx_test,:] = select_relevant_features(X[idx_train,n_temporal_feat:], X[idx_test,n_temporal_feat:], bestk=bestk_freq, method=spectral_feat_selection_criterion, label=y[idx_train])
                    X = np.concatenate(( X[:,:n_temporal_feat], tmp_X ), axis=1)
                    del tmp_X


            ### TRAIN THE CLASSIFIER
            # print('** Train the classifier')
            clf = trainmodel(X[idx_train,:], y[idx_train], classifier_type, hyp_opt=optimize_hyperparameters, param_grid=params, validation_subsampling_rate=4)


            ### TEST
            # print('** Testing the classifier')
            if majority_vote == False:
                y_pred.append(clf.predict(X[idx_test,:]))
                if 'SVM' in classifier_type:
                    y_pred_conf.append(clf.predict_proba(X[idx_test,:])[:,1])
                else:
                    y_pred_conf.append(clf.decision_function(X[idx_test,:]))
            else: # if smoothing of the prediction is required
                smoothing_weights = np.array([0.1,0.3,0.6]) # define smoothing weights
                smoothing_weights = np.flip(smoothing_weights, axis=0)
                if 'SVM' in classifier_type:
                    tmp_y_pred_conf = clf.predict_proba(X[idx_test,:])[:,1] # obtain conf_score
                    tmp_y_pred_conf = np.convolve(np.pad(tmp_y_pred_conf, (len(smoothing_weights)-1,0), 'edge'), smoothing_weights, 'valid') # smooth the conf_score
                    y_pred_conf.append(tmp_y_pred_conf)
                    tmp_y_pred = tmp_y_pred_conf # obtain y_pred from smoothed pred conf
                    tmp_y_pred[tmp_y_pred>=0.5] = 1
                    tmp_y_pred[tmp_y_pred<0.5] = 0
                    y_pred.append(tmp_y_pred)
                else:
                    tmp_y_pred_conf = clf.decision_function(X[idx_test,:]) # obtain conf_score
                    tmp_y_pred_conf = np.convolve(np.pad(tmp_y_pred_conf, (len(smoothing_weights)-1,0), 'edge'), smoothing_weights, 'valid') # smooth the conf_score
                    y_pred_conf.append(tmp_y_pred_conf)
                    tmp_y_pred = tmp_y_pred_conf # obtain y_pred from smoothed pred conf
                    tmp_y_pred[tmp_y_pred>=0] = 1
                    tmp_y_pred[tmp_y_pred<0] = 0
                    y_pred.append(tmp_y_pred)




        ### MAKE STATISTICS ON THE RESULTS
        # print('** Statistics')
        avg_bsl_acc, avg_clf_acc, avg_conf_mat, avg_sensitivity, avg_specificity, avg_bal_acc, avg_auc, avg_roc = make_avg_statistics(y_test, y_pred, y_pred_conf, n_splits, print_results=False)
        avg_bsl_acc_list.append(avg_bsl_acc)
        avg_clf_acc_list.append(avg_clf_acc)
        avg_bal_acc_list.append(avg_bal_acc)
        avg_conf_mat_list.append(avg_conf_mat)
        avg_sensitivity_list.append(avg_sensitivity)
        avg_specificity_list.append(avg_specificity)
        avg_auc_list.append(avg_auc)
        avg_roc_list.append(avg_roc)


    print('** Average ('+str(n_splits)+'-fold) baseline classification accuracy subj-by-subj:')
    print(avg_bsl_acc_list)
    print('** Average ('+str(n_splits)+'-fold) classification accuracy subj-by-subj:')
    print(avg_clf_acc_list)
    print('** Average ('+str(n_splits)+'-fold) balanced classification accuracy subj-by-subj:')
    print(avg_bal_acc_list)
    print('** Average ('+str(n_splits)+'-fold) AUC subj-by-subj:')
    print(avg_auc_list)
    # print('** Average ('+str(n_splits)+'-fold) confusion matrix subj-by-subj:')
    # print(avg_conf_mat_list)

    print('')
    print('** Average ('+str(n_splits)+'-fold) baseline classification accuracy, averaged over sbjs:')
    print(np.mean(avg_bsl_acc_list))
    print('** Average ('+str(n_splits)+'-fold) classification accuracy, averaged over sbjs:')
    print(np.mean(avg_clf_acc_list))
    print('** Average ('+str(n_splits)+'-fold) balanced classification accuracy, averaged over sbjs:')
    print(np.mean(avg_bal_acc_list))

    print('')
    avg_conf_mat_allsbj = np.mean( np.concatenate(( np.empty((0,2,2)), avg_conf_mat_list ), axis=0 ), axis=0 )
    print('Number of test instances: '+ str(np.floor(np.sum(np.sum(avg_conf_mat_allsbj)))))
    print('** Average ('+str(n_splits)+'-fold) confusion matrix, averaged over sbjs and normalized to 1:')
    avg_conf_mat_allsbj /= np.sum(np.sum(avg_conf_mat_allsbj))
    print( avg_conf_mat_allsbj )
    print('** Average ('+str(n_splits)+'-fold) sensitivity (detectedErr/totErr), averaged over sbjs:')
    print(np.mean(avg_sensitivity_list))
    print('** Average ('+str(n_splits)+'-fold) specificity (undetectedErr/totNoErr), averaged over sbjs:')
    print(np.mean(avg_specificity_list))
    print('** Average ('+str(n_splits)+'-fold) AUC, averaged over sbjs:')
    print(np.mean(avg_auc_list))

    for roc in avg_roc_list:
        plt.plot(roc[:,0], roc[:,1])
    plt.show()


if __name__ == '__main__':
    main()
