import numpy as np
import time
from sklearn.metrics import confusion_matrix
from HPtuning_GA import genetic_algorithm, decode
import fitsio
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, f1_score


#Find FDR threshold
def FDRthreshold_js(Y_test, prob, fdr=0.05, iterations=1000):
    def FDR(true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels)
        fp = cm[0, 1]  # False Positives
        tp = cm[1, 1]  # True Positives
        fdr = fp / max(1, (fp + tp))  # False Discovery Rate
        return fdr

    for treshold in np.linspace(1, 0, iterations+1):
        if FDR(Y_test, prob>=treshold)>=fdr: return treshold+1/iterations
    return 0


# Optimize hyperparameters and Train the model
def train_CNN_js(CNN_model, bounds_CNN, x_train, y_train, x_valid, y_valid, x_test, y_test, strategy):
    # set up optimizer:
    # define the total iterations
    n_iter = 10  # 20
    # bits per variable
    n_bits = 16
    # define the population size (always even!!)
    n_pop = 10 
    # crossover rate
    r_cross = 0.9


    ## optim:
    start_o_CNN = time.time()

    # mutation rate
    r_mut_CNN = 1.0 / (float(n_bits) * len(bounds_CNN))
    # perform the genetic algorithm search
    best_CNN, score_CNN, best_accuracies_valid_CNN, best_accuracies_train_CNN, track_generation_CNN, track_hyperparams_CNN = genetic_algorithm(
        CNN_model, bounds_CNN, n_bits, n_iter, n_pop, r_cross, r_mut_CNN, x_train, y_train, x_valid, y_valid, strategy)
    decoded_CNN = decode(bounds_CNN, n_bits, best_CNN)
    # end time
    end_o_CNN = time.time()

    # test model:
    print("Test model")
    start_m_CNN = time.time()
    res_opt_method = CNN_model(decoded_CNN, x_train, y_train, x_test, y_test, strategy)
    end_m_CNN_js1 = time.time()

    CM_opt = np.array((confusion_matrix(y_test, res_opt_method['Y_pred']).ravel()))
    hyperparam_optim = decoded_CNN

    # optimization results
    optim_results = {'best_accuracy_valid': best_accuracies_valid_CNN,
                             'best_accuracy_train': best_accuracies_train_CNN, 'generation': track_generation_CNN,
                             'hyperparams': track_hyperparams_CNN, 'runtime_GA': (end_o_CNN - start_o_CNN),
                             'runtime_model': (end_m_CNN_js1 - start_m_CNN)}

    return res_opt_method, CM_opt, hyperparam_optim, optim_results



# Discretize kernelsize, batch_size and nfilters
def ks(kernelsize):
    if kernelsize < 4: return 3
    elif kernelsize < 6: return 5
    else: return 7

def bs(batch_size):
    if batch_size < 6: return 4
    elif batch_size < 12: return 8
    elif batch_size < 24: return 16
    elif batch_size < 48: return 32
    elif batch_size < 96: return 64
    elif batch_size < 150: return 128
    else: return 256

def nf(nfilters):
    if nfilters < 6: return 4
    elif nfilters < 12: return 8
    if nfilters < 24: return 16
    elif nfilters < 48: return 32
    elif nfilters < 96: return 64
    elif nfilters < 150: return 128
    else: return 256



def extract_center(data, percentage):
    start = int((0.5-percentage/200) * data.shape[1])
    end = int((0.5+percentage/200) * data.shape[1])
    return data[:, start:end, :]


def extract_data_from_fits(file_path):
    print(f"Processing file: {file_path}")
    with fitsio.FITS(file_path) as hdul:
        target_ids = hdul[1]['TARGETID'][:]
        redshifts = hdul[1]['ELG_Z'][:]
        flux = hdul[1]['FLUX'][:]
        labels = hdul[1]['LABEL'][:]
    return flux, labels, redshifts



def ROC_PR(Y_pred_prob, Y_test):

    #No Skill (Diagonal)
    ns_probs0 = np.zeros_like(Y_test)

    ls_fpr_noskill, ls_tpr_noskill, _ = roc_curve(Y_test, ns_probs0)
    ls_roc_auc_noskill = roc_auc_score(Y_test, ns_probs0)  #Area under Curve (auc) to quantify curve quality
    pr_no_skill = len(Y_test[Y_test == 1.0]) / len(Y_test)

    # ROC
    ls_fpr, ls_tpr, _ = roc_curve(Y_test, Y_pred_prob)
    ls_roc_auc = roc_auc_score(Y_test, Y_pred_prob)
    # PR
    ls_precision, ls_recall, _ = precision_recall_curve(Y_test, Y_pred_prob)
    ls_pr_auc = auc(ls_recall, ls_precision)

    return ls_roc_auc, ls_fpr, ls_tpr, ls_pr_auc, ls_precision, ls_recall, pr_no_skill, ls_roc_auc_noskill, ls_fpr_noskill, ls_tpr_noskill