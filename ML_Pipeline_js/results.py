# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 2024

@author: jspiller
"""


## LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from ml_spectroscopy.config import path_init
from ml_spectroscopy.plottings_utils_results import ROC_PR_js
from ml_spectroscopy.utility_functions import FDRthreshold_js
from sklearn.metrics import confusion_matrix
import os

## SET SEED FOR REPRODUCIBILITY
random.seed(100)


## Settings
data_name = 'GQlupb'
planet = 'GQlupB'
bal=50
frame='simple'


plt.style.use('seaborn')

## ACTIVE SUBDIR
subdir = path_init()

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/"
plot_path = subdir + "60_plots/"
results_path = subdir + "70_results/"
visual_path = subdir + "80_visualisation/"




# =============================================================================
# # Change this part each time
# =============================================================================

alpha=8
version="6" #6

folds = [0,1,2]

plotname = 'test'

methods_ls = ['CNN1 v6', 'CNN2 v6', 'CNN1t_3Conv v6', 'CNN1 v06', 'CNN2 v06', 'CNN1t_3Conv v06']

# Directory to fetch results
dir_path = results_path + "finalresults/"

SAVE = True  #Should the plots be saved?  (default=False)
baseline_included = True  #Are the baseline models (PCT, CNN1) part of methods_ls? (default=True)

#Directory to store results
visualisation_path = visual_path + "finalresults/"

#where to get baseline results from
baseline_path = visual_path + '0103_test/BaselineForPlotting/'

# =============================================================================
# =============================================================================

colors_v6 = ['#0096ff', '#2ca02c', '#ff7f0e']  # Blue (CNN1), Green (CNN2), Red (Att/ConvA/PCT)
colors_v06 = ['#2727ff', '#026c45', '#d62728']  # Darker Blue (CNN1), Darker Green (CNN2), Darker Red (Att/ConvA/PCT)

if version=="6":
    colors={
        'CNN1 v6': colors_v6[0],
        'CNN1 v06': colors_v06[0],
        'CNN2 v6': colors_v6[1],
        'CNN2 v06': colors_v06[1],
        'CNN1t_3Conv': colors_v6[2],
        'CNN1t_3Conv v6': colors_v6[2],
        'CNN1t_3Conv v06': colors_v06[2],
        'CNN1': colors_v6[0],
        'CNN2': colors_v6[1],
        'PCT': colors_v6[2],
        'CNN_js_AttE v6': colors_v6[2],
        'CNN_js_AttE v06': colors_v06[2]
    }
else:
    colors={
        'CNN1 v6': colors_v6[0],
        'CNN1 v06': colors_v06[0],
        'CNN2 v6': colors_v6[1],
        'CNN2 v06': colors_v06[1],
        'CNN1t_3Conv': colors_v06[2],
        'CNN1t_3Conv v6': colors_v6[2],
        'CNN1t_3Conv v06': colors_v06[2],
        'CNN1': colors_v06[0],
        'CNN2': colors_v06[1],
        'PCT': colors_v06[2],
        'CNN_js_AttE v6': colors_v6[2],
        'CNN_js_AttE v06': colors_v06[2]
    }

if SAVE: os.makedirs(visualisation_path, exist_ok=True)

if not baseline_included: # Get Baseline models (CNN1, PCT) from file
    with np.load(baseline_path + 'baselineplotting_rocpr_GQlupb_data_0_alpha_' + str(alpha) + '_nfolds_' + str(len(folds))+ '_CNN1.npz') as data:
        CNN1_fpr, CNN1_tpr, CNN1_recall, CNN1_precision = data['array1'], data['array2'], data['array3'], data['array4']
    with np.load(baseline_path + 'baselineplotting_rocpr_GQlupb_data_0_alpha_' + str(alpha) + '_nfolds_' + str(len(folds))+ '_PCT.npz') as data:
        PCT_fpr, PCT_tpr, PCT_recall, PCT_precision = data['array1'], data['array2'], data['array3'], data['array4']
    with open(baseline_path + 'baselineplotting_class_GQlupb_data_0_alpha_' + str(alpha) + '_nfolds_' + str(len(folds))+ '_CNN1.pkl', "rb") as f:
        CNN1_fp, CNN1_tp, CNN1_roc_auc, CNN1_pr_auc = pickle.load(f)
    with open(baseline_path + 'baselineplotting_class_GQlupb_data_0_alpha_' + str(alpha) + '_nfolds_' + str(len(folds))+ '_PCT.pkl', "rb") as f:
        PCT_fp, PCT_tp, PCT_roc_auc, PCT_pr_auc = pickle.load(f)


# What was the base template used for the experiments?
#template_characteristics = {'Temp': 1200, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
# And the base dataset corresponding to the base template? #Also possible to take ls_results[0]['y_test']
# data1=pd.read_pickle(data_path+'data_4ml/v'+str(v)+'_ccf_4ml_trim_robustness_simple/H2O_'+data_name+'_scale'+str(alpha)+'_bal'+str(bal)+'_temp1200.0_sg4.1_ccf_4ml_trim_norepetition_v'+str(v)+'_simple.pkl')

ls_data = {j: 'results_GQlupb_data_0_alpha_'+str(alpha)+'_CV_testfold'+str(j)+ '_v'+str(version) + '_js.pkl' for j in folds}
ls_results = {key: None for key in range(9)}

for j in folds:
    with open(dir_path + str(ls_data[j]), "rb") as f:
        ls_results[j] = pickle.load(f)

# Hyperparameters: ls_results[0]['hyperparameters']
# Probabilities ls_results[1]['results']['CNN1']['Y_pred_prob']

#Concatenate all data
y_test=[]
for j in folds:
    #Y_test = data1.loc[(str(data1.index.levels[0][j]), slice(None)), 'H2O']
    Y_test = np.array(ls_results[j]['y_test'])[:, 1]
    y_test += list(Y_test)
y_test=np.array(y_test)


# # =============================================================================
# # # =============================================================================
# # # ROC & PR curves
# # # =============================================================================
# # =============================================================================
# for j in folds:
#     #Y_test=data1.loc[(str(data1.index.levels[0][j]), slice(None)), 'H2O']
#     Y_test = np.array(ls_results[j]['y_test'])[:, 1]
#     ls_roc_auc, ls_fpr, ls_tpr, ls_pr_auc, ls_precision, ls_recall, pr_no_skill = ROC_PR_js([ls_results[j]['results']], np.array(Y_test), methods_ls)
#
#     #ROC
#     plt.figure()
#     for met in methods_ls:
#         #plt.plot(ls_fpr[met], ls_tpr[met], lw=1, color=color_ls[met], label=met+" (AUC={})".format(np.round(ls_roc_auc[met],3)))
#         plt.plot(ls_fpr[met], ls_tpr[met], lw=1, label=met+" (AUC={})".format(np.round(ls_roc_auc[met],3)))
#     plt.plot(ls_fpr['noskill'], ls_tpr['noskill'], linestyle='--', lw=1, color='gray')  #'No Skill'
#     plt.ylabel('True positive rate')
#     plt.xlabel('False positive rate')
#     plt.title('ROC-Curve (CV test fold: {})'.format(j))
#     plt.legend()
#     if SAVE: plt.savefig(visualisation_path + 'ROC_' + planet + '_alpha' + str(alpha) + '_bal' + str(bal) + '_combined_CV_' + str(plotname) + '_version' + str(version) + 'frame' + str(frame) + '_fold' + str(j) + '_js.pdf')
#     plt.show()
#
#
#     #PR
#     plt.figure()
#     for met in methods_ls:
#         #plt.plot(ls_recall[met], ls_precision[met], lw=1, color=color_ls[met], label=met + " (AUC={})".format(np.round(ls_pr_auc[met],3)))
#         plt.plot(ls_recall[met], ls_precision[met], lw=1, label=met + " (AUC={})".format(np.round(ls_pr_auc[met],3)))
#     plt.plot([0, 1], [pr_no_skill, pr_no_skill], linestyle='--', lw=1, color='gray')
#     plt.ylabel('Precision')
#     plt.xlabel('Recall')
#     plt.title('PR-Curve (CV test fold: {})'.format(j))
#     plt.legend()
#     if SAVE: plt.savefig(visualisation_path + 'PR_' + planet + '_alpha' + str(alpha) + '_bal' + str(bal) + '_combined_CV_' + str(plotname) + '_version' + str(version) + 'frame' + str(frame) + '_fold' + str(j) + '_js.pdf')
#     plt.show()

# =============================================================================
# # =============================================================================
# # Aggregated ROC & PR curves
# # =============================================================================
# =============================================================================

ls_roc_auc, ls_fpr, ls_tpr, ls_pr_auc, ls_precision, ls_recall, pr_no_skill = ROC_PR_js([ls_results[j]['results'] for j in folds], np.array(y_test), methods_ls)
#Aggregated ROC
plt.style.use('seaborn')
plt.figure()
for met in methods_ls:
    #plt.plot(ls_fpr[met], ls_tpr[met], lw=1, color=color_ls[met], label=met+" (AUC={})".format(np.round(ls_roc_auc[met],3)))
    plt.plot(ls_fpr[met], ls_tpr[met], lw=1, color=colors[met], label=met+" (AUC={})".format(np.round(ls_roc_auc[met],3)))
if not baseline_included:
    plt.plot(CNN1_fpr, CNN1_tpr, lw=1, label="CNN1_bl (AUC={})".format(np.round(CNN1_roc_auc,3)))
    plt.plot(PCT_fpr, PCT_tpr, lw=1, label="PCT_bl (AUC={})".format(np.round(PCT_roc_auc,3)))
plt.plot(ls_fpr['noskill'], ls_tpr['noskill'], linestyle='--', lw=1, color='gray')  #'No Skill'
plt.title('Aggregated ROC Curve (folds 0,1,2)', fontsize=22)
plt.ylabel('True positive rate', fontsize=20)
plt.xlabel('False positive rate', fontsize=20)
plt.legend(fontsize=12, loc='lower right')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
if SAVE: plt.savefig(visualisation_path + 'Aggregated_ROC_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(version) + 'frame' + str(frame) + '_js.pdf')
plt.show()

# #Aggregated PR
# plt.figure()
# for met in methods_ls:
#     #plt.plot(ls_recall[met], ls_precision[met], lw=1, color=color_ls[met], label=met + " (AUC={})".format(np.round(ls_pr_auc[met],3)))
#     plt.plot(ls_recall[met], ls_precision[met], lw=1, label=met + " (AUC={})".format(np.round(ls_pr_auc[met],3)))
# if not baseline_included:
#     plt.plot(CNN1_recall, CNN1_precision, lw=1, label="CNN1_bl (AUC={})".format(np.round(CNN1_pr_auc,3)))
#     plt.plot(PCT_recall, PCT_precision, lw=1, label="PCT_bl (AUC={})".format(np.round(PCT_pr_auc,3)))
# plt.plot([0, 1], [pr_no_skill, pr_no_skill], linestyle='--', lw=1, color='gray')
# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.title('Aggregated PR-Curve')
# plt.legend()
# if SAVE: plt.savefig(visualisation_path + 'Aggregated_PR_alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(version) + 'frame' + str(frame) + '_js.pdf')
# plt.show()





# =============================================================================
# # =============================================================================
# # Histogram
# # =============================================================================
# =============================================================================


Y_prob = {key: None for key in methods_ls}
prob_treshold = {key: None for key in methods_ls} # False Discovery Rate < 5%
CM = {key: None for key in methods_ls} #Confusion Matrix
for met in methods_ls:
    Y_prob[met] = []
    for j in folds:
        Y_prob[met] += list(ls_results[j]['results'][met]['Y_pred_prob'][:, 1])
    Y_prob[met]=np.array(Y_prob[met])
    prob_treshold[met] = FDRthreshold_js(y_test, Y_prob[met])
    CM[met] = confusion_matrix(y_test, Y_prob[met]>=prob_treshold[met])  # [['tn', 'fp'], ['fn', 'tp']]

# # Plot Histograms
# for met in methods_ls:
#     plt.figure()
#     x1 = np.array(Y_prob[met][y_test == 1])
#     x1_med = np.median(x1)
#     weightsx1 = np.ones_like(x1) / len(x1)
#     x2 = np.array(Y_prob[met][y_test == 0])
#     x2_med = np.median(x2)
#     weightsx2 = np.ones_like(x2) / len(x2)
#     kwargs = dict(histtype='stepfilled', bins=60, density=False)
#     plt.hist(x1, **kwargs, color='deepskyblue', alpha=0.5, edgecolor='xkcd:azure', weights=weightsx1)
#     plt.axvline(x1_med, color="steelblue", alpha=0.7, lw=2, label='H2O group,\n median = ' + str(np.round(x1_med, 2)))
#     plt.hist(x2, **kwargs, color='crimson', alpha=0.5, edgecolor='red', weights=weightsx2)
#     plt.axvline(x2_med, color="indianred", alpha=0.7, lw=2, label=' No H2O group,\n median = ' + str(np.round(x2_med, 2)))
#     plt.axvline(prob_treshold[met], color='rebeccapurple', alpha=0.7, label='FDR=0.05: T*=' + str(np.round(prob_treshold[met],3)), lw=2) # Threshold for 5% False Discovery Rate (FDR)
#     plt.ylabel('Relative Frequency', fontsize=20, color='black')
#     plt.xlabel('Probability scores', fontsize=20, color='black')
#     plt.xlim([0, 1])
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.title('Empirical distribution of ' + str(met) + ' scores', fontsize=22, color="black")
#     plt.legend(fontsize=12)
#     if SAVE: plt.savefig(visualisation_path + 'Prob_signals_method' + str(met) + 'alpha' + str(alpha) + '_bal' + str(bal) + '_version' + str(version) + 'frame' + str(frame) + '_js.pdf')
#     plt.show()



# =============================================================================
# # =============================================================================
# # Classification
# # =============================================================================
# =============================================================================

if baseline_included:
    X=np.arange(len(methods_ls))
    fp=[CM[met][0,1] for met in methods_ls]
    tp=[CM[met][1,1] for met in methods_ls]
    methods_ls1 = methods_ls
else:
    X = np.arange(len(methods_ls)+2)
    fp = [CM[met][0, 1] for met in methods_ls] + [CNN1_fp, PCT_fp]
    tp = [CM[met][1, 1] for met in methods_ls] + [CNN1_tp, PCT_tp]
    methods_ls1 = methods_ls+['CNN1_bl','PCT_bl']

metlen=max([len(met) for met in methods_ls])
size=min(14, np.round(14*6*6/len(methods_ls)/metlen))

# plt.figure(figsize=(16.0, 5.5))
# plt.bar(X - 0.2, fp, 0.4, label='False Positives', color="crimson", alpha=0.5, edgecolor='red')
# plt.bar(X + 0.2, tp, 0.4, label='True Positives', color="deepskyblue", alpha=0.5, edgecolor='xkcd:azure')
# for i,x in enumerate(X):
#     plt.text(x - 0.2, fp[i] + 15, str(fp[i]), ha='center', fontsize=12)
#     plt.text(x + 0.2, tp[i] + 15, str(tp[i]), ha='center', fontsize=12)
# plt.tick_params(labelsize=16)
# plt.xticks(X, methods_ls1, size=16)
# plt.xlabel("Method", fontsize=20, color="black")
# plt.ylabel("Planet Detections", fontsize=20, color="black")
# plt.ylim(0, max(*fp,*tp)*1.25)
# plt.xlim(-0.6, 5.6)
# plt.axvline(x=2.5, color='grey', linestyle=':', linewidth=2)
# plt.title("Classification results for a 5% FDR threshold (v06)", fontsize=22, color="black")
# plt.legend(framealpha=0.1, facecolor='gray', prop=dict(size=12), loc='upper left', fontsize=12)
# plt.tight_layout()
# if SAVE: plt.savefig(visualisation_path + 'FDR_results_alpha' + str(alpha) + '_version' + str(version) + '_js.pdf', bbox_inches='tight')
# plt.show()



# # Saving Baseline model Results
# baseline_path = visualisation_path + 'BaselineForPlotting/'
# np.savez(baseline_path+'baselineplotting_rocpr_GQlupb_data_0_alpha_'+str(alpha)+'_nfolds_'+str(len(folds))+'_CNN1.npz', array1=ls_fpr['CNN1'], array2=ls_tpr['CNN1'], array3=ls_recall['CNN1'], array4=ls_precision['CNN1'])
# np.savez(baseline_path+'baselineplotting_rocpr_GQlupb_data_0_alpha_'+str(alpha)+'_nfolds_'+str(len(folds))+'_PCT.npz', array1=ls_fpr['PCT'], array2=ls_tpr['PCT'], array3=ls_recall['PCT'], array4=ls_precision['PCT'])
# with open(baseline_path+'baselineplotting_class_GQlupb_data_0_alpha_'+str(alpha)+'_nfolds_'+str(len(folds))+'_CNN1.pkl', "wb") as f:
#     pickle.dump((CM['CNN1'][0,1], CM['CNN1'][1,1], ls_roc_auc['CNN1'], ls_pr_auc['CNN1']), f)
# with open(baseline_path+'baselineplotting_class_GQlupb_data_0_alpha_'+str(alpha)+'_nfolds_'+str(len(folds))+'_PCT.pkl', "wb") as f:
#     pickle.dump((CM['PCT'][0,1], CM['PCT'][1,1], ls_roc_auc['PCT'], ls_pr_auc['PCT']), f)
