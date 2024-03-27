import numpy as np
from scipy.io import loadmat
from pathlib import Path
import tester
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import data
from tqdm import tqdm
from evaluation2020 import evaluate_12ECG_score as evalecg
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support, average_precision_score
import matplotlib.pyplot as plt

def F1(TP, TN, FP, FN, beta=2, return_total=True):
    '''
    beta adds more weight to recall than precision.
    '''
    class_weight = np.ones(9)
    numerator = (1+beta**2) * TP
    denominator = (1+beta**2)*TP + FP + beta**2*FN
    res = numerator/denominator
    if return_total:
        res = (class_weight * res).mean()
    return res

def jaccard(TP, TN, FP, FN, beta=2, return_total=True):
    '''
    beta gives missed diagnoses twice as much weight as correct diagnoses and false alarms.
    '''
    class_weight = np.ones(9)
    res = TP/(TP+FP+beta*FN)
    if return_total:
        res = (class_weight * res).mean()
        
    return res

def laadfunctie():
# Hier laden we de ECG data dmv loadmat, header_data bevat de diagnose, leeftijd, geslacht etc.
    src = Path(r'D:\data\ECG\12lead\PhysioNetChallenge2020_Training_CPSC\Training_WFDB')
    for fname in src.glob('*.mat'):
        data = loadmat(fname)
        for filename in src.glob('*.hea'):
            with open(filename, 'r') as the_file:
                all_data = [line.strip() for line in the_file.readlines()]
                height_line = all_data[3]
                header_data = all_data[8:]
            if fname.stem == filename.stem:
                # Het is belangrijk om beide namen gelijk aan elkaar te hebben, anders zou je verkeerde data aan elkaar kunnen koppelen.
                yield header_data, data
                # Return sends a specified value back to its caller whereas Yield can produce a sequence of values. 
                # We should use yield when we want to iterate over a sequence, but don’t want to store the entire sequence in memory.

def load():
    src = Path(r'D:\data\ECG\12lead\PhysioNetChallenge2020_Training_CPSC\Training_WFDB')
    fnames = list(src.glob('*.hea'))
    np.random.RandomState(808).shuffle(fnames) # Een random shuffle, door de 808 is deze shuffle op elke computer gelijk
    sep = int(len(fnames) * .9) # 90% als training set
    fnames = fnames[sep:]  # [:10], laad de laatste 10% als validatie set

    for fname in fnames:
        ecg = data.ECGPN2020(str(fname)[:-4])
        yield ecg.signal, ecg.label_mask

def main():
    model_fname = '../../PhysioNet_Challenge/model.pt'
    state_dict = torch.load(model_fname)['model']
    tstr = tester.Tester(state_dict)
    predictions = list()
    all_labels = list()

    for data, label_mask in tqdm(load()):
        # print(data.shape)
        probabilities = tstr.predict(data)
        #classes = probabilities >= 0.5
        predictions.append(probabilities)
        all_labels.append(label_mask)
    
    # Evaluate performance -- ROC curves
    fpr = dict()
    tpr = dict()
    thr = dict()
    roc_auc = dict()
    all_labels = np.array(all_labels)
    predictions = np.array(predictions)
    for i in range(all_labels.shape[1]):
        fpr[i], tpr[i], thr[i] = roc_curve(all_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure(0)
        plt.plot(fpr[i], tpr[i], label=f'{i}: auc: {roc_auc[i]:0.3f}')
    plt.legend()

    # Evaluate performance -- precision and recall
    precision = dict()
    recall = dict()
    thresholds = dict()
    roc_auc = dict()
    average_precision = dict()
    for i in range(all_labels.shape[1]):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(all_labels[:, i], predictions[:, i])
        average_precision[i] = average_precision_score(all_labels[:, i], predictions[:, i])
        plt.figure(1)
        plt.plot(recall[i], precision[i], label='lab {}'.format(i))
    plt.legend()


    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(all_labels.ravel(),
        predictions.ravel())
    average_precision["micro"] = average_precision_score(all_labels, predictions,
                                                         average="micro")
    
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))


    # Scores
    threshold = 0.5
    beta = 2
    num_classes = 9
    
    auroc, auprc = evalecg.compute_auc(all_labels, predictions, num_classes)
    accuracy, f_measure, f_beta, g_beta = evalecg.compute_beta_score(all_labels, predictions>threshold, beta, num_classes)

    print(f'AUROC: {auroc:.3f}')
    print(f'AURPC: {auprc:.3f}')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1: {f_measure:.3f}')
    print(f'F1 beta: {f_beta:.3f}')
    print(f'Jaccard beta: {g_beta:.3f}')


if __name__ == '__main__':
    main()