import numpy as np
from sklearn.metrics import average_precision_score


def metrics_ranking(score_k, score_u, topk_list=[10, 100, 1000], topp_list=[1, 5, 10]):
    """Compute the metrics:
    Top-10-acc: acc of top 10 for the novel class samples, 
    Top-100-acc: acc of top 100 for the novel class samples, 
    Top-1000-acc: acc of top 1000 for the novel class samples, 
    RANK-last: the max rank at which all novel samples are found.
    """
    results = {}

    scores = np.concatenate([score_k, score_u])
    novel = np.concatenate([np.zeros_like(score_k), np.ones_like(score_u)])

    # sort based on scores ascending (low scores correspond to novel samples)
    sorted_indices = np.argsort(scores)
    novel_sorted = novel[sorted_indices]

    # top-k accuracy for novel samples (num of novel samples in top k sorted results)
    for k in topk_list:
        topk_acc = 100 * np.sum(novel_sorted[:k]) / float(k)
        print(f"Topk-{k}-acc: {topk_acc:.1f}")
        results[f"Topk-{k}-acc"] = topk_acc
    
    # top-p accuracy for novel samples (num of novel samples in top p percent sorted results)
    for p in topp_list:
        top_p = int(p/100 * len(scores))
        topp_acc = 100 * float(np.sum(novel_sorted[:top_p]) / float(top_p))
        print(f"Topp-{p}-acc: {topp_acc:.1f} within top {top_p} samples")
        results[f"Topp-{p}-acc"] = topp_acc

    # get the index of the last novel sample
    last_novel_index = np.argwhere(novel_sorted==1)[-1][0]
    # get rank of first occurence per category
    rank = np.linspace(0, 1.0, len(scores))
    rank_last = float(rank[last_novel_index])
    print(f"Rank-last: {rank_last:.4f}")
    results["RANK-last"] = rank_last
    return results



def get_curve_online(known, novel, stypes = ['Bas']):
    """positive: known, negative: novel"""
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(score_k, score_u, labels, probs_k, stypes=['Bas'], verbose=True):
    """positive: known, negative: novel"""

    tp, fp, tnr_at_tpr95 = get_curve_online(score_k, score_u, stypes)
    results = dict()
    mtypes = ['R@P95', 'TNR@TPR95', 'AUROC', 'AUROC2', 'DTACC', 'AUIN', 'AUOUT', 'AUPR', 'OSCR']
    if verbose:
        print('{:35s} '.format(''), end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:35} '.format(stype=stype), end='')
        results[stype] = dict()
        
        #Recall@Precision95
        mtype = 'R@P95'
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.concatenate([[0.5], tp[stype] / (tp[stype] + fp[stype]), [1.]])
            precision[np.isnan(precision)] = 1.
        recall = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]]) # tpr
        # find the index where precision is >= 0.95
        p95_pos = np.argwhere(precision>=0.95)[0][0]
        r_at_p95 = recall[p95_pos]
        results[stype][mtype] = 100.*r_at_p95
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')

        # TNR
        mtype = 'TNR@TPR95'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(-np.trapz(1.-fpr, tpr))
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')

        # AUROC2 (to check implementation is correct. Should be the same as AUROC)
        mtype = 'AUROC2'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(np.trapz(y=tpr[::-1], x=fpr[::-1]))
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')

        # AUPR (as percentage)
        mtype = 'AUPR'
        results[stype][mtype] = average_precision_score([0] * len(score_k) + [1] * len(score_u),
                                                        list(-score_k) + list(-score_u)) * 100.
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')

        # OSCR
        mtype = 'OSCR'
        results[stype][mtype] = get_oscr(score_k, score_u, labels, probs_k)
        if verbose:
            print(' {val:6.1f}'.format(val=results[stype][mtype]), end='')
            print('')
        
    return results


def get_oscr(score_k, score_u, labels, probs_k):
    pred = np.argmax(probs_k, axis=1)
    m_score_k = np.zeros(len(score_k))
    m_score_k[pred == labels] = 1
    k_target = np.concatenate((m_score_k, np.zeros(len(score_u))), axis=0)
    u_target = np.concatenate((np.zeros(len(score_k)), np.ones(len(score_u))), axis=0)
    predict = np.concatenate((score_k, score_u), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(score_k))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(score_u))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    # return as percentage
    return OSCR * 100.

