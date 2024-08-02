import numpy as np
import calibration as cal
import matplotlib
import matplotlib.pyplot as plt


def plot_settings():
    usetex = False
    fontsize = 18  # paper: fontsize=18, poster fontsize=28
    #plt.rc('font', **{'sans-serif': ['Arial']})
    plt.rc('text', usetex=usetex)
    matplotlib.rcParams.update({'font.size': fontsize, "axes.axisbelow": True})


def get_calibration_plot_arrays(labels, probs, num_bins=10):
    """ Get binned confidence and accuracy for multi-class problems for calibration plots"""
    confidence_bins, accuracy_bins, count_bins = [], [], []
    max_probs = np.max(probs, axis=1)
    preds_class = np.argmax(probs, axis=1)
    # bin data based on top_probs
    bins = np.linspace(0, 1, num_bins)
    bin_ids = np.digitize(max_probs, bins)
    # start bin id at 0
    bin_ids = bin_ids - 1
    for b in range(0, max(bin_ids)):
        bin_mask = bin_ids == b
        confidence_bins.append(np.mean(max_probs[bin_mask]))
        bin_acc = np.sum(preds_class[bin_mask] == labels[bin_mask]).astype(np.float32)/len(labels[bin_mask])
        accuracy_bins.append(bin_acc)
        count_bins.append(len(labels[bin_mask]))
    return np.array(confidence_bins), np.array(accuracy_bins), np.array(count_bins)


def get_toplabel_calibration_error(labels, probs, num_bins=10):
    """Top-label calibration error over all classes."""
    # Note: the bins with no samples have NaN values.
    confidence_bins, accuracy_bins, count_bins = get_calibration_plot_arrays(labels, probs, num_bins)
    return np.nansum(count_bins/np.nansum(count_bins) * np.abs(accuracy_bins - confidence_bins))


def get_marginal_calibration_error(labels, probs, num_bins=10):
    """Marginal calibration error (or average calibration error), which averages the TCE of individual classes"""
    tce_per_class = []
    labels_unique = np.unique(labels)
    for label in labels_unique:
        l_mask = labels == label
        tce_per_class.append(get_toplabel_calibration_error(labels=labels[l_mask], probs=probs[l_mask], num_bins=num_bins))
    return np.mean(np.array(tce_per_class))


def plot_calibration(confidence_bins, accuracy_bins, figsize=(12, 8)):
    """ Calibration plot (reliability diagram) for multi-class problems"""

    plot_settings()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(confidence_bins, accuracy_bins)
    # plot identity line
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.)

    ticks = np.arange(0, 1.1, 0.1)
    ax.set_xticks(ticks, minor=False)
    ax.set_yticks(ticks, minor=False)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.grid()
    return fig


def get_calibration_metrics(probs, labels, num_bins=10):
    results = {}
    # Note: get_calibration_error() expects labels to be of shape (n,). i.e. class indices not one-hot encodings.
    # Adaptive bins with equal number of samples
    results['MCE_equal'] = cal.get_calibration_error(probs=probs, labels=labels, p=2, debias=False, mode='marginal')
    results['TCE_equal'] = cal.get_calibration_error(probs=probs, labels=labels, p=2, debias=False, mode='top-label')

    # Fixed bins with equal intervals
    results['TCE'] = get_toplabel_calibration_error(labels=labels, probs=probs, num_bins=num_bins)
    results['MCE'] = get_marginal_calibration_error(labels=labels, probs=probs, num_bins=num_bins)
    return results

