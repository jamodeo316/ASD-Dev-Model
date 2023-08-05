import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Below are functions for analyzing and plotting neural network data

def plot_loss_over_time(loss_over_time_list, session_list):
    """
    This function takes a list of loss over time (which contains the losses calculated for
    each image in a given session) and a list of session numbers that correspond to each loss
    value. It calculates the mean loss for each session using its associated losses repeatedly
    for bootstrapped 95% confidence intervals. These means with confidence intervals are plotted
    over the session numbers found in the session list.

    Parameters
    ----------
    loss_over_time_list
    session_list
    """

    sns.pointplot(x=session_list, y=loss_over_time_list, estimator='mean', errorbar=('ci', 95),
                  markers=['o', 's', 'D'], scale=0.85, linestyles='--', errwidth=1.75, capsize=0.15)
    plt.xlabel('Session', fontsize=12)
    plt.ylabel('Mean Loss', fontsize=12)

    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    x_ticks = MultipleLocator(9)
    ax.xaxis.set_major_locator(x_ticks)
    plt.xlim(0, max(session_list))

    sns.despine()
    plt.show()
