import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import os
import datetime


# Below are functions for analyzing and plotting neural network data

def plot_loss_over_time(loss_over_time_list, session_list):
    """
    This function takes a list of loss over time (which contains the losses calculated for
    each image in a given session) and a list of session numbers that correspond to each loss
    value. It calculates the mean loss for each session using its associated losses repeatedly
    for bootstrapped 95% confidence intervals. These means with confidence intervals are plotted
    over the session numbers found in the session list.

    A result folder, named according to the date and time of training, is made in the 'results'
    directory of the model if one does not exist already. The plot mentioned above is saved in
    this result folder as 'training_loss_plot'.

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

    y_max = 2
    if max(loss_over_time_list) > y_max:
        y_max = max(loss_over_time_list)
    plt.ylim(0, y_max)

    parent_dir = os.getcwd() + "/results/"
    child_dir = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm") + " results"
    try:
        os.mkdir(parent_dir + child_dir)
    except FileExistsError:
        pass
    file_name = "/training_loss_plot.png"
    plt.savefig(parent_dir + "/" + child_dir + file_name, dpi=300, bbox_inches='tight')


def export_loss_table(loss_table_list, session_list):
    """
    This function takes a loss table list comprised of dictionaries with image class (key)-list
    of image class losses (value) pairs and iterates over it (list comprehension) to convert
    each dictionary to a pandas dataframe. These dataframes are concatenated for a final
    combined dataframe with session number in  the left most column, image set index in
    the second leftmost column, and calculated loss according to image class in the remaining
    columns (column header reflects image class name).

    A result folder, named according to the date and time of training, is made in the 'results'
    directory of the model if one does not exist already. The dataframe mentioned above is saved in
    this result folder as 'training_loss_table'.

    Parameters
    ----------
    loss_table_list
    session_list
    """

    loss_dfs = [pd.DataFrame(x) for x in loss_table_list]
    final_df = pd.concat(loss_dfs, keys=set(session_list))

    parent_dir = os.getcwd() + "/results/"
    child_dir = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm") + " results"
    try:
        os.mkdir(parent_dir + child_dir)
    except FileExistsError:
        pass
    file_name = "/training_loss_table.xlsx"
    final_df.to_excel(parent_dir + child_dir + file_name, sheet_name="Loss per Class per Session")
