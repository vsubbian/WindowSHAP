import numpy as np
import math
from copy import deepcopy

def xai_eval_fnc(model, relevence, input_x, model_type='lstm', percentile=90,
                 eval_type='prtb', seq_len=10, by='all'):
    """
    Evaluates the quality metrics of time-series importance scores using various evaluation methods.

    Parameters
    ----------
    model : prediction model that is explained 
    relevance : A 3D array of importance scores for each time step of the time-series data
    input_x : input data of the prediction model. If the input data consists of different modalities, the first module should be a 3D time series data
    model_type (optional) : type of model, either 'lstm' or 'lstm_plus'. Use 'lstm' when the time series data is the only modality of the input, otherwise use 'lstm_plus'
    percentile (optional) : percentile of top time steps that are going to be pertubed
    eval_type (optional) : evaluation method, either 'prtb' for the perturbation analysis metric or 'sqnc' for sequence analysis metric
    seq_len (optional) : sequence length for 'sqnc' method
    by (optional) : whether to evaluate each temporal feature separately or all time steps together, either 'time' or 'all'

    Returns : prediction of the modified input time-series data using the input model
    """
    
    input_new = deepcopy(input_x)
    relevence = np.absolute(relevence)
    
    # TO DO: Add other type of models
    if model_type == 'lstm_plus':
        input_ts = input_x[0]
        input_new_ts = input_new[0]
    elif model_type == 'lstm':
        input_ts = input_x
        input_new_ts = input_new
    
    assert len(input_ts.shape)==3 # the time sereis data needs to be 3-dimensional
    num_feature = input_ts.shape[2]
    num_time_step = input_ts.shape[1]
    num_instance = input_ts.shape[0]
        
    if by=='time':
        top_steps = math.ceil((1 - percentile/100) * num_time_step) # finding the number of top steps for each feature
        top_indices = np.argsort(relevence, axis=1)[:, -top_steps:, :] # a 3d array of top time steps for each feature
        for j in range(num_feature): # converting the indices to a flatten version
            top_indices[:, :, j] = top_indices[:, :, j] * num_feature + j
        top_indices = top_indices.flatten()
    elif by=='all':
        top_steps = math.ceil((1 - percentile/100) * num_time_step * num_feature) # finding the number of all top steps
        top_indices = np.argsort(relevence, axis=None)[-top_steps:]
    
    # Create a masking matrix for top time steps
    top_indices_mask = np.zeros(input_ts.size)
    top_indices_mask[top_indices] = 1
    top_indices_mask = top_indices_mask.reshape(input_ts.shape)
    
    
    # Evaluating different metrics
    for p in range(num_instance):
        for v in range(num_feature):
            for t in range(num_time_step):
                if top_indices_mask[p, t, v]:
                    if eval_type == 'prtb':
                        input_new_ts[p,t,v] = np.max(input_ts[p,:,v]) - input_ts[p,t,v]
                    elif eval_type == 'sqnc':
                        input_new_ts[p, t:t + seq_len, v] = 0
    
    return model.predict(input_new)


def heat_map(start, stop, x, shap_values, var_name='Feature 1', plot_type='bar', title=None):
    """
    A function that generates a heatmap with the temporal sequence alongside its Shapley values

    Parameters
    ----------
    start (int): the starting point of the temporal sequence
    stop (int): the ending point of the temporal sequence
    x (np.ndarray): the sequence data
    shap_values (np.ndarray): the Shapley values corresponding to the sequence data
    var_name (str): the name of the variable being plotted (default: 'Feature 1')
    plot_type (str): the type of plot to generate ('bar' or 'heat' or 'heat_abs', default: 'bar')
    title (str): the title for the plot (default: None)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import BoundaryNorm
    from textwrap import wrap
    import numpy as np; np.random.seed(1)
    
    ## ColorMap-------------------------
    # define the colormap
    cmap = plt.get_cmap('PuOr_r')

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(np.min(shap_values),np.max(shap_values),.005)
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    ##------------------------------------
    
    if title is None: title = '\n'.join(wrap('{} values and contribution scores'.format(var_name), width=40))
    
    if plot_type=='heat' or plot_type=='heat_abs':
        plt.rcParams["figure.figsize"] = 9,3
        if plot_type=='heat_abs':
            shap_values = np.absolute(shap_values)
            cmap = 'Reds'
        fig, ax1 = plt.subplots(sharex=True)
        extent = [start, stop, -2, 2]
        im1 = ax1.imshow(shap_values[np.newaxis, :], cmap=cmap, norm=norm, aspect="auto", extent=extent)
        ax1.set_yticks([])
        ax1.set_xlim(extent[0], extent[1])
        ax1.title.set_text(title)
        fig.colorbar(im1, ax=ax1, pad=0.1)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(start, stop), x, color='black')
    elif plot_type=='bar':
        plt.rcParams["figure.figsize"] = 8.5,2.5
        fig, ax1 = plt.subplots(sharex=True)
        mask1 = shap_values < 0
        mask2 = shap_values >= 0
        ax1.bar(np.arange(start, stop)[mask1], shap_values[mask1], color='blue', label='Negative Shapley values')
        ax1.bar(np.arange(start, stop)[mask2], shap_values[mask2], color='red', label='Positive Shapley values')
        ax1.set_title(title)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(start, stop), x, 'k-', label='Observed data')
        # legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    ax1.set_xlabel('Time steps')
    if plot_type=='bar': ax1.set_ylabel('Shapley values')
    ax2.set_ylabel(var_name + ' data values')
    plt.tight_layout()
    plt.show()