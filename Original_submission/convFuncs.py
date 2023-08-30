'''
Code for modeling apparent Pb loss distributions

Accompanyment to "XYZ" submitted to Geochronology
Glenn R. Sharman and Matthew A. Malkowski

'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Distributions used
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import weibull_min
from scipy.stats import pareto
from scipy.stats import gamma
from scipy.stats import uniform
from scipy.stats import rayleigh
from scipy.stats import gengamma
from scipy.stats import halfnorm
from scipy.stats import lognorm
from scipy.stats import rv_discrete

# Statistical tests
from scipy.stats import kstest
from astropy.stats import kuiper

# Other functions
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF

supported_dists = ['expon','gamma','pareto','uniform','weibull','rayleigh', 'gengamma', 'halfnorm', 'lognorm', 'constant', 'none', 'isolated']

def Pb_loss_fun(params, dist_type, x):
    '''
    Function that returns a distribution of Pb loss, in %

    Parameters:
    params : array of distribution parameters
             for 'none' : params are not used
             for 'constant' : [0] = % shift
             for 'isolated' : [0] = % shift, [1] = proportion shifted
             for 'uniform': [0] = u_min, [1] = delta_u
             for 'gamma': [0] = scale paramter (1/beta), [1] = shape paramter a
             for 'expon': [0] = scale parameter (1/lamda)
             for 'rayleigh': [0] = scale parameter
             for 'weibull': [0] = scale parameter, [1] = shape parameter, c
             for 'pareto': [0] = scale parameter, [1] = shape parameter b
             for 'halfnorm': [0] = scale parameter
             for 'lognorm' : [0] = scale parameter, [1] = shape parameter s
    dist_type : string, specifies which distribution type to use (options are listed in supported_dists)
    x : array, x-axis values in % over which to evaluate the Pb_loss_function

    Returns:
    Pb_loss_pct_pdf : array, relative probability of Pb-loss evaluted over x
    '''

    if not dist_type in supported_dists:
        print('Warning: given dist_type is not supported')

    if dist_type == 'none':
        pb_loss_func_const = rv_discrete(values=([0],[1]))
        cumulative_probs_const = pb_loss_func_const.cdf(x)
        Pb_loss_pct_pdf = np.gradient(cumulative_probs_const, len(x))
    if dist_type == 'constant':
        pb_loss_func_const = rv_discrete(values=([params[0]],[1]))
        cumulative_probs_const = pb_loss_func_const.cdf(x)
        Pb_loss_pct_pdf = np.gradient(cumulative_probs_const, len(x))
    if dist_type == 'isolated':
        pb_loss_func_const = rv_discrete(values=([0,params[0]],[1-params[1],params[1]]))
        cumulative_probs_const = pb_loss_func_const.cdf(x)
        Pb_loss_pct_pdf = np.gradient(cumulative_probs_const, len(x))
    if dist_type == 'uniform':
        Pb_loss_pct_pdf = uniform.pdf(x=-x, loc=params[0], scale=params[1])
    if dist_type == 'gamma':
        Pb_loss_pct_pdf = gamma.pdf(x=-x, loc=0, a=params[1], scale=params[0])
    if dist_type == 'expon':
        Pb_loss_pct_pdf = expon.pdf(x=-x, loc=0, scale=params[0])
    if dist_type == 'rayleigh':
        Pb_loss_pct_pdf = rayleigh.pdf(x=-x, loc=params[0], scale=params[0])
    if dist_type == 'weibull':
        Pb_loss_pct_pdf = weibull_min.pdf(x=-x, c=params[1], loc=0, scale=params[0])
    if dist_type == 'pareto':
        Pb_loss_pct_pdf = pareto.pdf(x=-x, loc=-1, scale=1, b=params[0]) # Pareto is shifted
    if dist_type == 'gengamma':
        Pb_loss_pct_pdf = gengamma.pdf(x=-x, loc=0, scale=params[0], a=params[1], c=params[2])
    if dist_type == 'halfnorm':
        Pb_loss_pct_pdf = halfnorm.pdf(x=-x, loc=0, scale=params[0])
    if dist_type == 'lognorm':
        Pb_loss_pct_pdf = lognorm.pdf(x=-x, loc=0, s=params[1], scale=params[0])

    Pb_loss_pct_pdf[np.isinf(Pb_loss_pct_pdf)] = 0 # To avoid values very close to x=0 that become infinite
    Pb_loss_pct_pdf = Pb_loss_pct_pdf/np.sum(Pb_loss_pct_pdf)

    return Pb_loss_pct_pdf    

def cdf_fun(xage, conv_Ma_pdf):
    '''
    Helper function used in the K-S test
    Returns CDF for values given along xage array
    '''
    cdf = interp1d(xage, np.cumsum(conv_Ma_pdf))
    return cdf

def misfit_conv(params_Pb_loss, dist_type, norm_loc, norm_scale, x, xage, dates_nonCA, method='ss', verbose=False):
    '''
    Calculates misfit between a specified normal distribution and an ECDF
    Used as input to scipy.optimize.minimize() function when modeling data with
    both non-CA and CA U-Pb dates

    Parameters:
    params_Pb_loss : list, parameters that are input to Pb_loss_fun()
    dist_type : str, type of distribution modeled (see supported_dists)
    norm_loc : float, mean of normal distribution
    norm_scale : float, 1 s.d. of normal distribution
    x : list or array, x-axis values (in %)
    xage : list or array, x-axis values (in Ma)
    dates_nonCA : list or array, U-Pb dates (non-CA)
    method : string, options: 'ss'
    verbse : bool, set to True to print out parameters during minimization

    Returns:
    misfit : float, estimate of misfit between modeled CDF and ECDF
    '''
    import ot

    if verbose:
        print(params_Pb_loss)

    Pb_loss_pct_pdf = Pb_loss_fun(params_Pb_loss, dist_type, x)

    rv_norm_Ma = norm(loc=norm_loc, scale=norm_scale)
    norm_Ma_pdf = rv_norm_Ma.pdf(xage)
    norm_Ma_pdf = norm_Ma_pdf/np.sum(norm_Ma_pdf) # Normalize so area under the curve = 1

    conv_pdf = convolve(Pb_loss_pct_pdf, norm_Ma_pdf, mode='same')
    conv_pdf = conv_pdf/np.sum(conv_pdf)

    if method == 'ss':
        ecdf = ECDF(dates_nonCA)(xage)
        cdf_conv = np.cumsum(conv_pdf)
        misfit = np.sum((ecdf-cdf_conv)**2)

    return misfit

def misfit_ECDF(params, dist_type, x, xage, dates_input, method='ss', verbose=False):
    '''
    Calculates misfit between an unspecified normal distribution and an ECDF
    Used as input to scipy.optimize.minimize() function when modeling data with
    non-CA U-Pb dates

    Parameters:
    params : list, parameters with mean and 1s.d. of normal distriubtion and parameters that are input to Pb_loss_fun()
    dist_type : str, type of distribution modeled (see supported_dists)
    x : list or array, x-axis values (in %)
    xage : list or array, x-axis values (in Ma)
    dates_input : list or array, U-Pb dates (non-CA)
    method : string, options: 'ss'
    verbse : bool, set to True to print out parameters during minimization

    Returns:
    misfit : float, estimate of misfit between modeled CDF and ECDF
    '''

    params_norm = params[0:2]
    params_Pb_loss = params[2:]

    if verbose:
        print(params_Pb_loss)

    Pb_loss_pct_pdf = Pb_loss_fun(params_Pb_loss, dist_type, x)

    rv_norm_Ma = norm(loc= params_norm[0], scale = params_norm[1])
    norm_Ma_pdf = rv_norm_Ma.pdf(xage)
    norm_Ma_pdf = norm_Ma_pdf/np.sum(norm_Ma_pdf) # Normalize so area under the curve = 1

    conv_pdf = convolve(Pb_loss_pct_pdf, norm_Ma_pdf, mode='same')
    conv_pdf = conv_pdf/np.sum(conv_pdf)

    if method == 'ss':
        ecdf = ECDF(dates_input)(xage)
        cdf_conv = np.cumsum(conv_pdf)
        misfit = np.sum((ecdf-cdf_conv)**2)

    return misfit

def misfit_norm(params, xage, dates_CA, method='ss', verbose=False):
    '''
    Calculates misfit between CA U-Pb dates and a normal distribution
    Used as input to scipy.optimize.minimize() function when modeling data with
    CA U-Pb dates

    Parameters:
    params : list, parameters with mean and 1s.d. of normal distriubtion
    xage : list or array, x-axis values (in Ma)
    dates_CA : list or array, U-Pb dates (CA)
    method : string, options: 'ss'
    verbse : bool, set to True to print out parameters during minimization

    Returns:
    misfit : float, estimate of misfit between modeled CDF and ECDF
    '''
    rv_norm_Ma = norm(loc= params[0], scale = params[1])
    norm_Ma_pdf = rv_norm_Ma.pdf(xage)
    norm_Ma_pdf = norm_Ma_pdf/np.sum(norm_Ma_pdf) # Normalize so area under the curve = 1

    if method == 'ss':
        ecdf = ECDF(dates_CA)(xage)
        cdf_conv = np.cumsum(norm_Ma_pdf)
        misfit = np.sum((ecdf-cdf_conv)**2)

    return misfit

def misfit_poly(params_Pb_loss, dist_type, gmm_Ma_pdf, x, xage, dates_nonCA, method='ss', verbose=False):

    '''
    Calculates misfit between a specified pdf that represents multiple normal distributions and an ECDF
    Used as input to scipy.optimize.minimize() function when modeling data with
    multiple age modes

    Parameters:
    params_Pb_loss : list, parameters that are input to Pb_loss_fun()
    dist_type : str, type of distribution modeled (see supported_dists)
    gmm_Ma_pdf : array, pdf values of multi-modal Gaussian distribution
    x : list or array, x-axis values (in %)
    xage : list or array, x-axis values (in Ma)
    dates_nonCA : list or array, U-Pb dates (non-CA)
    method : string, options: 'ss'
    verbse : bool, set to True to print out parameters during minimization

    Returns:
    misfit : float, estimate of misfit between modeled CDF and ECDF
    '''

    if verbose:
        print(params_Pb_loss)

    Pb_loss_pct_pdf = Pb_loss_fun(params_Pb_loss, dist_type, x)

    conv_pdf = convolve(Pb_loss_pct_pdf, gmm_Ma_pdf, mode='same')
        
    if method == 'ss':
        ecdf = ECDF(dates_nonCA)(xage)
        cdf_conv = np.cumsum(conv_pdf)
        misfit = np.sum((ecdf-cdf_conv)**2)

    return misfit

def plot_Pb_loss_model_approach_1(params_norm, params_Pb_loss, fit, dates_input, errors_1s_input, xage, x, xlim, xlim_Pb_loss, dist_type,
    elinewidth=0.5, plot_ref_age=False, ref_age=None, ref_age_2s_uncert=None, dates_input_CA = None, errors_1s_input_CA = None):

    '''
    Function for plotting the a comparison of the model output with input data


    Parameters:
    params_norm : list, mean and 1 s.d. of normal distributions
    params_Pb_loss : list, parameters that are input to Pb_loss_fun()
    fit : float, lowest misfit value from scipy.optimize.minimize()
    dates_input : list or array, U-Pb dates (non-CA)
    errors_1s_input : list or array, 1-sigma uncertainties of U-Pb dates (non-CA)
    xage : list or array, x-axis values (in Ma)
    x : list or array, x-axis values (in %)
    xlim : list or tuple, min and max x-axis values (Ma)
    xlim_Pb_loss : list or tuple, min and max x-axis values (%)
    dist_type : str, type of distribution modeled (see supported_dists)
    elinewidth : float, error bar width
    plot_ref_age : bool, set to True to plot the reference age
    ref_age : float, reference age
    ref_age_2s_uncert : float, 2-sigma uncertainty of reference age
    dates_input_CA : list or array, U-Pb dates (CA)
    errors_1s_input_CA : list or array, 1-sigma uncertainties of U-Pb dates (CA)

    Returns:
    fig : matplotlib figure
    '''

    dates_errors_1s = list(zip(dates_input, errors_1s_input))
    dates_errors_1s.sort(key=lambda d: d[0]) # Sort based on age
    y_toplot = np.linspace(1/len(dates_input), 1, len(dates_errors_1s)) # First one plots a little above 0 line

    if dates_input_CA is not None:
        dates_errors_1s_CA = list(zip(dates_input_CA, errors_1s_input_CA))
        dates_errors_1s_CA.sort(key=lambda d: d[0]) # Sort based on age
        y_toplot_CA = np.linspace(1/len(dates_errors_1s_CA), 1, len(dates_errors_1s_CA)) # First one plots a little above 0 line

    Pb_loss_pct_pdf = Pb_loss_fun(params_Pb_loss, dist_type, x)

    rv_norm_Ma = norm(loc= params_norm[0], scale = params_norm[1])
    norm_Ma_pdf = rv_norm_Ma.pdf(xage)
    norm_Ma_pdf = norm_Ma_pdf/np.sum(norm_Ma_pdf) # Normalize so area under the curve = 1

    conv_Ma_pdf = convolve(Pb_loss_pct_pdf, norm_Ma_pdf, mode='same')
    
    ecdf = ECDF(dates_input)
    if dates_input_CA is not None:
        ecdf_CA = ECDF(dates_input_CA)

    fig, ax = plt.subplots(1, figsize=(8,5))
    
    ax.errorbar([x[0] for x in dates_errors_1s], y_toplot, xerr = np.asarray([x[1]*2 for x in dates_errors_1s]), ecolor='red', elinewidth=elinewidth, fmt='none', color='red')
    if dates_input_CA is not None:
        ax.errorbar([x[0] for x in dates_errors_1s_CA], y_toplot_CA, xerr = np.asarray([x[1]*2 for x in dates_errors_1s_CA]),
         ecolor='navy', elinewidth=elinewidth, fmt='none', color='red')
    ax.plot(xage, np.cumsum(conv_Ma_pdf), '--', color='red')
    ax.plot(xage, ecdf(xage), '-', color='red')
    ax.plot(xage, np.cumsum(norm_Ma_pdf), '--', color='navy')
    if dates_input_CA is not None:
        ax.plot(xage, ecdf_CA(xage), '-', color='navy')
    ax.axvline(params_norm[0], ls='--', color='navy')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0, 1)

    if plot_ref_age:
        ax.axvline(ref_age, ls='-', color='orange')
        if ref_age_2s_uncert is not None:
            #add rectangle to plot
            ax.add_patch(Rectangle((ref_age-ref_age_2s_uncert, 0), ref_age_2s_uncert*2, 1, facecolor='orange', alpha=0.5))  
    
    ax.text(0.05, 0.80, 'n: '+str(len(dates_input)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    ax.text(0.05, 0.75, '\u03BC (Ma): '+str(np.round(params_norm[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    ax.text(0.05, 0.70, '\u03C3 (Ma): '+str(np.round(params_norm[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)

    if dist_type == 'none':
        ax.text(0.05, 0.92, 'Model fit (none)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'constant':
        ax.text(0.05, 0.92, 'Model fit (constant)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, '% shift: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'isolated':
        ax.text(0.05, 0.92, 'Model fit (isolated)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, '% shift: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, '% of grains shifted: '+str(np.round(params_Pb_loss[1]*100,1)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'expon':
        ax.text(0.05, 0.92, 'Model fit (exponential)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'shape: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'halfnorm':
        ax.text(0.05, 0.92, 'Model fit (half norm)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'lognorm':
        ax.text(0.05, 0.92, 'Model fit (lognorm)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape: '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'rayleigh':
        ax.text(0.05, 0.92, 'Model fit (Rayleigh)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'weibull':
        ax.text(0.05, 0.92, 'Model fit (weibull)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape: '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'pareto':
        ax.text(0.05, 0.92, 'Model fit (pareto)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'shape: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'uniform':
        ax.text(0.05, 0.92, 'Model fit (uniform)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'u (min): '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'u (max): '+str(np.round(params_Pb_loss[0]+params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'gamma':
        ax.text(0.05, 0.92, 'Model fit (gamma)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape: '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)    
    if dist_type == 'gengamma':
        ax.text(0.05, 0.92, 'Model fit (generalized gamma)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape (a): '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)    
        ax.text(0.05, 0.55, 'shape (c): '+str(np.round(params_Pb_loss[2],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)    

    ax.text(0.025, 1.02, '2\u03C3 uncertainties shown', size='x-small', horizontalalignment='left', transform=ax.transAxes)

    ax.set_xlabel('Age (Ma)')

    # Create a smaller subplot to show the distribution of Pb loss in the sample
    ax_sub = ax.inset_axes([0.7, 0.15, 0.27, 0.3], transform=ax.transAxes)

    ax_sub.plot(x, np.cumsum(Pb_loss_pct_pdf))
    ax_sub.axvline(x=0.0, ymin=0, ymax=1, ls='--', color='gray')
    ax_sub.set_title('''Cumulative
    apparent Pb loss''', ha='center')

    ax_sub.set_xlim(xlim_Pb_loss[0],xlim_Pb_loss[1])
    ax_sub.set_ylim(0,1)
    ax_sub.set_xlabel('Pb loss (%)')

    return fig

def plot_Pb_loss_model_poly(norm_Ma_pdf, conv_Ma_pdf, params_Pb_loss, fit, dates_input, errors_1s_input, xage, x, xlim, xlim_Pb_loss, dist_type,
    elinewidth=0.5, plot_ref_age=False, ref_age=None, ref_age_2s_uncert=None, dates_input_CA = None, errors_1s_input_CA = None):

    '''
    Function for plotting the a comparison of the model output with input data for multi-modal distributions

    Parameters:
    norm_Ma_pdf : list or array, pdf of input multi-modal normal distribution
    conv_Ma_pdf : list or array, pdf of input convolved distribution
    params_Pb_loss : list, parameters that are input to Pb_loss_fun()
    fit : float, lowest misfit value from scipy.optimize.minimize()
    dates_input : list or array, U-Pb dates (non-CA)
    errors_1s_input : list or array, 1-sigma uncertainties of U-Pb dates (non-CA)
    xage : list or array, x-axis values (in Ma)
    x : list or array, x-axis values (in %)
    xlim : list or tuple, min and max x-axis values (Ma)
    xlim_Pb_loss : list or tuple, min and max x-axis values (%)
    dist_type : str, type of distribution modeled (see supported_dists)
    elinewidth : float, error bar width
    plot_ref_age : bool, set to True to plot the reference age
    ref_age : float, reference age
    ref_age_2s_uncert : float, 2-sigma uncertainty of reference age
    dates_input_CA : list or array, U-Pb dates (CA)
    errors_1s_input_CA : list or array, 1-sigma uncertainties of U-Pb dates (CA)

    Returns:
    fig : matplotlib figure
    '''

    dates_errors_1s = list(zip(dates_input, errors_1s_input))
    dates_errors_1s.sort(key=lambda d: d[0]) # Sort based on age
    y_toplot = np.linspace(1/len(dates_input), 1, len(dates_errors_1s)) # First one plots a little above 0 line

    if dates_input_CA is not None:
        dates_errors_1s_CA = list(zip(dates_input_CA, errors_1s_input_CA))
        dates_errors_1s_CA.sort(key=lambda d: d[0]) # Sort based on age
        y_toplot_CA = np.linspace(1/len(dates_errors_1s_CA), 1, len(dates_errors_1s_CA)) # First one plots a little above 0 line

    Pb_loss_pct_pdf = Pb_loss_fun(params_Pb_loss, dist_type, x)

    ecdf = ECDF(dates_input)
    if dates_input_CA is not None:
        ecdf_CA = ECDF(dates_input_CA)

    fig, ax = plt.subplots(1, figsize=(8,5))
    
    ax.errorbar([x[0] for x in dates_errors_1s], y_toplot, xerr = np.asarray([x[1]*2 for x in dates_errors_1s]), ecolor='red', elinewidth=elinewidth, fmt='none', color='red')
    if dates_input_CA is not None:
        ax.errorbar([x[0] for x in dates_errors_1s_CA], y_toplot_CA, xerr = np.asarray([x[1]*2 for x in dates_errors_1s_CA]),
         ecolor='navy', elinewidth=elinewidth, fmt='none', color='red')
    ax.plot(xage, np.cumsum(conv_Ma_pdf), '--', color='red')
    ax.plot(xage, ecdf(xage), '-', color='red')
    ax.plot(xage, np.cumsum(norm_Ma_pdf), '--', color='navy')
    if dates_input_CA is not None:
        ax.plot(xage, ecdf_CA(xage), '-', color='navy')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0, 1)

    if plot_ref_age:
        ax.axvline(ref_age, ls='-', color='orange')
        if ref_age_2s_uncert is not None:
            #add rectangle to plot
            ax.add_patch(Rectangle((ref_age-ref_age_2s_uncert, 0), ref_age_2s_uncert*2, 1, facecolor='orange', alpha=0.5))  
    
    ax.text(0.05, 0.80, 'n: '+str(len(dates_input)), size='x-small', horizontalalignment='left', transform=ax.transAxes)

    if dist_type == 'none':
        ax.text(0.05, 0.92, 'Model fit (none)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'constant':
        ax.text(0.05, 0.92, 'Model fit (constant)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, '% shift: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    
    if dist_type == 'isolated':
        ax.text(0.05, 0.92, 'Model fit (isolated)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, '% shift: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, '% of grains shifted: '+str(np.round(params_Pb_loss[1]*100,1)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'expon':
        ax.text(0.05, 0.92, 'Model fit (exponential)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'shape: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'halfnorm':
        ax.text(0.05, 0.92, 'Model fit (half norm)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'lognorm':
        ax.text(0.05, 0.92, 'Model fit (lognorm)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape: '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'rayleigh':
        ax.text(0.05, 0.92, 'Model fit (Rayleigh)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'weibull':
        ax.text(0.05, 0.92, 'Model fit (weibull)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape: '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'pareto':
        ax.text(0.05, 0.92, 'Model fit (pareto)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'shape: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'uniform':
        ax.text(0.05, 0.92, 'Model fit (uniform)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'u (min): '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'u (max): '+str(np.round(params_Pb_loss[0]+params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
    if dist_type == 'gamma':
        ax.text(0.05, 0.92, 'Model fit (gamma)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape: '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)    
    if dist_type == 'gengamma':
        ax.text(0.05, 0.92, 'Model fit (generalized gamma)', horizontalalignment='left', size='small', transform=ax.transAxes)
        ax.text(0.05, 0.85, 'fun: '+'{:.3f}'.format(float(fit)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.65, 'scale: '+str(np.round(params_Pb_loss[0],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.05, 0.60, 'shape (a): '+str(np.round(params_Pb_loss[1],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)    
        ax.text(0.05, 0.55, 'shape (c): '+str(np.round(params_Pb_loss[2],2)), size='x-small', horizontalalignment='left', transform=ax.transAxes)    

    ax.text(0.025, 1.02, '2\u03C3 uncertainties shown', size='x-small', horizontalalignment='left', transform=ax.transAxes)

    ax.set_xlabel('Age (Ma)')

    # Create a smaller subplot to show the distribution of Pb loss in the sample
    ax_sub = ax.inset_axes([0.7, 0.15, 0.27, 0.3], transform=ax.transAxes)

    ax_sub.plot(x, np.cumsum(Pb_loss_pct_pdf))
    ax_sub.axvline(x=0.0, ymin=0, ymax=1, ls='--', color='gray')
    ax_sub.set_title('''Cumulative
    apparent Pb loss''', ha='center')

    ax_sub.set_xlim(xlim_Pb_loss[0],xlim_Pb_loss[1])
    ax_sub.set_ylim(0,1)
    ax_sub.set_xlabel('Pb loss (%)')

    return fig