# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:26:32 2021

@author: Iveta
"""

"""

Experimental assumptions under this work: 
    sdt.data[1] - M2 TCSPC card - perpendicular decay 
    sdt.data[0] - M1 TCSPC card - parallel decay 
    in case of lifetime data where the parallel detector has been shut down, sdt.data[1] will be empty 

The data has to be reshaped in (t,x,y) format, in accordance with numpy array shape convention. 

Future: 
    0. parametrise the functions so they can easily be done for anisotropy as well as lifetime decays 
    1. control when matplotlib plots close and stay open  
    2. intergrate G factor in the main code 
    3. interactive plots - draw vertical lines and save them after they are closed 
    4. log file containing the following info 
    - chosen indexes 
    - 
"""

import os
import numpy as np
from sdtfile import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import scipy.optimize
from PIL import Image
from scipy.signal import convolve, deconvolve
from scipy import ndimage
import scipy.sparse
import math
import pandas as pd 

def fill_zeros(data):  # think of a better way to do this
    """
    Replace zeros in a 2D array with the average of its neighbouring elements.
    """
    filled = data.copy()
    count = 0  # count how many zeros will be replaced
    for i in range(len(filled) - 1):
        if filled[i] == 0:
            filled[i] = np.ceil(np.mean([filled[i - 1], filled[i + 1]], axis=0))
            count += 1
        else:
            pass
    print(f"Total of {count} zeros have been replaced, {((count/len(data))*100)}% of all values ")
    return filled  # print(indices)


def load_anifit_from_image(par_img,perp_img, IRF, x, y, time):
    '''
    Creates a pandas dataframe to hold the decay data 
    from a given pixel at x,y coordinates of choice 
    of both a parallel and a perpendicular component image. 
    The function aligns the par and perp decays. 
    Then it aligns the IRF to the par decay peak 
    and normalises it to its height. 
    
    Use df.to_csv() to save as .csv file. 
    
    Parameters:
        par_img, perp_img - anisotropy images (3D arrays of each component)
        x, y - pixel indices of the single decays of interest
        time - array of time in ns 
        IRF - full IRF array - same length 
    
    Returns:
        single .csv file with all relevant decays. 
             In line with requierments from AniFit software, the 
             columns are: TimeVV, VV, TimeVH, VH, TimeIRF, IRF
        
    '''
    ani_file = pd.DataFrame()
    parallel = par_img[:, x, y].flatten()
    perp = perp_img[:, x, y].flatten()
    perp, parallel, peak_idx, _ = align_peaks(perp, parallel, time, plot = True, norm = False)    
    #IRF = IRF * (max(par)/max(IRF))
    # align IRF (back) to the parallel decay and normalised wrt it 
    parallel, IRF, _, _ = align_peaks(parallel, IRF, time, plot = True, norm = True)
    
    # how many photons are at the peak? 
    
    print(f''''Peak photon count 
          VV = {np.amax(parallel)}, 
          VH = {np.amax(perp)}
          ''')
    
    ani_file = ani_file.assign(timeVV = time)
    ani_file = ani_file.assign(VV = parallel)
    ani_file = ani_file.assign(timeVH = time)
    ani_file = ani_file.assign(VH = perp)
    ani_file = ani_file.assign(timeIRF = time)
    ani_file = ani_file.assign(IRF = IRF)
    return ani_file

def load_ani_decays(par_decay, perp_decay, IRF, time): 
    ani_file = pd.DataFrame()
    perp_decay, par_decay, _, _ = align_peaks(perp_decay, par_decay, time, plot = True, norm = False)
    par_decay, IRF, _, _ = align_peaks(par_decay, IRF, time, plot = True, norm = True)
    
    ani_file = ani_file.assign(timeVV = time)
    ani_file = ani_file.assign(VV = par_decay)
    ani_file = ani_file.assign(timeVH = time)
    ani_file = ani_file.assign(VH = perp_decay)
    ani_file = ani_file.assign(timeIRF = time)
    ani_file = ani_file.assign(IRF = IRF)
    
    print(f''''Peak photon count 
      VV = {np.amax(par_decay)}, 
      VH = {np.amax(perp_decay)}
      ''')
          
    return ani_file
    
def cumulative_decay(image):
    '''
    Takes a 3D image with dimensions (t,x,y) and outputs a single decay. 
    The decay is a result of summing all the pixels in each time bin. 
    '''
    t,x,y = image.shape
    decay = np.array([])
    for i in range(t): 
        decay = np.append(decay, image[i].sum())
    print(f'Maximum photon count in this array is {np.amax(decay)}')
    return decay 

def shift_IRF(IRF, t, shift, plot = False):
    '''
    the IRF should be shift by some amount of time bins 
    (usually 2)
    
    '''
    IRF_shifted = np.pad(IRF, (0, shift), mode = "constant")[shift:]
    if plot: 
        plt.plot(t, IRF, label= 'initial')
        plt.plot(t, IRF_shifted, label = 'shifted')
        plt.legend(loc = 'upper right')
        plt.show()
        plt.pause(10)
        plt.title('shifted IRF')
        plt.close()
        
    return IRF_shifted 
    
    
def create_lifetime_decay_img(perp, par, G):
    """
    This function takes the perpendicular and parallel images
    and creates a lifetime image by pixel-wise application
    of the formula Ipar + 2 * G * Iperp.
    It is important the input image shape is (t,x,y), i.e. the time
    channel is first and both images have identical shape.

    Parameters
    ----------
    perp : perpendicular image, a 3D array of shape (t,x,y)
    par : parallel image, a 3D array of shape (t,x,y)

    Returns
    -------
    lifetime : a single 3D array (image) containing lifetime decays
    in each pixel.
    """
    if perp.shape != par.shape:
        print("Per and par images have uneven size, check transform!")
        # break
    else:
        pass
    dims = perp.shape
    t = dims[0]
    x = dims[1]
    y = dims[2]
    lifetime = np.empty_like(perp)
    # for i in range(x):
    #    for j in range(y):
    #        lifetime[:,i,j] = np.add(par[:,i,j], (2.0*G*perp[:,i,j]))
    # changed it to be vectorized
    lifetime[:] = np.add(par[:], (2.0 * G * perp[:]))
    return lifetime

def create_anisotropy_img(perp,par,G): 
    '''
    Assume t,x,y array format 
    
    '''

def percent_in_bounds(image, bounds):
    '''
    Calculate and prints the % of non-nan values within
    the given range
    
    image - 2D numpy array of parameters 
    bounds - list of values  
    '''
    lower = bounds[0]
    upper = bounds[1]
    x,y = image.shape
    # how many pixels total that are not np.nan ? 
    
    # get rid off nans and make it into an array 
    oneD = image[~np.isnan(image)].ravel()
    total = len(oneD)
    #print(type(oneD[0]))
    # set up a counter for successful 
    in_bounds = []
    counter = 0
    for val in oneD:
        if lower <= val <= upper:
            in_bounds.append(val)
            counter += 1
        else: 
            pass 
    
    percent_in_bounds = (counter/total)*100
    print(f'Percent within bounds is: {percent_in_bounds}')
    return in_bounds, percent_in_bounds
            

def plot_peak_intensity_histogram(image, mask): 
    '''
    Make a list of all the peak intensities within this image
    Plot histogram
    '''
    peak_photon_list = []
    t,x,y = image.shape
    for j in range(x): 
        for i in range(y):
            #print(f'pixel: {j} x {i} ')
            if mask[:,j,i].any() == False:
                pass
            else:
                decay = image[:,j,i]
                # get value at peak for this decay 
                peak = np.amax(decay)
                # append to list 
                peak_photon_list.append(peak)
    plt.hist(peak_photon_list, bins = 50)
    plt.xlabel("Photons at peak")
    plt.ylabel("Frequency")
    plt.show()
    return peak_photon_list

def plot_histogram(image, x, y, range_, title):
    """
    x - int - the x coord where the mean and std will be written
    y - int - the y coord where the mean and std will be written
    range_ - tuple - histogram range to be shown
    title - string - title shown above
    """
    plt.hist(image.ravel(), bins=300, range=range_)
    plt.xlabel(title, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    mean = np.mean(image[~np.isnan(image)].ravel())
    sdt = np.std(image[~np.isnan(image)].ravel())
    plt.axvline(x=mean, color="r", linewidth=2)
    plt.text(
        x,
        y,
        f"{mean.round(2)} ± {sdt.round(2)}",
        bbox=dict(facecolor="yellow", alpha=0.5),
        fontsize=16,
    )


def show_param_map(image, range_, title):
    plt.imshow(image, clim=range_)
    plt.suptitle(title, fontsize=20)
    plt.colorbar()


def image_to_decay(image):
    """
    This function sums the photons in each time slice of an image and
    returns a single decay of the same time dimension as the image.

    PARAMETERS:
        image - 3D array - TCSPC image

    RETURNS:
        decay - 1D array (float) - same dimension as the input image
    """

    t, x, y = image.shape
    decay = np.zeros(t)
    for z in range(t):
        decay[z] = np.sum(image[z, :, :])
    return decay


def bin_image_convolve(data, filter_type, kernel_size, mode):
    """
    Bin image by applying a kernel of a specified size.
    Uses the ndimage convolve function.

    Filter_type: 
        convolve = ndimage.convolve
        Gaussian = ndimage.gaussian_filter
        median = ndimage.median_filter
        
    """
    dims = data.shape
    t = dims[0]
    x = dims[1]
    y = dims[2]
    binned = np.zeros_like(data)
    kernel = np.ones((kernel_size, kernel_size), dtype=int)
    # print(kernel)
    # print(f'This image has {t} time bins and pixel dims of {x} x {y}')
    if filter_type == "convolve": 
        for i in range(t):
            binned[i] = ndimage.convolve(data[i], kernel, mode=mode)
    elif filter_type == "Gaussian":
        for i in range(t): 
            binned[i] = ndimage.gaussian_filter(data[i], sigma = kernel_size)
    elif filter_type == "median": 
        for i in range(t): 
            binned[i] = ndimage.median_filter(data[i], size = kernel_size)
    
    return binned


def bin_image(data):
    """
    This function takes an image stack (3D array) and performs a
    3x3 binning (or bin=1 by SPCImage / Becker & Hickl convention),
    by summing the photons of the immediate neighbouring pixels in each
    stack/slice/time channel.
    The image data has to be of the shape (t,x,y), i.e. the time
    channel dimension has to be first. This can be achieved by
    data.transform((2,0,1))

    Parameters
    ----------
    data : image data, a 3D array (e.g. x, y and time dimension)
    xdim : size of x dimension of image
    ydim : size of y dimension of image
    t : number of stacks/slices/time channels

    Returns
    -------
    binned : 3x3 binned image ; array of float64

    """
    print("Binning started, this might take a while.")
    dims = data.shape
    t = dims[0]
    x = dims[1]
    y = dims[2]
    # print(f'Image.shape: {data.shape}')
    binned = np.zeros_like(data)  # create empty array of same dims to hold the data
    # print(f'Empty bin dimensions: {binned.shape}')
    # print(binned)
    for i in range(x):  # iterate columns
        # print(f'x = {i} of total {x} pixels ')
        for j in range(y):  # iterate along rows
            # print(f'y = {j} of total {y} pixels ')
            if i == 0:  # entire first row
                if j == 0:  # top left corner
                    binned[:, i, j] = np.sum(
                        [[data[:, i, j] for j in range(j, j + 2)] for i in range(i, i + 2)],
                        axis=0,
                    ).sum(axis=0)
                elif j == y - 1:  # top right corner
                    binned[:, i, j] = np.sum(
                        [[data[:, i, j] for j in range(j - 1, j + 1)] for i in range(i, i + 2)],
                        axis=0,
                    ).sum(axis=0)
                else:  # on edge (non-corner) of first row
                    binned[:, i, j] = np.sum(
                        [[data[:, i, j] for j in range(j - 1, j + 2)] for i in range(i, i + 2)],
                        axis=0,
                    ).sum(axis=0)

            elif i == x - 1:  # entire last row
                if j == 0:  # bottom left corner
                    binned[:, i, j] = np.sum(
                        [[data[:, i, j] for j in range(j, j + 2)] for i in range(i - 1, i + 1)],
                        axis=0,
                    ).sum(axis=0)
                elif j == y - 1:  # bottom right corner
                    binned[:, i, j] = np.sum(
                        [
                            [data[:, i, j] for j in range(j - 1, j + 1)]
                            for i in range(i - 1, i + 1)
                        ],
                        axis=0,
                    ).sum(axis=0)
                else:  # on edge (non-corner) of last row
                    binned[:, i, j] = np.sum(
                        [
                            [data[:, i, j] for j in range(j - 1, j + 2)]
                            for i in range(i - 1, i + 1)
                        ],
                        axis=0,
                    ).sum(axis=0)

            elif j == 0:  # entire first column, corners should be caught by now
                binned[:, i, j] = np.sum(
                    [[data[:, i, j] for j in range(j, j + 2)] for i in range(i - 1, i + 2)],
                    axis=0,
                ).sum(axis=0)

            elif j == y - 1:  # entire last column
                binned[:, i, j] = np.sum(
                    [[data[:, i, j] for j in range(j - 1, j + 1)] for i in range(i - 1, i + 2)],
                    axis=0,
                ).sum(axis=0)

            else:  # if on the inside of matrix
                binned[:, i, j] = np.sum(
                    [[data[:, i, j] for j in range(j - 1, j + 2)] for i in range(i - 1, i + 2)],
                    axis=0,
                ).sum(axis=0)
    print("Binning done.")
    return binned


def cum_projection(data):
    """
    This function takes an image stack (3D array) and sums up all
    values in all time bins into a single plane i.e. creates a
    projection of the 3D image.
    Each pixel of the 2D projection contains the total sum of
    photons in all the time channels.

    Parameters
    ----------
    data : image data, a 3D array of shape (t,x,y)
    xdim : size of x dimension of image
    ydim : size of y dimension of image
    t : number of stacks/slices/time channels

    Returns
    -------
    binned : 2D "intensity" image 

    """
    t, x, y = data.shape
    # print(f'Image dimensions from params: {x,y,t}')
    # print(f'Image.shape: {data.shape}')
    projection = np.empty_like(data)  # create empty array of same dims to hold the data
    # print(f'Empty bin dimensions: {binned.shape}')
    # print(binned)
    for i in range(x):  # iterate columns
        for j in range(y):  # iterate along rows
            projection[:, i, j] = np.sum(data[:, i, j])  # [:] - sums all of them along the time axis
    return projection[0, :, :]


# def convolve_IRF(IRF, model, time, plot = True):

# convolved_model = IRF*

def align_image(perp, par, time): 
    t,x,y = par.shape 
    peaks = []
    for i in range(x): 
        for j in range(y):
            perp[:,i,j], par[:,i,j], peak, _ = align_peaks(perp[:,i,j], par[:,i,j], time, plot = False)
            peaks.append(peak)
    print('Image peak alignment finalised.')
    peaks = np.array(peaks)
    return perp, par, peaks

def deconvolve_IRF(IRF, decay, time, plot=False):
    """
    This function deconvolves the instrument
    response function from the observed lifetime decay.
    Pass plot = True to plot the IRF and the convolved decay, normalised to the same height.
    It uses the numpy convolution function.

    Requires:
        align_peaks from anisotropy_functions

    Parameters
    ----------
    IRF : numpy array of instrument response function
    decay : numpy array of decay we want to convolve
    time : numpy array of time vector in nanoseconds
    plot : Boolean : default = False

    Returns
    -------
    deconvolved : 2D array - the convolved decay normalised to the height
    of the raw (input) decay

    """
    # dx = time[1] - time[0]
    # convolved = scipy.signal.convolve(IRF, decay, mode = 'full', method = 'fft')[:len(time)]

    IRF = IRF * (max(decay) / max(IRF))

    # convolved = np.convolve(IRF,decay)[:len(time)]

    # print(decay.shape)
    # print(time.shape)
    # print(IRF.shape)

    # print(type(IRF))

    # minIRF = min(IRF)
    # print(minIRF)
    # IRF = np.where(IRF == 0, 0.1, IRF)
    # decay = np.where(decay == 0, 0.1, decay)
    # print(IRF[0])

    # deconvolve returns a tuple of (signal,remainder)
    # deconvolution result is zero??
    # deconv = scipy.signal.deconvolve(decay, IRF)[0]

    N = 2 ** (math.ceil(math.log2(len(IRF) + len(decay))))

    print(f"N: {N}")
    filtered = np.fft.ifft(np.fft.fft(decay, N) / np.fft.fft(IRF, N))

    print(f" Real shape: {filtered.real.shape}")
    print(f" Imaginary shape: {filtered.imag.shape}")

    # the deconvolution has n = len(signal) - len(gauss) + 1 points
    # extend by difference between decay and deconv
    # s = int((len(decay)-(len(deconv)))/2)  ## diff - 1 on each side
    # on both sides (hence divide by 2)
    # deconv_res = np.zeros(len(decay))   ##will hold the final result - same length as original decay

    # if (len(decay)-len(deconv)%2==0):   # if difference is even
    #    end = int(len(decay)-s-1)  # final index - same length as
    # else:
    #    print('Difference is uneven')
    #    end = int(len(decay)-s-1)

    # print(s)
    # print(end)
    # print(deconv_res.shape)
    # print(deconv.shape)

    # deconv_res[s:end] = deconv
    # deconv = deconv_res
    # convolved normalized to decay height
    # print(deconvolved)
    # deconv_n = deconv * (max(decay)/max(deconv))
    # deconvolved_n = deconvolved
    # print(len(deconvolved_n))
    deconv = filtered.real
    timenew = np.array(range(len(deconv)))
    # align peaks for a plot only
    # convolved_n, IRF, _, _ = align_peaks(convolved_n, IRF, time, plot = False)
    if plot:
        plt.plot(timenew, deconv, label="deconvolved")
        # plt.plot(timenew, filtered.real, label = 'real')
        #    #plt.plot(time[:len(IRF)], IRF, label = 'IRF')
        #    #plt.plot(time, deconv_n, label = 'deconvolved')
        # plt.legend(loc = 'lower left', fontsize= 14)
        #    plt.xlabel('Time(ns)', fontsize= 16)
        #    plt.ylabel('Counts', fontsize= 16)
        plt.show()
    return filtered


def plot_IRF_decay(IRF, decay, time, scale):
    """
    This function plots an IRF and any decay, by normalising
    them to the same height.
    By default it provides a linear plot; passing 'scale = "log"'
    plots a logarithmic plot instead.
    """
    IRFnorm = IRF * (max(decay) / max(IRF))
    plt.plot(time, IRFnorm, label="IRF")
    plt.plot(time, decay, label="Lifetime decay")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Time(ns)", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    if scale == "log":
        plt.yscale("log")
    plt.show()


def read_sdt(cwd, filename):
    """
    This function reads any Becker&Hickl .sdt file, uses sdt
    library (from sdtfile import *) to return the file in SdtFile
    format. This file is later used in process_sdt() to read the individual
    decays and times in numpy array format.

    Parameters
    ----------
    cwd : the current working directory where your files are (obtain by os.getcwd())
    filename : string representing the name of the .sdt file

    Returns
    -------
    data : data read by SdtFile - SdtFile object

    """
    cwd = cwd
    # path = os.path.join(cwd, folder)
    path = os.path.join(cwd, filename)
    data = SdtFile(path)
    return data


def read_gfactor(cwd):
    """
    This function reads the Gfactor single decay .sdt file into an SdtFile
    type. It searches the current working directory for a file containing
    the 'gfactor' key word => the Gfactor file HAS TO CONTAIN 'gfactor' in
    its filename. If more than one file contain 'gfactor' in their filename,
    the second file will be read.

    Parameters
    ----------
    cwd : the current working directory where your files are (obtain by os.getwd())

    Returns
    -------
    data : data read by SdtFile

    """
    for file in os.listdir(cwd):
        if "gfactor" in file and file.endswith(".sdt"):
            path = os.path.join(cwd, file)
            print(file)
    data = SdtFile(path)
    return data


def process_sdt_image(data):
    """
    This function takes the SdtObject (raw anisotropy image data)
    and returns the perpendicular and parallel matrix decays as 3D
    numpy arrays. The arrays are transposed to have a shape (t,x,y),
    as this is the required

    Parameters
    ----------
    data : SdtObject (image with parallel and perpendicular decay component)

    Returns
    -------
    perpimage : uint16 : 3D numpy array holding perpendicular image data
    transposed into shape (t,x,y).
    parimage : uint16 : 3D numpy array holding perpendicular image data
    transposed into shape (t,x,y).
    time : float64 1D numpy array holding the time in seconds
    timens : float64 1D numpy array holding the time in nanoseconds
    bins : scalar, number of time bins

    """
    perpimage = np.array(data.data[1]).transpose((2, 0, 1))  # return an x by y by t matrix
    parimage = np.array(data.data[0]).transpose((2, 0, 1))
    time = np.array(data.times[0])  # return array of times in seconds
    bins = len(time)
    ns = 1000000000
    timens = np.multiply(time, ns)  # time array in nanoseconds
    return perpimage, parimage, time, timens, bins


def process_sdt(data):
    """
    This function takes the single SdtFile anisotropy file,
    and reads the perpendicular (1), parallel (2) decays
    and the time bins into numpy arrays.
    It assumes the time is in nanoseconds, and uses that to
    create a time vector of nanoseconds.

    Parameters
    ----------
    data :SdtFile anisotropy data with two channels (one for
    each component - parallel and perpendicular)

    Returns
    -------
    perpdecay : uint16 array of the perpendicular decay histogram
    pardecay : uint16 array of the parallel decay histogram
    time : float64 array of the times in nanoseconds
    bins: number of time bins

    """
    perpdecay = np.array(data.data[1]).flatten()  ##  M2 - perpendicular
    pardecay = np.array(data.data[0]).flatten()  ## M1 - parallel
    time = np.array(data.times[0])
    bins = len(time)
    ns = 1000000000
    time = np.multiply(time, ns)  ## convert to ns
    return perpdecay, pardecay, time, bins


## plot decay in log
def plot_decays(decay1, decay1_name, decay2, decay2_name, time, scale, norm=False, title = None ):
    """
    Plots the perpendiculal and parallel decays on the same plot.

    Parameters
    ----------
    decay1 : data 1 as numpy array
    decay1_name : the label we want on this data 
    decay2 : data 2 as numpy array
    decay2_name : the label we want on this data 
    time: time vector as numpy array
    scale : if want log scale, pass "log" as argument; otherwise scale = None
        (the default is linear)
    norm: should both decays be normalised to the same counts? Default is False
    title : any string you want as title of your plot; default: None

    Returns
    -------
    None. Just shows the plot

    """
    if norm:
        decay1 = decay1 * (max(decay2) / max(decay1))
    fig = plt.figure()
    plt.plot(time, decay2, label=decay2_name)
    plt.plot(time, decay1, label=decay1_name)
    plt.xlabel("Time(ns)", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.suptitle(title, fontsize=16)
    plt.legend(loc="upper right", fontsize=14)
    if scale == "log":
        plt.yscale("log")
    plt.show()


def get_data_from_pixel(user_img, time, data_imgs, legends, label): 
    
    '''
    
    This function takes a 2D image, lets the user select 
    a pixel on this image, and outputs data from the selected 
    3D matrices.
    
    Parameters: 
        user_img - 2D image
        time - time array (for plotting)
        data_imgs - list of 3D data matrices 
    
    '''
    plt.figure(0)
    plt.imshow(user_img)
    indices = plt.ginput(1)  # outputs list of tuples 
    row = int(round(indices[0][1],0))
    column = int(round(indices[0][0],0))
    print(f'Pixel chosen : {row} x {column} ')
    #peak_idx = int(peaks[row,column])
    #counter = 1 
    for data_img,legend in zip(data_imgs, legends):
        #counter += 1
        plt.figure(1)
        #y = data_img[peak_idx:,row,column]
        #x = time[:(len(time)-peak_idx)]
        y = data_img[:,row,column]
        x = time
        plt.plot(x, y, label = legend )
        plt.legend(loc = 'upper right', fontsize = 16)
        plt.xlabel('Time (ns)', fontsize = 16)
        plt.ylabel(label, fontsize = 16)
        plt.show()
        

def find_nearest_neighbor_index(array, points):
    """

    Helper function for background_subtract().
    Takes the time array and the points chosen
    by the user, and finds the indexes of the nearest
    time bin that the user-clicked point corresponds to.

    """
    store = []
    for point in points:
        distance = (array - point) ** 2
        idx = np.where(distance == distance.min())
        idx = idx[0][0]  # take first index
        store.append(idx)
    return store


def background_subtract(perpdecay, pardecay, time):
    """
    This function plots logarithmically the two decays
    components. The user has to choose two points on
    the plot to show the area where the background counts are
    (i.e. before the peak)
    The average photon intensity/counts within this time range
    is calculated from the perpendicular decay (arbitrary choice)
    and subtracted from each time bin in each decay.

    Requires find_nearest_neighbor_index() function.

    Parameters
    ----------
    perpdecay : perpendicular component of data as numpy arrays
    pardecay : parallel component of data as numpy array
    time: time vector as numpy array

    Returns
    -------
    BG_indices : TYPE
        DESCRIPTION.
    perpdecay : background subtracted perpendicular decay data as numpy array
    pardecay : background subtracted parallel decay data as numpy array
    BG : the averaged background used for the subtraction

    """
     
    # Plot decay
    x = time
    y1 = perpdecay
    y2 = pardecay
    # plt.switch_backend('Qt5Agg')
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel("Time (ns)")
    plt.ylabel("Intensity (photons)")
    plt.yscale("log")
    plt.title("Select background range (2 points)")
    plt.show()

    # Select background region and draw on plot
    boundaries = plt.ginput(2)
        
    # Update L with x_points array
    boundaries_x = []
    for i in range(len(boundaries)):
        boundaries_x.append(boundaries[i][0])
        plt.axvline(boundaries_x[i], color="r")  # add vertical line
    # plt.savefig('Gfactor_range')
    plt.close()
    # Get indices for the background delimiters- i.e. the number of counts in these bins
    BG_indices = find_nearest_neighbor_index(time, boundaries_x)
    # Calculate background by averaging - use perpendicular decay
    # BG_sum - sum of intensities in this time window (BG_indices[0] and [1])
    BG_sum = sum([perpdecay[i] for i in range(BG_indices[0], BG_indices[1] + 1)])
    # BG_indices[1] - BG_indices[0] + 1 => how many time bins selected?
    BG = BG_sum / (BG_indices[1] - BG_indices[0] + 1)

    # Subtract background from decay
    perpdecay = perpdecay - BG
    pardecay = pardecay - BG

    plot_decays(perpdecay, "perpendicular", pardecay, "parallel", time, scale="log", title="background corrected", norm=False)
    plt.show()
    plt.pause(5)
    plt.close()
    # Replace all negative values by 0
    # perpdecay = np.where(perpdecay < 0, 0, perpdecay)
    # pardecay = np.where(pardecay < 0, 0, pardecay)

    return BG_indices, perpdecay, pardecay, BG


def background_subtract_lifetime(data, time):
    """
    This function plots logarithmically the single decay.
    The user has to choose two points on
    the plot to show the area where the background counts are
    (i.e. before the peak)
    The average photon intensity/counts within this time range
    is calculated and subtracted from each time bin in the decay.

    Requires find_nearest_neighbor_index() function.

    Parameters
    ----------
    data : numpy array that holds the data
    time: time vector as numpy array

    Returns
    -------
    BG_indices :
    data : background subtracted perpendicular decay data as numpy array
    BG : the averaged background used for the subtraction

    """

    # Plot decay
    x = time
    y = data
    # plt.switch_backend('Qt5Agg')
    plt.plot(x, y)
    plt.xlabel("Time (ns)")
    plt.ylabel("Intensity (photons)")
    plt.yscale("log")
    plt.title("Select background range (2 points)")
    plt.show()

    # Select background region and draw on plot
    boundaries = plt.ginput(2)

    # Update L with x_points array
    boundaries_x = []
    for i in range(len(boundaries)):
        boundaries_x.append(boundaries[i][0])
        plt.axvline(boundaries_x[i], color="r")  # add vertical line
    # plt.savefig('Gfactor_range')
    plt.close()
    # Get indices for the background delimiters- i.e. the number of counts in these bins
    BG_indices = find_nearest_neighbor_index(time, boundaries_x)
    # Calculate background by averaging - use perpendicular decay
    # BG_sum - sum of intensities in this time window (BG_indices[0] and [1])
    BG_sum = sum([data[i] for i in range(BG_indices[0], BG_indices[1] + 1)])
    # BG_indices[1] - BG_indices[0] + 1 => how many time bins selected?
    BG = BG_sum / (BG_indices[1] - BG_indices[0] + 1)

    # Subtract background from decay
    data = data - BG

    # plot_decay(perpdecay, pardecay, time, 'log', 'background corrected')
    plt.close()
    # Replace all negative values by 0
    # perpdecay = np.where(perpdecay < 0, 0, perpdecay)
    # pardecay = np.where(pardecay < 0, 0, pardecay)

    return BG_indices, data, BG


def get_peaks(perp, par):
    """
    Finds the time bin index of the peak for both the perpendicular
    and the parallel decay.

    Parameters
    ----------
    perpdecay : perpendicular component of data as numpy arrays
    pardecay : parallel component of data as numpy array

    Returns
    -------
    peak_perp : index of peak
    peak_par : index of peak

    """
    peak_perp = np.argmax(perp)
    peak_par = np.argmax(par)
    print(f"Perpendicular peak at index {peak_perp}")
    print(f"Parallel peak at index {peak_par}")
    return peak_perp, peak_par


def threshold_array(array, thresh, keyword, save=False):
    """
    This function creates a mask using user-defined photon count threshold,
    Values equal to and greater than this threshold are set to True, lower
    values are set to False. It returns a binary T/F array (mask)
    which is to be used for selective analysis later. It
    also returns a binary image for visualisation of the segmented image.
    Choosing the threshold is an iterative process, while trying different
    values the image is not saved (save = False)

    Once a suitable threshold is chosen, pass save = True to the
    arguments and keyword to define this image.
    Image is saved in the cwd under the name 'binarized_{keyword}.png'.

    Parameters
    --------
    array - 2d numpy array - the cumulative photon projection obtained from cum_projection
    function
    thresh - integer - a photon value
    cwd - string - the cwd directory where the binary image will be saved
    keyword - string - the word that will be added to the end of the 'binarized' file name
    save - bool - whether to save the image or not

    Returns
    --------
    im_binary - binary image (2d numpy array, values 0 / 255)
    im_bool - boolean 2d array (T/F mask)

    """
    im_bool = array >= thresh  # T/F mask- true pixels which are higher than the threshhold given
    maxval = 255
    im_binary = (array > thresh) * maxval  # image where True is 255 and False is 0
    Image.fromarray(np.uint8(im_binary)).show()
    if save:
        Image.fromarray(np.uint8(im_binary)).save(f"binarized_{keyword}.png")
    return im_binary, im_bool


def mask2Dto3D(mask2D, array3D):
    """
    This function takes a 2D T/F array (mask) and
    extends it to a 3D mask of desired shape,
    as provided by a 3D array.

    Parameters
    ----------
    mask2D : T/F array of shape (x,y)
    array3D : array of shape (z,x,y) - image
    that will be masked

    Returns
    -------
    mask3D : T/F array of shape (z,x,y)

    """
    mask3D = np.zeros_like(array3D, bool)
    mask3D[:, :, :] = mask2D[np.newaxis, :, :]
    return mask3D


def apply_savgol(
    decay,
    timens,
    window_size,
    polynomial,
    peak_idx=None,
    limit_idx_from_start=None,
    range=False,
    plot=True,
    norm=True,
):

    """

    This function applies the Savitzky Golay filter and plots
    the smoothed and the raw decay for comparison, normalized
    to the same height.

    """
    savgol = savgol_filter(decay, window_size, polynomial)  # window size, polynomial order
    if norm:  # normalize to decay height
        savgol = savgol * (max(decay) / max(savgol))
    if plot:
        plt.plot(timens, decay, label="Raw")
        plt.plot(timens, savgol, "r", label="SavGol", linewidth=1)
        # plt.plot(timens, fullperpsmooth, label = 'Conv')
        plt.legend(loc="upper right")
        plt.xlabel("Time (ns)", fontsize=16)
        plt.ylabel("Anisotropy (r)", fontsize=16)
        plt.pause(7)
        plt.close()
    if range:
        plt.xlim(timens[peak_idx], timens[limit_idx_from_start])
    return savgol


def align_peaks(perpdecay, pardecay, time, plot=False, norm=False, scale = None):
    
    """

    The function calculates the shift between the peaks of the two decays and shifts
    the second peak to align to the first one.

    The first decay is the leading decay - shifting and normalisation is
    performed according to its position and height.

    Takes three arguments:
        perpdecay, pardecay (array) - the arrays of perpendicular and parallel decays and
    the time array.
        plot (bool) - default True - pass plot
        norm (bool) - default False - if you want the plotted decays to be
    normalised to the same height, relative to the first decay (i.e. the
        second decay is normalised such that its height is the same as that
        of the first)

    Returns four variables:
        the decays of the peaks,aligned
        the peak index - index of the maximum counts
        the shift (in index values)

    """
    # find INDEX of max value of each decay
    maxindexperp = np.argmax(perpdecay)
    maxindexpar = np.argmax(pardecay)
    shift = maxindexpar - maxindexperp
    #print(pardecay.shape)
    #print(shift)
    if shift > 0:  # if perpendicular decay first => shift the parallel one
        pardecay = np.pad(pardecay, (0, shift), mode="constant")[shift:]
        peak_index = np.argmax(pardecay)
    else:  # and if par decay first, shift the perp one by *shift*
        perpdecay = np.pad(perpdecay, (0, abs(shift)), mode="constant")[abs(shift) :]
        peak_index = np.argmax(perpdecay)

    # normalise the second decay to be the same height as the first decay
    if norm:
        pardecay = pardecay * (max(perpdecay) / max(pardecay))

    if plot:
        # plot_decays(perpdecay, pardecay, time, scale = False, norm = norm, title = 'Peaks aligned')
        plt.plot(time, perpdecay, label = 'perp')
        plt.plot(time, pardecay, label = 'par')
        plt.title("Aligned peaks")
        plt.legend(loc = "upper right")
        # plt.axvline(x = time[peak_index])
        if scale == "log":
            plt.yscale("log")
        plt.pause(4)
        plt.close()
    #print('peaks aligned')
    return perpdecay, pardecay, peak_index, shift

def get_gfactor(Gfact, t, peak_idx):
    """
    This function takes the G factor array and makes an interactive plot,
    starting from just after the peak. The user has to choose two points where
    the G factor array has a somewhat constant fluctuation (about the middle
    of the time series). The G factor value is calculated by taking an average
    from the user-selected range.

    Requires the find_nearest_neighbor_index() function.

    Parameters
    ----------
    Gfact : Gfactor array obtained by the element-wise division of the parallel
    by the perpendicular decay

    t : time array

    peak_idx : time index of the peak

    Returns
    -------
    Gvalue : a single float G-value.

    """
    plt.plot(t, Gfact)
    plt.xlim([t[peak_idx], 50])
    plt.ylim([0, 2])
    plt.suptitle("Select G factor range", fontsize = 16)
    plt.ylabel("I(par)/I(perp)", fontsize = 16)
    plt.xlabel("Time (ns)", fontsize = 16)

    # select area that os
    boundaries = plt.ginput(2)

    # Update L with x_points array
    boundaries_x = []
    for i in range(len(boundaries)):
        # boundaries holds the value in nanoseconds
        boundaries_x.append(boundaries[i][0])
        #plt.axvline(boundaries_x[i], color="r")  # add vertical line
    print(f"Time range chosen is at {boundaries_x[0]} ns and {boundaries_x[1]} ns")
    plt.axvspan(boundaries_x[0], boundaries_x[1], alpha=0.3, color='yellow')
    plt.pause(8)
    plt.close()
    G_indices = find_nearest_neighbor_index(t, boundaries_x) # returns all the par/perp values within this range 
    G_sum = sum([Gfact[i] for i in range(G_indices[0], G_indices[1] + 1)])
    # BG_indices[1] - BG_indices[0] + 1 => how many time bins selected?
    Gvalue = G_sum / (G_indices[1] - G_indices[0] + 1)
    return Gvalue


def plot_lifetimes(DOPCtotal, DPPCtotal, time, scale=True, title="Lifetime decays"):
    """
    Accepts the total intensity of two different samples (here DOPC and DPPC:Chol, in this order),
    calculated by Ipar + 2*G*Iperp, normalizes according to the decay with the greater intensity
    and plots them on a logarithmic scale.

    It is possible to plot the non-normalized decays by passing scale = False, as well as
    to change the default title, by passing title = 'My title'.
    """
    if scale:
        # find which one has the greater intensity and scale the other decay
        # according to it
        if DOPCtotal.max() > DPPCtotal.max():
            scaler = MinMaxScaler(feature_range=(DOPCtotal.min(), DOPCtotal.max()))
            DPPCtotal = scaler.fit_transform(DPPCtotal.reshape(-1, 1))
        else:
            scaler = MinMaxScaler(feature_range=(DPPCtotal.min(), DPPCtotal.max()))
            DOPCtotal = scaler.fit_transform(DOPCtotal.reshape(-1, 1))
    plt.plot(time, DOPCtotal, label="DOPC")
    plt.plot(time, DPPCtotal, label="DPPC:Chol")
    plt.legend(loc="upper right", fontsize=14)
    plt.suptitle(title, fontsize=16)
    plt.yscale("log")
    plt.ylabel("Normalized counts", fontsize=16)
    plt.xlabel("Time (ns", fontsize=16)
    plt.show()


def get_anisotropy_decay(
    perp,
    par,
    G,
    time,
    bgcorr=True,
    align=True,
    from_peak=True,
    plot=False,
    title="Anisotropy decay",
):
    """
    This function takes the perp and parallel component, as well as the time vector
    and the G value. After optional background subtraction (requires user input) and
    peak alignment, it calculates the anisotropy decay as per [(Ipar - GIperp)/Itotal]
    and plots it from the peak onwards (pass from_peak = False to plot the entire decay).

    Returns:
        r - the full anisotropy decay
        peak_idx - the peak index of the aligned decays - 10
    """
    # first bg correction if not already done
    if bgcorr:
        _, perp, par, _ = background_subtract(perp, par, time)
    else:
        perp = perp
        par = par
    # second peak alignment
    if align:
        perp, par, peak_idx, shift = align_peaks(perp, par, time, plot=False)
    else:
        perp = perp
        par = par
        peak_idx, _ = get_peaks(perp, par)

    #peak_idx = peak_idx - 20
    plt.cla()
    plt.plot(time, perp)
    plt.plot(time, par)
    plt.axvline(x=time[peak_idx], color="r")
    plt.pause(2)
    plt.yscale("log")
    plt.close()

    total = np.add(par, (2 * G * perp))
    r_decay = np.divide((par - G * perp), total)

    if plot:
        plt.cla()
        plt.plot(time, r_decay)
        if from_peak:
            plt.xlim(time[peak_idx], 50)  # plot only from peak
        else:
            plt.axvline(x=time[peak_idx], color="r")
        plt.ylim(-0.3, 0.5)
        plt.suptitle(title, fontsize=16)
        plt.xlabel("Time (ns)", fontsize=16)
        plt.ylabel("Anisotropy (r)", fontsize=16)
        plt.pause(5)
        plt.close()
    return r_decay, peak_idx, total


def choose_anisotropy_limit(r, t, peak_idx):
    """
    This function plots the anisotropy decay from
    the decay peak onwards, and lets the user choose the
    final point where the anisotropy model will take place.
    The model model range should be chosen such that the
    anisotropy is relatively stable and not noisy.
    The function returns the chosen anisotropy range,
    as well as the index of the user-chosen point,
    starting from t=0 (not from the peak)

    Parameters
    ----------
    r : 1D array of anisotropy decay
    t : 1d array pf time in ns
    peak_idx :

    Returns
    -------
    r : the chosen anistropy range, from peak to final index
    t : the time array for the anistropy range, from peak to final index
    limit_idx : the index of the final final point,
    starting from t=0 (not from the peak)

    """
    # chop off data before peak
    r = r[peak_idx:]
    t = t[peak_idx:]

    plt.plot(t, r)
    plt.xlabel("Time(ns)")
    plt.ylabel("Anisotropy(r)")
    plt.ylim([-0.5, 1.0])
    plt.suptitle("Choose a final point of your decay (1 point)", fontsize=16)
    boundary = plt.ginput(1)  # let user choose
    limit_idx = find_nearest_neighbor_index(t, boundary[0])
    plt.close()
    limit_idx = limit_idx[0]  # final point

    # now chop off data after the user selection
    r = r[:limit_idx]
    t = t[:limit_idx]

    plt.plot(t, r)
    plt.xlabel("Time(ns)", fontsize=16)
    plt.ylabel("Anisotropy(r)", fontsize=16)
    plt.ylim([-0.5, 1.0])
    plt.suptitle("Range that will be analysed")

    limit_idx = peak_idx + limit_idx
    return r, t, limit_idx


def highlight_chosen_ani_range(r, t, peak_idx, limit_idx):
    """
    This function plots the full anisotropy decay, and
    highlights with vertical lines the range that will
    be model to the model.

    Parameters
    ----------
    r : 1D array of FULL anisotropy decay
    t : 1D array of full time series in ns
    peak_idx : index where peak starts
    limit_idx : index chosen by user for final limit
    of anisotropy decay

    Returns
    -------
    None. Just plots the anisotropy decay.

    """
    plt.plot(t, r)
    plt.xlabel("Time(ns)", fontsize=16)
    plt.ylabel("Anisotropy(r)", fontsize=16)
    plt.axvline(x=t[peak_idx], c="r", linewidth=4)
    plt.axvline(x=t[limit_idx], c="r", linewidth=4)
    # plt.fill_between(y = 1.0, x1 = timens[peak_idx_max], x2 = timens[limit_idx], c = ())
    plt.ylim([-0.5, 1.0])
    plt.xlim([0, 40])
    plt.show()


# some small helpers functions
def print_decay_stats(decay):
    bins = decay.shape[0]
    print(f"Number of time bins: {bins}")
    print("-------------")
    print("Number of NaNs:")
    print(
        f"{sum(np.isnan(decay))} NaN values, or {((sum(np.isnan(decay))/bins)*100).round(1)} % of all bins."
    )
    print("-------------")
    print("Number of zero values:")
    print(
        f"{bins-np.count_nonzero(decay)} zero values, or {(((bins-np.count_nonzero(decay))/bins)*100)} % of all bins."
    )
    print("-------------")
    print("Number of negative values:")
    print(
        f"{sum(np.array(decay)<0)} negative values, or {(((sum(np.array(decay)<0))/bins)*100).round(1)} % of all bins."
    )


def count_zeros(perpdecay, pardecay, bins=4096):
    """
    Returns the number of zero values in the perpendicular
    and parallel decay, respectivaly. Requires the number
    of time bins - default value is 4096.

    """
    zerosperp = bins - np.count_nonzero(perpdecay)  # 3316
    zerospar = bins - np.count_nonzero(pardecay)  # 3313
    return zerosperp, zerospar


def count_negatives(perpdecay, pardecay):
    """
    Returns the number of negative values in the perpendicular
    and parallel decay, respectivaly.
    """
    negativeperp = np.sum(np.array(perpdecay) < 0)
    negativepar = np.sum(np.array(pardecay) < 0)
    return negativeperp, negativepar


def simulate_rinf_conv_single(p1, p2, p3, perp, par, time, G, IRF, conv=True):
    """
    p1 - r0
    p2 - theta
    p3 - r_inf
    """
    total = np.add(par, (2 * G * perp))

    perp, par, peak_index, _ = align_peaks(perp, par, time, plot=False)
    # plt.pause(5)
    # plt.close()
    perp, total, peak_index, _ = align_peaks(perp, total, time, plot=False)
    # plt.pause(5)
    # plt.close()

    #
    perp_s = apply_savgol(perp, time, 9, 2, plot=True, norm=False)
    plt.pause(10)
    plt.close()
    par_s = apply_savgol(par, time, 9, 2, plot=True, norm=False)
    plt.pause(10)
    plt.close()
    total_s = apply_savgol(total, time, 9, 2, plot=True, norm=False)
    plt.pause(10)
    plt.close()

    # normalize to raw decay height
    perp_s = perp_s * (max(perp) / max(perp_s))
    par_s = par_s * (max(par) / max(par_s))
    total_s = total_s * (max(total) / max(total_s))

    # get rid of 0s
    perp_s = np.where(perp_s < 0.1, 0.1, perp_s)
    par_s = np.where(par_s < 0.1, 0.1, par_s)
    total_s = np.where(total_s < 0.1, 0.1, total_s)

    # get weights
    w_perp = 1 / np.sqrt(perp_s)
    w_par = 1 / np.sqrt(par_s)
    # weights_total =np.divide(1, np.sqrt(totalimage_s))

    print(perp_s)
    print(w_perp)

    plt.plot(time, perp_s, "r")
    plt.plot(time, w_perp, "b")
    plt.plot(time, perp)
    plt.pause(20)
    plt.close()

    t = time

    # print(t)
    if conv:
        model_func_perp = lambda IRF, t, total, G, perp, p1, p2: (
            (
                max(perp)
                / max(np.convolve(IRF, ((total / 3) * G * (1 - (p1 * np.exp(-t / p2)))))[: len(t)])
            )
            * (np.convolve(IRF, (total / 3) * G * (1 - (p1 * np.exp(-t / p2))))[: len(t)])
        )
        model_func_par = lambda IRF, t, total, par, p1, p2: (
            (
                max(par)
                / max(np.convolve(IRF, ((total / 3) * (1 + 2 * (p1 * np.exp(-t / p2)))))[: len(t)])
            )
            * (np.convolve(IRF, ((total / 3) * (1 + 2 * (p1 * np.exp(-t / p2)))))[: len(t)])
        )

        objective = (
            lambda p, IRF, t, total, G, perp, par, w_perp, w_par: norm(
                (model_func_par(IRF, t, total, par, p[0], p[1]) - par) * w_par
            )
            ** 2
            + norm((model_func_perp(IRF, t, total, G, perp, p[0], p[1]) - perp) * w_perp) ** 2
        )
        obj = objective([p1, p2], IRF, t, total, G, perp, par, w_perp, w_par)

        # model with provided params
        y_perp_fit = model_func_perp(IRF, t, total, G, perp, p1, p2) * w_perp
        y_par_fit = model_func_par(IRF, t, total, par, p1, p2) * w_par

    else:
        model_func_perp = (
            lambda t, total, G, p1, p2: (total / 3) * G * (1 - (p1 * np.exp(-t / p2)))
        )
        model_func_par = lambda t, total, p1, p2: (total / 3) * (1 + 2 * (p1 * np.exp(-t / p2)))

        objective = (
            lambda p, t, total, G, perp, par, w_perp, w_par: norm(
                (model_func_par(t, total, p[0], p[1]) - par) * w_par
            )
            ** 2
            + norm((model_func_perp(t, total, G, p[0], p[1]) - par) * w_perp) ** 2
        )
        obj = objective([p1, p2], t, total, G, perp, par, w_perp, w_par)

        y_perp_fit = model_func_perp(t, total, G, p1, p2)
        y_par_fit = model_func_par(t, total, p1, p2)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(t, y_perp_fit, "r", label="model")
    ax[0, 0].plot(t, perp, "--", label="raw")
    ax[0, 0].set_title("Perpendicular decay")
    ax[0, 1].plot(t, y_par_fit, "r", label="model")
    ax[0, 1].plot(t, par, "--", label="raw")
    ax[0, 1].set_title("Parallel decay")
    # ax[1,0].plot(t, res_perp)
    # ax[1,1].plot(t, res_par)
    for ax in ax[0, :].flat:
        ax.set(xlabel="Time (ns)", ylabel="Counts")
        # ax.set(xlim = (t,30))
    plt.show()

    # get objective function
    # objective = lambda p, t, total, G, y_perp, y_par, w_perp, w_par: norm((model_func_par(t,total, p[0], p[1], p[2]) - y_par)*w_par)**2 + norm(( model_func_perp(t,total, G, p[0], p[1], p[2]) - y_par)*w_perp)**2

    # objective with these particular initial params
    # obj = objective([p1,p2,p3], t, total_s, G, perp, par, w_perp, w_par)

    print(f"Objective function = {obj}")
    return obj, y_perp_fit, y_par_fit


# def simulate_r_NLLS_hindered(r, time, r0, rinf, theta, SG_window, SG_polynomial)


def simulate_par_NLLS_IRF(perp, par, time, G, IRF, SG_window, SG_polynomial, r00, theta0, rinf0):
    """
    p1 - r0
    p2 - theta
    p3 - r_inf
    """
    t = time
    p10, p20, p30 = r00, theta0, rinf0

    # convolve with IRF
    par_conv = np.convolve(IRF, par)[: len(t)]
    perp_conv = np.convolve(IRF, perp)[: len(t)]

    # normalize convolved to raw decay height
    par_conv = par_conv * (max(par) / max(par_conv))
    perp_conv = perp_conv * (max(perp) / max(perp_conv))

    # =============================================================================
    #     plt.plot(time, par,  '-', label = 'raw decay')
    #     plt.plot(time, par_conv, label = 'convolved')
    #     plt.ylabel('Counts', fontsize = 16)
    #     plt.xlabel('Time (ns)', fontsize = 16)
    #     plt.legend(loc = 'upper right')
    #     plt.show()
    #     plt.pause(20)
    #     plt.close()
    #
    # =============================================================================
    # align peaks for perp and par
    perp_conv, par_conv, peak_idx, _ = align_peaks(perp_conv, par_conv, t, plot=False)

    # calculate lifetime
    total = np.add(par_conv, (2 * G * perp_conv))

    # =============================================================================
    #     plt.plot(t, par_conv, label = 'parallel')
    #     plt.plot(t, perp_conv, label = 'perpendicular')
    #     plt.plot(t, total, label = 'lifetime')
    #     plt.title('Convolved and aligned data')
    #     plt.legend(loc = 'upper right')
    #     plt.ylabel('Counts', fontsize = 16)
    #     plt.xlabel('Time (ns)', fontsize = 16)
    #     plt.show()
    #     plt.pause(15)
    #     plt.close()
    # =============================================================================

    # apply savitzky golay filter
    par_s = apply_savgol(par, time, SG_window, SG_polynomial, plot=True, norm=True)
    # total_s = apply_savgol(total, time, SG_window, SG_polynomial,plot = False, norm = True)

    # normalize smoothed decay to raw decay height
    par_s = par_s * (max(par) / max(par_s))
    # total_s = total_s * (max(total)/max(total_s))

    # take away zeros
    par_s = np.where(par_s < 0.1, 0.1, par_s)
    # total_s = np.where(total_s < 0.1, 0.1, total_s)
    print(peak_idx)
    # get weights
    w = 1 / np.sqrt(par_s)

    peak_idx = peak_idx - 1
    # choose data to be used for fits
    y = par_conv[peak_idx:]
    total = total[peak_idx:]
    w = w[peak_idx:]
    t = t[peak_idx:]

    # y = par
    # total = total
    # t = t

    model_func = lambda t, total, p1, p2, p3: (total / 3) * (
        1 + 2 * ((p1 - p3) * np.exp(-t / p2) + p3)
    )

    p = [p10, p20, p30]

    # sum of differences, weighted by Poisson noise from Sav Gol filter
    error_func = lambda p, t, total, y, w: (y - (model_func(t, total, p[0], p[1], p[2]))) * w
    # error_func = lambda p, t, total, y, w: (y - y) * w
    # bounds = ([0, 1], [0,10], [-0.2, 1])

    # Root Mean Squared error - reduced chi squared
    s_sq = np.sum(error_func(p, t, total, y, w) ** 2) / (len(y) - len(p))

    # Root Mean Squared error - residuals
    res = error_func(p, t, total, y, w) / (len(y) - len(p))

    print("Reduced chi-square = ", s_sq)
    # print("Error on function = ", error_func(p,t,total,y,w))

    y_fit = model_func(t, total, p[0], p[1], p[2])

    plt.plot(t, y_fit, "-r", label="model")
    plt.plot(t, y, ".", label="raw decay")
    plt.legend(loc="upper right", fontsize=14)
    plt.ylabel("Counts", fontsize=16)
    plt.xlabel("Time (ns)", fontsize=16)
    plt.title(f"r0 = {p[0]}, theta = {p[1]}, rinf = {p[2]}, chi_sq = {s_sq.round(2)}")

    return res, y_fit, y
