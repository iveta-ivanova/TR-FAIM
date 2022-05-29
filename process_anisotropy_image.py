# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 00:14:20 2022

@author: Iveta

This class contains the functions required to process raw anisotropy data as per requirements below. 

Experimental assumptions under this work: 
    sdt.data[1] - M2 TCSPC card - perpendicular decay 
    sdt.data[0] - M1 TCSPC card - parallel decay 
    in case of lifetime data where the parallel detector has been shut down, sdt.data[1] will be empty 

The data has to be reshaped in (t,x,y) format, in accordance with numpy array shape convention. 

"""

import numpy as np 

from sdtfile import *
from scipy import ndimage
from anisotropy_functions import align_peaks
import matplotlib.pyplot as plt

class process_anisotropy_image():
    '''
    
    Requires path: 
        path = directory/file.sdt 
        G - G factor 
    
    Processes raw .sdt anisotropy files 
    
    Includes additional methods to: 
        create lifetime matrix 
        create 2D projection of all photons (intensity img)

    '''
    def __init__(self, image_path): 
        self.image_path = image_path
        self.perp_img = None
        self.par_img = None
        self.timens = None
        self.img2D = None
        self.lifetime_img = None
    
    def process_image(self): 
        '''
        This function reads any Becker&HicWkl .sdt file, uses sdt
        library (from sdtfile import *) to return the file in SdtFile
        format. Then it uses numpy to process the stdfile object 
        and return the parallel and perpendicular 3D arrays, and 
        the time (in ns) array. 
        
        Input: 
            string, full path to image in format dir/file.sdt  
        Returns: 
            perp_img - 3D anisotropy raw data of the perpendicular compoent 
            par_img - 3D anisotropy raw data of the parallel compoent 
            timens - array of the time channels in nanoseconds 
            
            
        '''
        data = SdtFile(self.image_path)  # returns sdtfile object 
        self.perp_img = np.array(data.data[1]).transpose((2, 0, 1))  # return an x by y by t matrix
        self.par_img = np.array(data.data[0]).transpose((2, 0, 1))
        time = np.array(data.times[0])  # return array of times in seconds
        self.timens = np.multiply(time, 1000000000)  # time array in nanoseconds
            
    def align_par_perp_img(self): 
        t,x,y = self.perp_img.shape 
        
        for i in range(x): 
            for j in range(y):
                align_peaks(self.perp_img[:,i,j], self.par_img[:,i,j], self.timens, plot = False)
        
        print('Image peak alignment finalised.')
                
    
    def create_lifetime_img(self, par_img, perp_img, G): 
        """
        This function takes chosen raw perpendicular and parallel files 
        and creates a lifetime data matrix by pixel-wise application
        of the formula:
            Ipar + 2 * G * Iperp.
        It is important the input image shape is (t,x,y), i.e. the time
        channel is first and both images have identical shape.
    
        Parameters
        ----------
        perp : perpendicular image, a 3D array of shape (t,x,y)
        par : parallel image, a 3D array of shape (t,x,y)
    
        Returns
        -------
        lifetime_img : a single 3D array (image) containing lifetime decays
        in each pixel.
        """
        t,x,y = self.perp_img.shape
 
        self.lifetime_img = np.empty_like(perp_img)
        
        self.lifetime_img[:] = np.add(par_img[:], (2.0 * G * perp_img[:]))
            
    def get_cumulative_projection(self, img):
        
        """
        This function takes an image stack (3D array) and sums up all
        values in all time bins into a single plane i.e. creates an intensity
        projection of the 3D image.
        Each pixel of the 2D projection contains the total sum of
        photons in all the time channels.
    
        Parameters
        ----------
        img : image data, a 3D array of shape (t,x,y)
    
        Returns
        -------
        2D "intensity" image  
    
        """
        t,x,y = img.shape
        projection = np.empty_like(img)
        for i in range(x):  # iterate columns
            for j in range(y):  # iterate along rows
                projection[:, i, j] = np.sum(img[:, i, j])  # [:] - sums all of them along the time axis
        self.img2D = projection[0, :, :]
    
        plt.imshow(self.img2D, cmap = 'gray')
        plt.pause(7)
        plt.close()
        return self.img2D
        
        
        
        
        