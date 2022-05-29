# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 12:13:55 2022

@author: Iveta
"""

'''
This class contains important functions for analysis of time-resolved 
anisotropy image data.

Needs to be run after process_anisotropy_image and after Gfactor - 
i.e. after the raw data has been processes and a G factor value 
has been calculated. 

'''

import matplotlib.pyplot as plt 
from matplotlib.widgets import RectangleSelector, LassoSelector, PolygonSelector
from matplotlib.path import Path
import numpy as np 
from anisotropy_functions import align_peaks, threshold_array, mask2Dto3D, find_nearest_neighbor_index
from scipy import ndimage, optimize
import os 
import os.path 


class image_analysis: 
    '''
    Attributes: intensity image, parallel and perp matrix. 
    These will be changed in-place during cropping. 
    
    '''
    def __init__(self, img2D, imgPar, imgPerp, timens, G): 
        self.img2D = img2D  # created from raw non-binned image 
        self.imgPar = imgPar          # raw non-binned 
        self.imgPerp = imgPerp
        self.timens = timens
        self.rect_selector = None
        self.cropped = None
        #self.ax = None
        self.G = G
        self.cropprops = dict(facecolor='yellow', 
                          edgecolor='white', 
                          alpha=0.5, 
                          fill=False)
        
        self.ROIprops = dict(color='white', 
                          linestyle='-.', 
                          linewidth = '2',
                          alpha=0.5)
        self.ROImask = None
        self.decay_par = None
        self.decay_perp = None
        self.r = None
        self.img_r = None
        self.img_par_binned = None
        self.img_perp_binned = None
        self.img_r_binned = None
        self.lifetime = None
        self.peak_idx = None
                
    def show_intensity_img(self): 
        plt.imshow(self.img2D, cmap = 'gray')
        plt.show()
    
    def align_par_perp_img(self, perp, par): 
        t,x,y = self.par.shape 
        
        for i in range(x): 
            for j in range(y):
                perp, par = align_peaks(perp[:,i,j], par[:,i,j], self.timens, plot = False)
        
        print('Image peak alignment finalised.')
        
        return par, perp
    
    def crop_select_callback(self, eclick, erelease):
        '''
        Function to be executed after crop area chosen 
        
        Both the intensity image and the parallel and perpendicular
        compoent matrices are modified to the cropped dimensions.
        
        '''
        #global x1, y1, x2, y2
        #global cropped
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        # crop 2D image that is now open and show it 
        self.img2D = self.img2D[y1:y2,x1:x2]
        self.imgPar = self.imgPar[:,y1:y2,x1:x2]
        self.imgPerp = self.imgPerp[:,y1:y2,x1:x2]
        
        print(f'New image has shape {self.img2D.shape}')
        
        self.ax.imshow(self.img2D, cmap='gray')
        #self.ROIverts = None
    
    def crop_img(self): 
        '''
        Opens image with option to draw rectangle of area to be cropped. 
        The parallel and perpendicular matrices are also cropped to the 
        desired area. 

        '''
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img2D, cmap='gray')
        self.rect_selector = RectangleSelector(self.ax, 
                                               self.crop_select_callback, 
                                               interactive = False, 
                                               drawtype = 'box', 
                                               rectprops = self.cropprops,
                                               button = [1])
        plt.show()
    
    def get_anisotropy_decay_single(self, plot = True):
        '''
        Uses current decay_par and decay_perp to calculate and plot
        anisotropy decay (r). 

        Parameters
        ----------
        G : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None 

        '''
        self.r = np.divide( (self.decay_par - self.G * self.decay_perp), ( self.decay_par + 2 * self.G * self.decay_perp))
        if plot: 
            self.fig, self.ax = plt.subplots(1,2)
            self.ax[0].plot(self.timens, self.decay_par, label = 'Par')
            self.ax[0].plot(self.timens, self.decay_perp, label = 'Perp')
            self.ax[0].set_xlabel("Time (ns)", fontsize = 14)
            self.ax[0].set_ylabel("Counts", fontsize = 14)
            self.ax[0].set_yscale("log")
            self.ax[0].legend(loc = 'upper right', fontsize = 14)
            self.ax[1].plot(self.timens, self.r)
            self.ax[1].set_xlabel("Time (ns)", fontsize = 14)
            self.ax[1].set_ylabel("Anisotropy", fontsize = 14)
            self.ax[1].set_ylim(0,0.4)
            plt.show()
    
    def convolve_img(self, img, filter_type, kernel_size, mode = "nearest"): 
        """
        Bin image by applying a kernel of a specified size.
        Uses the ndimage convolve function.
    
        Filter_type: 
            convolve = ndimage.convolve
            Gaussian = ndimage.gaussian_filter
            median = ndimage.median_filter
        
        Parameters: 
            img : 3D anisotropy raw data matrix (time bins, x, y)
            Filter_type: 
                convolve = ndimage.convolve
                Gaussian = ndimage.gaussian_filter
                median = ndimage.median_filter
            Kernel size: e.g. kernel = 3 = 3x3 matrix 
            Mode: how to treat edges, default is "nearest"
        
        """        
        t,x,y = img.shape
        binned = np.zeros_like(img)
        kernel = np.ones((kernel_size, kernel_size), dtype=int)
        if filter_type == "convolve": 
            for i in range(t):
                binned[i] = ndimage.convolve(img[i], kernel, mode=mode)
        elif filter_type == "Gaussian":
            for i in range(t): 
                binned[i] = ndimage.gaussian_filter(img[i], sigma = kernel_size)
        elif filter_type == "median": 
            for i in range(t): 
                binned[i] = ndimage.median_filter(img[i], size = kernel_size)
        #print(f'Finished binning of {img}')
        print(f'max photon count at peak is {np.amax(binned)}')
        return binned 
    
    def plot_anisotropy_from_peak(self, ylim): 
        '''
        Plot anisotropy decay with ylimits of choice. 
        Plot is created from peak of par and perp decays. 
        They are first aligned if they are not already. 
        
        Assumes anisotropy decay (self.r) already calculated. 
        
        Parameters: 
            tupil of the desired y limits. 
        
        Returns: 
            anisotropy decay from peak 
            
        '''
        # check if aligned 
        if np.argmax(self.decay_par) == np.argmax(self.decay_perp): 
            peak_idx = np.argmax(self.decay_par)
            print(f'Peak index = {peak_idx}')
        else: 
            print('.. aligning decays..')
            self.decay_par, self.decay_perp, peak_idx, _ = align_peaks(self.decay_perp, 
                                                            self.decay_par, 
                                                            self.timens, 
                                                            plot = True, 
                                                            norm = False)
        peak_idx += 1    
        plt.plot(self.timens[peak_idx:], self.r[peak_idx:])
        plt.xlabel("Time (ns)", fontsize = 16)
        plt.ylabel("Anisotropy", fontsize = 16)
        plt.ylim(ylim)
        plt.xlim(self.timens[peak_idx], 30)
        plt.show()
        return self.r[peak_idx:]
        
    def create_anisotropy_decay_img(self,img_par, img_perp, G): 
        '''
        
        Create matrix of anisotropy decays using par and perp raw data of choice. 
        
        Assign it to an attribute - either self.img_r or self.img_r_binned
        
        '''
        
        ani_img = np.divide( ( img_par - G * img_perp), (img_par + 2 * G * img_perp))
                
        return ani_img    
        
    def select_decays_pixel(self): 
        # open intensity image 
        # user clicks on pixel 
        # output is the par and perp decay, ani decay, 
        pass 
    
    def get_plots_for_pixel(self, mode, i, j, par, perp): 
        '''
        i = row number = y coordinate of image 
        j = column number = x coordinate of image
        
        '''
        fig, ax = plt.subplots(1,2)
        
        peak_idx = np.argmax(self.imgPar[:,i,j])
        
        if mode == "raw": 
            print(f'Raw decays at pixel position {(i,j)}')
            ax[0].plot(self.timens,self.imgPar[:,i,j], label = "par")
            ax[0].plot(self.timens,self.imgPerp[:,i,j], label = "perp")
            ax[0].set_yscale('log')
            ax[0].set_xlabel('Time (ns)', fontsize = 16)
            ax[0].set_ylabel('Counts', fontsize = 16)
            ax[0].legend(loc = "upper right")
            if self.img_r is not None: 
                ax[1].plot(self.timens, self.img_r[:,i,j])
                ax[1].set_xlabel('Time (ns)', fontsize = 16)
                ax[1].set_ylabel('Anisotropy', fontsize = 16)
                ax[1].set_xlim(self.timens[peak_idx], 50)
                ax[1].set_xlim(self.timens[peak_idx], 50)
            else: 
                r = np.divide((self.imgPar[:,i,j] - self.G * self.imgPerp[:,i,j]), ( self.imgPar[:,i,j] + 2 * self.G * self.imgPerp[:,i,j]))
                ax[1].plot(self.timens, r)
                ax[1].set_xlabel('Time (ns)', fontsize = 16)
                ax[1].set_ylabel('Anisotropy', fontsize = 16)
                ax[1].set_xlim(self.timens[peak_idx], 50)
                ax[1].set_xlim(self.timens[peak_idx], 50)
        elif mode == "binned": 
            print(f'Binned decays at pixel position {(i,j)}')
            peak_idx = np.argmax(par[:,i,j])
            r = np.divide((par[:,i,j] - self.G * perp[:,i,j]), ( par[:,i,j] + 2 * self.G * perp[:,i,j]))
            ax[0].plot(self.timens,par[:,i,j], label = "par")
            ax[0].plot(self.timens,perp[:,i,j], label = "perp")
            ax[0].set_yscale('log')
            ax[0].set_xlabel('Time (ns)', fontsize = 16)
            ax[0].set_ylabel('Counts', fontsize = 16)
            ax[0].legend(loc = "upper right")
            ax[1].plot(self.timens, r)
            ax[1].set_xlabel('Time (ns)', fontsize = 16)
            ax[1].set_ylabel('Anisotropy', fontsize = 16)
            ax[1].set_xlim(self.timens[peak_idx], 50)
            ax[1].set_ylim(-0.2,1)
                
    def get_max_ind_at_peak(self, img):
        '''
        Takes a 3D image and returns the i (row) and j (column)
        of the decay with the highest intensity at the peak. 
        '''
        print(f'''Max number of pixels *at peak* is {np.amax(self.imgPar)} 
              at pixels {np.where(self.imgPar == np.amax(self.imgPar))}
              ''')
            
        return np.where(img == np.amax(img))[1],np.where(img == np.amax(img))[2]
                
    def ROI_select_callback(self,verts):
        '''
        
        Callback of selectROI method. 
        Creates a boolean mask of same size as image with True values = 
        pixels within the the coordinates of the selected ROI. 
        
        
        https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        

        '''
        #print(verts)
        plt.close()
        t, dimx, dimy = self.imgPar.shape
        print(f'shape = {dimx, dimy}')
        
        verts = [(y,x) for x,y in verts]
        
        ROIpath = Path(verts)
        x, y = np.mgrid[:dimx, :dimy]
        coors = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        
        # creates essentially a mask
        # returns True for those indices that are inside the coords
        mask = ROIpath.contains_points(coors)
        self.ROImask = mask.reshape((dimx,dimy))
        
        #masked = np.ma.masked(self.img2D, self.ROImask)
        self.ROImask[np.where(self.ROImask)==0] = np.nan 
        
        # now build up cumulative decay for this ROI only
        # alter self par perp and r 
        
        self.decay_par, self.decay_perp, self.r = self.get_cum_decay_ROI()
        #self.r = np.divide((self.decay_par - G * self.decay_perp), ( self.decay_par + 2 * G * self.decay_perp))
        
        print("Done!")
        # now align peaks 
        self.decay_perp, self.decay_par, self.peak_idx, _ = align_peaks(self.decay_perp, 
                                                            self.decay_par, 
                                                            self.timens, 
                                                            plot = False, 
                                                            norm = False)
        
        


        
        # calculate lifetime 
        self.lifetime = self.get_lifetime()
        
        self.fig, self.ax = plt.subplots(2,2)
        self.ax[0,0].imshow(self.img2D, cmap = 'gray', interpolation='none')
        self.ax[0,0].imshow(self.ROImask, cmap = 'gist_heat', alpha = 0.6, interpolation='none')  # print including mask

        self.ax[0,1].plot(self.timens, self.decay_par, label = 'Par')
        self.ax[0,1].plot(self.timens, self.decay_perp, label = 'Perp')
        self.ax[0,1].set_xlabel("Time (ns)", fontsize = 14)
        self.ax[0,1].set_ylabel("Counts", fontsize = 14)
        self.ax[0,1].set_yscale("log")
        self.ax[0,1].legend(loc = 'upper right', fontsize = 14)
        self.ax[1,0].plot(self.timens[(self.peak_idx):], self.r[(self.peak_idx):])
        self.ax[1,0].set_xlabel("Time (ns)", fontsize = 14)
        self.ax[1,0].set_ylabel("Anisotropy", fontsize = 14)
        self.ax[1,1].plot(self.timens, self.lifetime)
        self.ax[1,1].set_xlabel("Time (ns)", fontsize = 14)
        self.ax[1,1].set_ylabel("Total counts", fontsize = 14)
        self.ax[1,1].set_yscale("log")
        plt.show()
        
    def selectROI(self): 
        # select coords with lines of image currently held 
 
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img2D, cmap='gray')
        self.ROI_selector = PolygonSelector(self.ax, 
                                            self.ROI_select_callback,
                                            lineprops = self.ROIprops)
        plt.show()
        #self.fig.canvas.mpl_connect('button_press_event', onPress)
        
    def show_ROI(self): 
        # 
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img2D, cmap = 'gray', interpolation='none')
        self.ax.imshow(self.ROImask, cmap = 'gist_heat', alpha = 0.6, interpolation='none')  # print including mask
        plt.show()
        
    def get_cum_decay_image(self):
         
        '''
        
        Using the current image (cropped or full image) add up the photons 
        from all the pixels to give a single set of par and perp decays 
        (summative over entire image) 
        
        Returns: 
            decay_par 
            decay_perp
            r 
        '''
        
        t,x,y = self.imgPar.shape
        # clear out decays in case any exist now 
        self.decay_par = np.array([])
        self.decay_perp = np.array([])
        
        for i in range(t): 
            self.decay_par = np.append(self.decay_par, self.imgPar[i].sum())
            self.decay_perp = np.append(self.decay_perp, self.imgPerp[i].sum())
        
        self.decay_perp, self.decay_par, self.peak_idx, _ = align_peaks(self.decay_perp, 
                                                            self.decay_par, 
                                                            self.timens, 
                                                            plot = False, 
                                                            norm = False)
        
        
        self.r = np.divide( (self.decay_par - self.G * self.decay_perp), ( self.decay_par + 2 * self.G * self.decay_perp))
        
        plt.figure(1)
        plt.imshow(self.img2D, cmap = 'gray', interpolation='none')
        plt.show()
        
        plt.figure(2)
        plt.plot(self.timens, self.decay_par, label = 'Par')
        plt.plot(self.timens, self.decay_perp, label = 'Perp')
        plt.xlabel("Time (ns)", fontsize = 14)
        plt.ylabel("Counts", fontsize = 14)
        plt.legend(loc = 'upper right', fontsize = 14)
        plt.yscale("log")
        plt.show()
        
        plt.figure(3)
        plt.plot(self.timens[self.peak_idx:], self.r[self.peak_idx:])
        #plt.plot(self.timens, self.decay_perp, label = 'Perp')
        plt.xlabel("Time (ns)", fontsize = 14)
        plt.ylabel("Anisotropy", fontsize = 14)
        plt.show()
        
        
        return self.decay_par, self.decay_perp, self.r
    
    def get_cum_decay_ROI(self):
        '''
        
        Using the current ROI selected, add up the photons from all the 
        pixels to give a single set of par and perp decays (summative over
                                                            entire ROI) 
        
        Returns: 
            decay_par 
            decay_perp 
            r 
        '''
        
        print("... ROI analysis in progress ..")
        time,x,y = self.imgPar.shape
        decay_par = np.zeros((time))
        decay_perp = np.zeros((time))
        count = 0 
        for t in range(time): 
            print(f"Currently at time bin {t} out of {time}.")
            for i in range(x):
                for j in range(y):
                    if self.ROImask[i,j] == False:
                        continue
                        #print(f'False mask at {i} x {j}')
                    else:
                        count += 1
                        #print(f'Count = {count}')
                        decay_par[t] += self.imgPar[t,i,j]
                        decay_perp[t] += self.imgPerp[t,i,j]
        
        r = np.divide( (decay_par - self.G * decay_perp), ( decay_par + 2 * self.G * decay_perp))
        
# =============================================================================
#         plt.figure(1)
#         plt.imshow(self.img2D, cmap = 'gray', interpolation='none')
#         plt.show()
#         
#         plt.figure(2)
#         plt.plot(self.timens, decay_par, label = 'Par')
#         plt.plot(self.timens, decay_perp, label = 'Perp')
#         plt.xlabel("Time (ns)", fontsize = 14)
#         plt.ylabel("Counts", fontsize = 14)
#         plt.legend(loc = 'upper right', fontsize = 14)
#         plt.yscale("log")
#         plt.show()
#         
# =============================================================================
        # calculates separate variables to assign later, does not alter self attributes  
        return decay_par, decay_perp, r    

    
    def get_photon_count_projection(self, img): 
        ##
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
        projected = projection[0, :, :]
    
        plt.imshow(projected, cmap = 'gray')
        #plt.pause(7)
        #plt.close()
        print(f'max photon count at peak is {np.amax(img)}')
        print(f'max photon count in projection is {np.amax(projected)}')
        return projected
    
    def threshold_img(self, img, thresh_value, keyword, save = False): 
        # img - which raw data matrix to be used 
        # intensity img will be created from given (binned or raw) data 
        # this will affect max number of photons 
        
        # create an intensity img 
        t,x,y = img.shape
        projection = np.empty_like(img)
    
        intensity = self.get_photon_count_projection(img)
        #plt.imshow(intensity, cmap = 'gray')
        
        binarised_img, binarised_mask = threshold_array(intensity, thresh_value, keyword, save = save)
        if save == True:  # if decided on a final mask 
            binarised_mask_3D = mask2Dto3D(binarised_mask, img)
            return binarised_mask_3D
        else: 
            return None
        
    def get_lifetime(self):
        return self.decay_par + 2*self.G*self.decay_perp
    
    def fit_single_ani_decay(self, r0_0, theta_0, rinf_0, model = "hindered"):
        
        '''
        Uses current peak_idx, decay_par, decay_perp 
        
        '''
        #peak_idx = np.argmax(self.decay_par)
        peak = self.peak_idx
        
        r = self.r[peak:]
        print(r.shape)
        t = self.timens[:(len(self.timens)-(peak))]
        print(t.shape)
        total = self.get_lifetime()[peak:]
        print(total.shape)
        
        plt.plot(self.timens, self.r)
        plt.axvline(self.timens[peak], color = 'r')
        plt.title("Chopped from peak+1")
        plt.pause(5)
        plt.close()
    
        # choose end of decay 
        plt.plot(t, r)
        plt.xlabel('Time(ns)')
        plt.ylabel('Anisotropy(r)')
        plt.ylim([-0.3, 0.5])
        plt.suptitle('Choose a final point of your decay (1 point)')
        boundary = plt.ginput(1)    # let user choose 
        limit_idx = find_nearest_neighbor_index(t,boundary[0])
        plt.close()    
        limit_idx = limit_idx[0]
        
        r = r[:limit_idx]
        t = t[:limit_idx]
        total = total[:limit_idx]
        
        plt.plot(t,r)
        plt.suptitle('Section of data that will be fit')
        plt.xlabel('Time (ns)', fontsize = 16)
        plt.ylabel('Anisotropy(r)', fontsize = 16)
        plt.ylim(-0.3, 0.5)
        plt.show()
        plt.pause(5)
        plt.close()
        
        w = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+self.G*(1+2*r)))/(3 * total))
        
        
        if model == "hindered": 
            model_func = lambda t, r0, rinf, theta: (r0-rinf)*np.exp(-t/theta) + rinf
            params = [r0_0, rinf_0, theta_0]
        elif model == "free": 
            model_func = lambda t, r0, theta: r0*np.exp(-t/theta)
            params = [r0_0, theta_0]
        else: 
            print("Model choice wrong, check documentation")
        
        
        error_func = lambda p, r, t, w: (r - model_func(t, p[0], p[1], p[2])) * w

        full_output = optimize.least_squares(error_func, 
                                               x0 = params,  # what to optimise 
                                               args = (r,t,w),  # independent variables
                                               method = 'lm',
                                               verbose = 2)
    
    
        # use jacobian to estimate cov. matrix and st error of each  
        Jac = full_output.jac
        cov = np.linalg.inv(Jac.T.dot(Jac))
        perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
            

        residuals = full_output.fun
        red_chi_sq = (full_output.fun**2).sum() / (len(r) - len(params))
        print(f''' 
              
          Success : {full_output.success}
          {full_output.message}
          function evaluations = {full_output.nfev}
          
          Optimum Params: 
          r0 = {full_output.x[0]:.2f} \u00B1 {perr[0]:.2f}, 
          rinf = {full_output.x[1]:.2f} \u00B1 {perr[1]:.2f}, 
          \u03F4 = {full_output.x[2]:.2f} \u00B1 {perr[2]:.2f}
          Red \u03C7 2 = {red_chi_sq:.2f}
          
          ''')
          
        r_fit = model_func(t, *full_output.x)
        
        plt.figure(1)
        plt.plot(t, r, label = 'raw decay')
        #plt.plot(t, r*w, label = 'smooth decay')
        plt.plot(t, r_fit, '-r', label = 'fit')
        #plt.plot(t, r, '-k', label = 'Sav Golay filter')
        plt.ylim(-0.3, 0.5)
        plt.xlim()
        plt.xlabel('Time (ns)', fontsize = 16)
        plt.ylabel('Anisotropy (r)', fontsize = 16)
        plt.legend(loc = 'upper right', fontsize = 14)
        plt.text(0.05, -0.25, fontsize = 14, 
                 bbox = dict(facecolor = "white", alpha = 0.2), 
                 s = f'''r(t) = ({full_output.x[0]:.2f}-{full_output.x[1]:.2f}) exp (-t/{full_output.x[2]:.1f}) - {full_output.x[1]:.2f}
                 Reduced \u03C7 2 = {red_chi_sq:.2f}''', )
        plt.show()
        #plt.pause(15)
        #plt.close()
        #plt.suptitle(title, fontsize = 16)
        
        plt.figure(2)
        plt.plot(t, residuals)
        plt.ylim(-3,3)
        plt.ylabel('residuals', fontsize = 20)
        
        return r_fit, full_output, perr, red_chi_sq
    
    def write_to_results_file(self, wd, file_name, sample_name, full_output, perr, red_chi_sq): 
        '''
        Takes the fits from fit_anisotropy_decay() 
        and write them to a text file in the current working directory. 
        If file with given name does not exist, it is created. 
        Otherwise results are appended  
        '''
        r0_string = f'{full_output.x[0]:.3f} \u00B1 {perr[0]:.3f}'
        rinf_string = f'{full_output.x[1]:.3f} \u00B1 {perr[1]:.3f}'
        theta_string = f'{full_output.x[2]:.3f} \u00B1 {perr[2]:.3f}'
        red_chi_sq_string = f'{red_chi_sq:.2f}'
                
        file_path = wd + '/' + file_name
        
        print(file_path)
        
        
        if not os.path.exists(file_path): 
                # if doesn't exist create table and column names 
                with open(file_name, "a") as f:
                    f.write('{0:20}  {1:21}  {2:21}   {3:21}    {4:5}\n'.format('sample', 'r0', 'rinf', 'theta', 'chi_squared'))
                    print(f"New results txt file created in folder {wd} ")
                    f.write('{0:20}  {1:21}  {2:21}   {3:21}    {4:5}\n'.format(sample_name, r0_string, rinf_string, theta_string, red_chi_sq_string))                    
        else:
            with open(file_name, "a") as f:
                f.write('{0:20}  {1:21}  {2:21}   {3:21}    {4:5}\n'.format(sample_name, r0_string, rinf_string, theta_string, red_chi_sq_string))
                print(f"Appending fits to existing file {file_name} in {wd}")
                            

    
    
    
     
    