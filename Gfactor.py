# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:34:25 2022

@author: Iveta
"""

'''
This class contains the functions necessary to interactively calculate the G factor from an anisotropy decay.

'''
from anisotropy_functions import get_gfactor, read_gfactor, process_sdt, plot_decays, background_subtract , align_peaks
import os
import numpy as np 

class Gfactor(): 
    def __init__(self, sample_dir): 
        self.sample_dir = sample_dir 
        self.par = None 
        self.perp = None 
        self.timens = None
        self.bins = None
        self.peak_idx = None 
        self.G_array = None
        self.G = None
        
    def process_gfactor(self): 
        '''
        Read sdt file 
        Process into numpy arrays 
        '''
        sdt_file= read_gfactor(self.sample_dir)
        self.perp, self.par, self.timens, self.bins = process_sdt(sdt_file)
        
    def plot_decays(self): 
        plot_decays(self.perp, "perp", self.par, "par", self.timens, scale = "log", norm = False, title = 'Rhodamine Gfactor')

    def background_subtract(self): 
        _, self.perp, self.par, _ = background_subtract(self.perp, self.par, self.timens)
    
    def align_peaks(self, plot = True): 
        self.perp, self.par, self.peak_idx, _ = align_peaks(self.perp, self.par, self.timens, plot = plot, scale = "log")
        
    def get_gfactor(self): 
        self.G_array = np.divide(self.par, self.perp)
        self.G = get_gfactor(self.G_array, self.timens, self.peak_idx)
        print(f"G factor value for this sample is: {self.G:.2f}")
        return self.G