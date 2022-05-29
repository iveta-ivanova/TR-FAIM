# TR-FAIM

This repository provides the minimum required code needed to perform fitting of Time-resolved fluorescence anisotropy (TR-FAIM) data. This includes 
* processing sdt anisotropy files 
* image segmentation using user-defined threshold
* pixel binning performed using kernel convolution 
* digital filter for smoothing the anisotropy decay
* G factor calculation
* fitting - single decays and pixel-by-pixel image fitting
* subsequent analysis and visualisation
* background subtraction (might be needed in case of very photon rich samples)
* ROI selection, cumulative decay creation from selected ROI
* basic statistics and exploratory functions 

This documentation describes the requirements, assumptions and conditions associated with using this code to perform this analysis. 
This documentation assumes knowledge of anisotropy principles, with microscopy image data and some familiarity with programming.

## Assumptions and requirements (physics): 

* The code performs a direct anisotropy fit on the anisotropy decay - fitting of parallel and perpendicular decays is not used here, which avoids iterative deconvolution using an IRF and hence simplifies the processing . 
. 
* The hindered rotation model is used for the fitting hence 3 parameters are estimated - r0, rinf and theta. Currently the software does not support the more popular free rotation model. 

## Assumptions and requirements (image data):

The code was tested only on image data obtained from TCSPC Becker & Hickl software and hence it was written in a way that support this data format. The output files are in sdt format and contain three channels:
* channel 1 - M1 TCSPC card - parallel decay 
* channel 2 - M2 TCSPC card - perpendicular decay (in the case of lifetime collection, this array will be empty)
* channel 3 - time channel 

## Assumptions and requirements (computational):
* the code was written in Python 3.8.8 (Spyder IDE) using Anaconda distribution (version 4.10.3)
* the code uses external Python packages already pre-packaged with Anaconda

The versions of some of the key packages used are: 
* numpy 1.20.1
* scipy 1.7.1
* matplotlib 3.3.3
* sdtfile 2021.3.21 - it also uses a ready package (https://pypi.org/project/sdtfile/) for reading sdt data which **has to be additionally installed** via `pip install sdt`    


The graphics backend on the IDE has to be set to _Automatic_, to make sure the interactive plots will pop up. In Spyder, this can be done from: 

Tools => Preferences => IPython console => Graphics => Graphics backend => Automatic 

![image](https://user-images.githubusercontent.com/52123994/170880354-77aca1c1-a8b1-4393-8e43-c035b391c1c3.png)
