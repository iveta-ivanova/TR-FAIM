# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:02:17 2022

@author: Iveta


This file contains sveral functions for image fitting of lifetime and anisotropy data. 
These include NLLS fitting using Levenberg-Marquardt, as well as optimisation 
using other minimizers. 

"""


import numpy as np
from scipy.signal import savgol_filter
from anisotropy_functions import percent_in_bounds, align_peaks, align_image ,find_nearest_neighbor_index
import scipy.optimize
from scipy.optimize import minimize,differential_evolution, dual_annealing,shgo
import matplotlib.pyplot as plt 
#import time
#import bigfloat

#bigfloat.exp(5000,bigfloat.precision(100))  # overcome exp overfow due to flow restrictions 

np.seterr(divide='ignore', invalid='ignore')   # ignore divide by 0 warnings 
#np.warnings.filterwarnings('ignore', '(overflow|invalid)')

def model_NLLS_lifetime_fit_no_decon(perpimage, parimage, time, G, mask, A_0, tau1_0, tau2_0):
    _,a,b = perpimage.shape
    p10,p20, _ = A_0, tau1_0, tau2_0
    
    # entire image - align 
    perpimage, parimage, peaks = align_image(perpimage, parimage, time)
    # calculate lifetime image 
    #totalimage = np.empty_like(perpimage)
    totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    peak_int = []
    
    #totalimage = np.add(parimage[:], (2.0*G*perpimage[:]))
        
    #3D images to hold the decays and fits 
    lifetime_fit = np.empty_like(totalimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float) 
    #par_conv_data = np.empty_like(perpimage, dtype = float)
    #2D images to hold the parameter maps 
    A_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #B_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    tau1_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #tau2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #tau_mean_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    tau1_img_err = np.empty((perpimage.shape[1], perpimage.shape[2]))
    A_img_err = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 
    #2D images to hold the errors 
    #r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    log_dict = {}

    
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                lifetime_fit[:,i,j] = np.nan
                res_img[:,i,j] = np.nan 
                
                A_img[i,j] = np.nan
                tau1_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                tau1_img_err[i,j] = np.nan
                A_img_err[i,j] = np.nan
                #tau2_img[i,j] = np.nan
                #tau_mean_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]
                
                tot = totalimage[:,i,j]
                
                peak_int.append(np.amax(tot))
                                                                                
                tot = np.where(tot < 1, 1, tot)
                #weights = 1/np.sqrt(tot)
                
                peak_idx = np.argmax(tot)
                    
                y = tot[peak_idx:]
                t = time[:(len(time)-peak_idx)]
                #w = weights
                #y = tot_conv[peak_idx:]
                #y = tot[peak_idx:]
                #total = total[peak_idx:]
                #t = time[peak_idx:]
                #weights = weights[peak_idx:]
                                
                '''
                p1 - r0 
                p2 - theta 
                p3 - rinf 
                '''
                #peak = np.argmax(y)
                
                #conv = lambda t, p1, p2: fft(IRF, p1 * np.exp(-t/p2), mode = 'full')[10:(len(t)+10)]
                model = lambda t, p1, p2: p1 * np.exp(-t/p2)

                # initial guesses 
                
                #plt.cla()
                #plt.plot(t, conv(t, p10, p20))
                #plt.plot(t, y)
                #plt.title('initial guesses')
                #plt.pause(5)
                #plt.close()
                #p = p10, p20
                
                
                def error_func(p, t, y): ## Pearson 
                    w = 1/np.sqrt(model(t, p[0], p[1]))[peak_idx:]
                    #print(f'in obj function at px {a} x {b}: {len(y)}, {len(t)}, {len(w)}')

                    err = ((y - model(t, p[0], p[1])[peak_idx:]) * w)**2
                    # error is a vector of values (the residuals)
                    # log the sum of squares of residuals
                    log_dict.setdefault(f'{i}x{j}', []).append(np.sum(err))
                    return err                               
                #error_func = lambda p, t, y, w: ((y - model_func(t, p[0], p[1], p[2])) * w)
                                                
                #full_output = scipy.optimize.leastsq(error_func , x0 = [p10, p20], 
                #                                     args = (t, total, y, w), 
                #                                     full_output = True)
                #bounds = ([0,0,-0.5],[1,100,1])
                result = scipy.optimize.least_squares(error_func,
                                               x0 = [p10, p20],
                                               args = (t, y),
                                               method = 'lm',
                                               verbose = 0)
                                                        
                A, tau = result.x[0], result.x[1]

                residuals = result.fun 
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)

                if result.success is True:                 
                    successful_conv += 1
                    
                #res = error_func(result.x, t, y, w)/(len(y)-len([p10, p20]))
                                
                #par_conv_data[peak_idx:,i,j] = y 
                fit = model(t, A, tau)
                
                
                lifetime_fit[peak_idx:,i,j] = fit
                
                # Pearson's chisq - uncertainty is 1/fit 
                w_fit = 1/np.sqrt(fit)
                red_chi_sq = (np.sum(((y-fit)*w_fit)**2))/(len(y)-len(result.x))
                
                chi2_img[i,j] = red_chi_sq
                
                A_img[i,j] = A   # r0  - 2d image 
                #B_img[i,j] = B
                tau1_img[i,j] = tau  # tau - 2d image    
                #tau2_img[i,j] = tau2
                res_img[peak_idx:,i,j] = residuals                
                
                if total_attempts < 50: 
                    plt.figure(1)
                    plt.cla()
                    plt.plot(t, y, label = 'raw')
                    plt.plot(t, fit, label = 'fit')
                    plt.title(f''' tau = {tau:.2f} with x2 = {red_chi_sq:.2f}''')
                    plt.pause(0.01)
                    
                    plt.figure(2)
                    plt.cla()
                    plt.plot(t, residuals)
                    plt.ylabel('residuals')
                    plt.pause(0.01)
                else: 
                    plt.close()
                
                # use jacobian to estimate cov. matrix 
                Jac = result.jac
                cov = np.linalg.inv(Jac.T.dot(Jac))
                perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                
                
                A_img_err[i,j] = perr[0]
                tau1_img_err[i,j] = perr[1]
                
    plt.figure(3)
    plt.imshow(tau1_img, clim = (2,5))
    plt.colorbar()
    plt.show()
    plt.figure(4)
    plt.hist(tau1_img.ravel(), bins = 50)
    plt.title("Tau")
    plt.show()
    plt.figure(5)
    plt.imshow(tau1_img_err)
    plt.title("Tau St dev")
    plt.colorbar()
    plt.show()
    plt.figure(6)
    plt.hist(tau1_img_err.ravel(), bins = 50)
    plt.title("St. deviation")
    plt.show()
    
    print(total_attempts)
    print(successful_conv)
    print(f'Success rate of convergence: {(successful_conv / total_attempts)*100} %')
                
    return A_img, A_img_err, tau1_img, tau1_img_err, lifetime_fit, chi2_img, res_img, np.array(peak_int), log_dict


np.seterr(divide='ignore', invalid='ignore')   # ignore divide by 0 warnings 
#np.warnings.filterwarnings('ignore', '(overflow|invalid)')

def model_curvefit_anisotropy_parallel_fit_no_decon(perpimage, parimage, time, G, mask, r0_0, rinf_0, theta_0):
    _,a,b = perpimage.shape
    p10,p20, p30 = r0_0, rinf_0, theta_0
    
    totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    peak_int = []
            
    #3D images to hold the decays and fits 
    parallel_fit = np.empty_like(totalimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float) 
    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 
    
    params = Parameters()
    params.add("r0", p10, max = 0.6, min = 0.1)
    params.add("rinf", p20, max = 0.6, min = -0.2)
    params.add("theta", p30, max = 20.0, min = 0.1)
    
    #2D images to hold the errors 
    #r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
     
    img_check_par_pep = np.empty((perpimage.shape[1], perpimage.shape[2]))       

    log_dict = {}       
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                parallel_fit[:,i,j] = np.nan
                res_img[:,i,j] = np.nan 
                
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
                r0_err_img[i,j] = np.nan
                rinf_err_img[i,j] = np.nan
                theta_err_img[i,j] = np.nan
                
                img_check_par_pep[i,j] = np.nan
                #tau2_img[i,j] = np.nan
                #tau_mean_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]

                peak_int.append(np.amax(par))
                              
                par, tot, peak, _ = align_peaks(par, tot, time, plot = False)

                #print(f'Total, IRF peak: {peak}')
                #print(f"total argmax = {np.argmax(tot)}")
                #print(f"IRF argmax = {np.argmax(IRF)}")
                #print(type(IRF))
                #print(IRF[0:20])
                                                
                tot = np.where(tot < 1, 1, tot)
                par = np.where(par < 1, 1, par)
                weights = 1/np.sqrt(par)
                    
                y = par
                t = time
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                #peak = np.argmax(y)
                
                #plt.plot(t, model(t, p10,p20,p30) )
                # non-normalised 
                #conv = lambda t, p1, p2, p3: fft(IRF_shifted, ((1/3)*tot*(p1-p2)*np.exp(-t/p3)-p2), mode = 'full')[:len(t)]
                #model = lambda t, p1,p2,p3: (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2
                                                                                                       
                #conv = lambda t, p1, p2, p3: fft(IRF_shifted, (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2, mode = 'full')[:len(t)] * (max(y)/max(fft(IRF_shifted, ((1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2), mode = 'full')[:len(t)]))
                
                def model_function(t,r0, rinf, theta):
                    model = (1/3)*tot*(r0-rinf)*np.exp(-t/theta)+rinf
                    log_dict.setdefault(f'{i}x{j}', []).append(np.sum((y-model)**2))
                    return model 
                #p = p10, p20, p30

                # initial guesses 
                
                #initial = model_function(t,p)
                #print(initial.shape)
                
                #plt.cla()
# =============================================================================
#                 plt.figure(2)
#                 plt.plot(t, model_function(t,p), label = "IRF*model")
#                 #plt.plot(t, model(t, p), label = "model" )
#                 plt.plot(t, y, label = "raw")
#                 plt.title(f'params r0={p10}, rinf={p20}, theta ={p30}')
#                 plt.legend(loc = "upper right")
#                 plt.pause(10)
# =============================================================================
                #plt.close()
                #p = p10, p20
                
                #def error_func(p, t, y, w): 
                #    return (conv(t, p[0], p[1],p[2]) - y) * w    
                            
                #error_func = lambda p, t, y, w: ((y - model_func(t, p[0], p[1], p[2])) * w)
                                                
                #full_output = scipy.optimize.leastsq(error_func , x0 = [p10, p20], 
                #                                     args = (t, total, y, w), 
                #                                     full_output = True)
                #bounds = ([0,0,-0.5],[1,100,1])
# =============================================================================
#                 result = scipy.optimize.least_squares(error_func,
#                                                x0 = [p10, p20, p30],
#                                                args = (t, y, w),
#                                                method = 'lm',
#                                                verbose = 0)
#                 
# =============================================================================

                
                #def model_fitting(x, r0_0, rinf_0, theta_0): 
                #    return conv(x, r0_0, rinf_0, theta_0)
                
                modelling = Model(model_function)
                
                result = modelling.fit(y, t=t, params = params, weights = w)
                
                r0, rinf, theta = result.best_values['r0'],result.best_values['rinf'], result.best_values['theta'] 
                
                
                fit = model_function(t, r0, rinf, theta)
                #fit = result.fit

                residuals = result.residual
                
                #red_chi_sq = (np.sum((((y - fit)**2)*w)/y))/(len(y)-len(result.best_values))

                #print(result.message)
                #print(result.optimality)
                #print(result.x)
                
                parallel_fit[:,i,j] = fit
                
                #red_chi_sq = (np.sum((((y - fit)**2)*w)/y))/(len(y)-len(result.best_values))
                red_chi_sq = result.redchi
                chi2_img[i,j] = red_chi_sq
                
                r0_img[i,j] = r0   # r0  - 2d image 
                #B_img[i,j] = B
                rinf_img[i,j] = rinf  # tau - 2d image    
                theta_img[i,j] = theta
                res_img[:,i,j] = residuals 
                
                print(type(fit))
                print(fit.shape)
                print(f"""Estimated results 
                      r0 = {r0}
                      rinf = {rinf}
                      theta = {theta}
                      chi_squared = {result.chisqr}
                      red chi sq = {result.redchi} """)
                
                if total_attempts < 30:
                    plt.figure(1)
                    plt.cla()
                    plt.plot(t, y, label = 'raw')
                    plt.plot(t, fit, label = 'fit')
                    plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                              theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                    plt.pause(0.01)
                    
                    plt.figure(2)
                    plt.cla()
                    plt.plot(t, residuals)
                    plt.ylabel('residuals')
                    plt.pause(0.01)

                #cov = result.covar
                #cov = np.linalg.inv(Jac.T.dot(Jac))
                #perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                
                #r0_err_img[i,j] = perr[0]
                #r0_err_img[i,j] = perr[1]
                #r0_err_img[i,j] = perr[1]
                
    plt.figure(3)
    plt.imshow(rinf_img)
    plt.title("Rinf map")
    plt.colorbar()
    plt.show()
    
    plt.figure(4)
    plt.hist(rinf_img.ravel(), bins = 50)
    plt.title("Rinf distribution")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img)
    plt.title("Theta map")
    plt.colorbar()
    plt.show()
    
    plt.figure(6)
    plt.hist(theta_img.ravel())
    plt.title("Theta distribution")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img)
    plt.title("r0 map")
    plt.colorbar()
    plt.show()
    
    plt.figure(8)
    plt.hist(r0_img.ravel(), bins = 50)
    plt.title("r0 distribution")
    plt.show()
    
 
    print(f'Success rate of convergence: {(successful_conv / total_attempts)*100} %')
    
    return r0_img, r0_err_img, rinf_img, rinf_err_img, theta_img, theta_err_img, parallel_fit, chi2_img, res_img, np.array(peak_int), log_dict


def model_minimization_anisotropy_parallel_fit_no_decon(perpimage, parimage, time, G, mask, r0_0, rinf_0, theta_0):
    _,a,b = perpimage.shape
    p10,p20, p30 = r0_0, rinf_0, theta_0
    
    totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    peak_int = []
            
    #3D images to hold the decays and fits 
    parallel_fit = np.empty_like(totalimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float) 
    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 
    
    params = Parameters()
    params.add("r0", p10, max = 0.6, min = 0.1)
    params.add("rinf", p20, max = 0.6, min = -0.2)
    params.add("theta", p30, max = 20.0, min = 0.1)
    
    #2D images to hold the errors 
    #r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    log_dict = {}
    img_check_par_pep = np.empty((perpimage.shape[1], perpimage.shape[2]))              
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                parallel_fit[:,i,j] = np.nan
                res_img[:,i,j] = np.nan 
                
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
                r0_err_img[i,j] = np.nan
                rinf_err_img[i,j] = np.nan
                theta_err_img[i,j] = np.nan
                
                img_check_par_pep[i,j] = np.nan
                #tau2_img[i,j] = np.nan
                #tau_mean_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]

                peak_int.append(np.amax(par))
                              
                par, tot, peak, _ = align_peaks(par, tot, time, plot = False)

                #print(f'Total, IRF peak: {peak}')
                #print(f"total argmax = {np.argmax(tot)}")
                #print(f"IRF argmax = {np.argmax(IRF)}")
                #print(type(IRF))
                #print(IRF[0:20])
                                                
                tot = np.where(tot < 1, 1, tot)
                par = np.where(par < 1, 1, par)
                weights = 1/np.sqrt(par)
                    
                y = par
                t = time
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                #peak = np.argmax(y)
                
                #plt.plot(t, model(t, p10,p20,p30) )
                # non-normalised 
                #conv = lambda t, p1, p2, p3: fft(IRF_shifted, ((1/3)*tot*(p1-p2)*np.exp(-t/p3)-p2), mode = 'full')[:len(t)]
                #model = lambda t, p1,p2,p3: (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2
                                                                                                       
                #conv = lambda t, p1, p2, p3: fft(IRF_shifted, (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2, mode = 'full')[:len(t)] * (max(y)/max(fft(IRF_shifted, ((1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2), mode = 'full')[:len(t)]))
                
                model = lambda t, p1, p2, p3: ((1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2)
                
                def objective_function(x, t):
                    
                    guess = model(t,x[1], x[2], x[3])                    
                                        
                    obj = 1/len(y) * (((y-guess)*w)**2)   # RMS 
                    # obj = norm(((y-guess)*w),1)         # norm 
                    # obj = ? 
                    log_dict.setdefault(f'{i}x{j}', []).append(obj)

                    return obj 
                
                bounds = [(0, 1),(-0.5, 0.5),(0,100)]

                #method = "Powell"
                method = "Nelder-Mead"
                
                result = minimize(objective_function, x0 = [p10,p20,p30], 
                                  method = method, 
                                  args = (t), 
                                  bounds = bounds, 
                                  options = {'disp': True})
                
                #red_chi_sq = (np.sum((((y - fit)**2)*w)/y))/(len(y)-len(result.best_values))

                #print(result.message)
                #print(result.optimality)
                #print(result.x)
                
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                
                fit = model(t,r0, rinf, theta)
                parallel_fit[:,i,j] = fit
                
                residuals = (y-fit)*w
                #red_chi_sq = (np.sum((((y - fit)*w)**2))/(len(y)-len(result.best_values))
                red_chi_sq = (np.sum(((y-fit)*w)**2))/(len(y)-len(result.x))
                chi2_img[i,j] = red_chi_sq
                
                r0_img[i,j] = r0   # r0  - 2d image 
                #B_img[i,j] = B
                rinf_img[i,j] = rinf  # tau - 2d image    
                theta_img[i,j] = theta
                res_img[:,i,j] = residuals
                
                print(type(fit))
                print(fit.shape)
                print(f"""Estimated results 
                      r0 = {r0}
                      rinf = {rinf}
                      theta = {theta}
                      red chi sq = {red_chi_sq} """)
                
                if total_attempts < 50:
                    plt.figure(1)
                    plt.cla()
                    plt.plot(t, y, label = 'raw')
                    plt.plot(t, fit, label = 'fit')
                    plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                              theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                    plt.pause(0.01)
                    
                    plt.figure(2)
                    plt.cla()
                    plt.plot(t, residuals)
                    plt.ylabel('residuals')
                    plt.pause(0.01)

                #cov = result.covar
                #cov = np.linalg.inv(Jac.T.dot(Jac))
                #perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                
                #Jac = result.jac
                #cov = np.linalg.inv(Jac.T.dot(Jac))
                #perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                
                #r0_err_img[i,j] = perr[0]
                #r0_err_img[i,j] = perr[1]
                #r0_err_img[i,j] = perr[1]
                
    plt.figure(3)
    plt.imshow(rinf_img)
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    plt.figure(4)
    plt.hist(rinf_img.ravel(), bins = 50)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img)
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    plt.figure(6)
    plt.hist(theta_img.ravel(), bins = 50)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img)
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    plt.figure(8)
    plt.hist(r0_img.ravel(), bins = 50)
    plt.title("r0")
    plt.show()
         
    return r0_img, r0_err_img, rinf_img, rinf_err_img, theta_img, theta_err_img, parallel_fit, chi2_img, res_img, log_dict

def model_MLE_anisotropy_parallel_fit_no_decon(perpimage, parimage, time, G, mask, r0_0, rinf_0, theta_0):
    _,a,b = perpimage.shape
    p10,p20, p30 = r0_0, rinf_0, theta_0
    
    totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    peak_int = []
            
    #3D images to hold the decays and fits 
    parallel_fit = np.empty_like(totalimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float) 
    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 
    
    #2D images to hold the errors 
    #r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    log_dict = {}
    
              
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                parallel_fit[:,i,j] = np.nan
                res_img[:,i,j] = np.nan 
                
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
                r0_err_img[i,j] = np.nan
                rinf_err_img[i,j] = np.nan
                theta_err_img[i,j] = np.nan
                                #tau2_img[i,j] = np.nan
                #tau_mean_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]

                peak_int.append(np.amax(par))
                              
                par, tot, peak, _ = align_peaks(par, tot, time, plot = False)

                #print(f'Total, IRF peak: {peak}')
                #print(f"total argmax = {np.argmax(tot)}")
                #print(f"IRF argmax = {np.argmax(IRF)}")
                #print(type(IRF))
                #print(IRF[0:20])
                                                
                tot = np.where(tot < 1, 1, tot)
                par = np.where(par < 1, 1, par)
                weights = 1/np.sqrt(par)
                    
                y = par
                t = time
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                #peak = np.argmax(y)
                
                #plt.plot(t, model(t, p10,p20,p30) )
                # non-normalised 
                #conv = lambda t, p1, p2, p3: fft(IRF_shifted, ((1/3)*tot*(p1-p2)*np.exp(-t/p3)-p2), mode = 'full')[:len(t)]
                #model = lambda t, p1,p2,p3: (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2
                                                                                                       
                #conv = lambda t, p1, p2, p3: fft(IRF_shifted, (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2, mode = 'full')[:len(t)] * (max(y)/max(fft(IRF_shifted, ((1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2), mode = 'full')[:len(t)]))
                
                model = lambda t, p1, p2, p3: (1/3)*tot*(p1-p2)*np.exp(-t/p3)+p2
                
                def objective_function(x, t, tot, w):
                    
                    #conv = conv*(max(y)/max(conv))
                    guess = model(t,x[0],x[1],x[2])
                    
                    # Poisson distribution                           
                    negLL = 2*(np.sum(guess - y)) + 2*(np.sum((y*np.log(guess/y))))
                    
                    # chisquared distribution, multinomial  - NOT Poisson  
                    negLL = 2*np.sum*((y*np.log(y/guess)))
                    
                    log_dict.setdefault(f'{i}x{j}', []).append(abs(negLL))
                    

                    return abs(negLL)
                
                #bounds = ([0,-0.5,0],[1,1,100])

                bounds = ((0, 1),(-0.5, 0.5),(0,100))

                result = minimize(objective_function, x0 = [p10,p20,p30], 
                                  method = "Nelder-Mead", 
                                  args = (t, tot, w), 
                                  bounds = bounds, 
                                  options = {'disp': False})
                
                #red_chi_sq = (np.sum((((y - fit)**2)*w)/y))/(len(y)-len(result.best_values))

                #print(result.message)
                #print(result.optimality)
                #print(result.x)
                
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                
                fit = model(t,r0, rinf, theta)
                parallel_fit[:,i,j] = fit
                
                residuals = (y-fit)*w
                #logresiduals = 
                #red_chi_sq = (np.sum((((y - fit)*w)**2))/(len(y)-len(result.best_values))
                red_chi_sq = (np.sum(((y-fit)*w)**2))/(len(y)-len(result.x))
                chi2_img[i,j] = red_chi_sq
                
                r0_img[i,j] = r0   # r0  - 2d image 
                #B_img[i,j] = B
                rinf_img[i,j] = rinf  # tau - 2d image    
                theta_img[i,j] = theta
                res_img[:,i,j] = residuals
                
                #print(type(fit))
                #print(fit.shape)
                print(f"""Estimated results 
                      r0 = {r0}
                      rinf = {rinf}
                      theta = {theta}
                      red chi sq = {red_chi_sq} """)
                
                if total_attempts < 50:
                    plt.figure(1)
                    plt.cla()
                    plt.plot(t, y, label = 'raw')
                    plt.plot(t, fit, label = 'fit')
                    plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                              theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                    plt.pause(0.01)
                    
                    plt.figure(2)
                    plt.cla()
                    plt.plot(t, residuals)
                    plt.ylabel('residuals')
                    plt.pause(0.01)

                #cov = result.covar
                #cov = np.linalg.inv(Jac.T.dot(Jac))
                #perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                
                #Jac = result.jac
                #cov = np.linalg.inv(Jac.T.dot(Jac))
                #perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                
                #r0_err_img[i,j] = perr[0]
                #r0_err_img[i,j] = perr[1]
                #r0_err_img[i,j] = perr[1]
                
    plt.figure(3)
    plt.imshow(rinf_img)
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    plt.figure(4)
    plt.hist(rinf_img.ravel(), bins = 50)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img)
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    plt.figure(6)
    plt.hist(theta_img.ravel(), bins = 50)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img)
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    plt.figure(8)
    plt.hist(r0_img.ravel(), bins = 50)
    plt.title("r0")
    plt.show()
    
    return r0_img, r0_err_img, rinf_img, rinf_err_img, theta_img, theta_err_img, parallel_fit, chi2_img, res_img, log_dict

def model_NLLS_anisotropy_fit(perpimage, parimage, time, G, mask, r0_0, rinf_0, theta_0, smooth = False):
    '''
    Direct fit of the anisotroyp decay. 
    Binning essential ! 
    '''
    _,a,b = perpimage.shape
    p10,p20, p30 = r0_0, rinf_0, theta_0 
    
    #totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    peak_int = []
            
    #3D images to hold the decays and fits 
    anisotropy_fit = np.empty_like(perpimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float)   # residuals 
    ani_smooth = np.empty_like(perpimage, dtype = float)
    raw_decay = np.empty_like(perpimage, dtype = float)

    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 

    log_dict = {}
    
    limit_idx = None
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                anisotropy_fit[:,i,j] = np.nan
                res_img[:,i,j] = np.nan 
                ani_smooth[:,i,j] = np.nan
                raw_decay[:,i,j] = np.nan

                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan

                r0_err_img[i,j] = np.nan
                rinf_err_img[i,j] = np.nan
                theta_err_img[i,j] = np.nan
                                #tau2_img[i,j] = np.nan
                #tau_mean_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                #tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]
                
                perp = perpimage[:,i,j]
                              
                perp, par, peak, _ = align_peaks(perp, par, time, plot = False)
                
                peak_int.append(np.amax(par))
                                                
                tot = par + 2 * G * perp
                
                r = (par - G * perp) / tot
                
                if limit_idx is None:
                    plt.plot(time, r)
                    plt.title("Choose single point as limit")
                    plt.axvline(x=time[peak],color = 'r')
                    limit = plt.ginput(1)
                    limit_idx = find_nearest_neighbor_index(time[peak:], limit[0])
                    limit_idx = limit_idx[0]
                    limit_idx = peak + limit_idx
                    plt.close()
                    plt.plot(time[:(limit_idx-peak)],r[peak:limit_idx])
                    plt.title('Part that will be fit')
                    plt.pause(5)
                    plt.close()
                r = r[peak:limit_idx]
                raw_decay[peak:limit_idx,i,j] = r
                             
                window_size = 15
                polynomial= 2
                # smooth 
                
                # implement buttons to choose suitable and move on 
                # s should have same length as r - peak to limit chosen 
                if smooth:
                    try: 
                        s = savgol_filter(r, window_size, polynomial)
                        if total_attempts < 3:
                            plt.plot(time[peak:limit_idx], s, 'r')
                            plt.plot(time[peak:limit_idx], r)
                            #plt.axvline(time, time[peak], 'r')
                            plt.title(f"Smoothed decay with window {window_size} and poly {polynomial}")
                            plt.show()
                            plt.pause(7)
                            plt.close()
                    
                        ani_smooth[peak:limit_idx,i,j] = s 
                            
                        weights = 1/np.sqrt(((1-s)*(1 + 2*s)*(1-s+G*(1+2*s)))/(3 * tot[peak:limit_idx]))
                    
                    except: 
                        weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
                        ani_smooth[peak:limit_idx,i,j] = np.nan 

                    
                else:
                    weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
            
                y = r
                t = time[:(limit_idx-peak)]
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                
                model = lambda t, p1, p2, p3: (p1-p2)*np.exp(-t/p3)+p2
                                
                def error_func(x,t,w):
                    err = (y - model(t,x[0],x[1],x[2]))*w
                                                                  
                    log_dict.setdefault(f'{i}x{j}', []).append(np.sum(err**2))

                    # for least squares - err function is the residuals 
                    return err
                
                #bounds = ([0,-0.5,0],[1,1,100])

                #bounds = ((0, 1),(-0.5, 0.5),(0,100))
                
                method = "lm"

                try:
                    result = scipy.optimize.least_squares(error_func,
                                                   x0 = [p10, p20, p30],
                                                   args = (t,w),
                                                   method = method,
                                                   verbose = 0)
                except ValueError as ve:
                    ## error, continue on to next pixel
                    
                    print('Value error, pixel fit aborted')
                    print(ve)
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
                    ani_smooth[:,i,j] = np.nan
                    raw_decay[:,i,j] = np.nan
    
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
    
                    r0_err_img[i,j] = np.nan
                    rinf_err_img[i,j] = np.nan
                    theta_err_img[i,j] = np.nan
                    
                    continue
                    
                                                                    
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]

                residuals = result.fun 
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)

                #if result.success is True:                 
                #    successful_conv += 1
                    
                #res = error_func(result.x, t, y, w)/(len(y)-len([p10, p20]))
                                
                #par_conv_data[peak_idx:,i,j] = y 
                fit = model(t, r0, rinf, theta)
                
                anisotropy_fit[peak:limit_idx,i,j] = fit
                
                red_chi_sq = np.sum(((y-fit)*w)**2)/(len(y)-len(result.x))
                
                chi2_img[i,j] = red_chi_sq
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)
                
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                
                r0_img[i,j] = r0   # r0  - 2d image 
                #B_img[i,j] = B
                rinf_img[i,j] = rinf  # tau - 2d image    
                theta_img[i,j] = theta
                res_img[peak:limit_idx,i,j] = residuals

                    
                if total_attempts < 100:
                    plt.figure(2)
                    plt.cla()
                    plt.plot(t, y, label = 'raw')
                    plt.plot(t, fit, 'r', label = 'fit')
                    if smooth: 
                        plt.plot(t, s, 'm', label = 'smoothed')
                    plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                              theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                    plt.pause(0.01)
                    
                    plt.figure(3)
                    plt.cla()
                    plt.plot(t, residuals)
                    plt.ylabel('residuals')
                    plt.pause(0.01)
                else: 
                    plt.close()

                #cov = result.covar
                #cov = np.linalg.inv(Jac.T.dot(Jac))
                #perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                                
                try:
                    Jac = result.jac
                    cov = np.linalg.inv(Jac.T.dot(Jac))
                    perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                    
                    r0_err_img[i,j] = perr[0]
                    rinf_err_img[i,j] = perr[1]
                    theta_err_img[i,j] = perr[1]
                    successful_conv += 1
                except:
                    print("convergence not reached")
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
                    ani_smooth[:,i,j] = np.nan
                    raw_decay[:,i,j] = np.nan
    
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
    
                    r0_err_img[i,j] = np.nan
                    rinf_err_img[i,j] = np.nan
                    theta_err_img[i,j] = np.nan
                    
                    r0_ib = np.nan 
                    rinf_ib = np.nan 
                    theta_ib = np.nan
                    chi2_ib = np.nan
                    
                          
                print(f"""Estimated results:
                          r0 = {r0}
                          rinf = {rinf}
                          theta = {theta}
                          red chi sq = {red_chi_sq} 
                      
                     """)
                
    print(f'% successfuly convergence = {(successful_conv/total_attempts)*100}')
    
    
    _, r0_ib =  percent_in_bounds(r0_img, [-0.3, 0.4])
    _, rinf_ib = percent_in_bounds(rinf_img, [-0.5, 0.3])
    _, theta_ib = percent_in_bounds(theta_img, [1,6])
    _, chi2_ib = percent_in_bounds(chi2_img, [-1.5, 1.5])
                
    print(f'''  Percent in bounds: 
                          r0 =  {r0_ib}
                          rinf = {rinf_ib}
                          theta = {theta_ib}
                          chi2 = {chi2_ib} ''')
    
    plt.figure(3)
    plt.imshow(rinf_img)
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    plt.figure(4)
    plt.hist(rinf_img.ravel(), bins = 50)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img)
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    plt.figure(6)
    plt.hist(theta_img.ravel(), bins = 50)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img)
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    plt.figure(8)
    plt.hist(r0_img.ravel(), bins = 50)
    plt.title("r0")
    plt.show()
    
    return raw_decay, peak_int, r0_img, r0_err_img, rinf_img, rinf_err_img, theta_img, theta_err_img, anisotropy_fit, chi2_img, res_img, log_dict

def model_minization_anisotropy_fit(perpimage, parimage, time, G, mask, r0_0, rinf_0, theta_0, method, smooth = False):
    '''
    Direct fit of the anisotroyp decay. 
    Binning essential ! 
    '''
    _,a,b = perpimage.shape
    p10,p20, p30 = r0_0, rinf_0, theta_0
    
    #totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    #peak_int = []
    
    perpimage, parimage, _ = align_image(perpimage,parimage, time)
    print('images aligned')
    totimg = parimage + 2 * G *perpimage
    print('total img created')

    rimg = (parimage - G * perpimage)/totimg
    print('r image created')

            
    #3D images to hold the decays and fits 
    anisotropy_fit = np.empty_like(perpimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float)   # residuals 
    
    #ani_smooth = np.empty_like(perpimage, dtype = float).nan
    #raw_decay = np.empty_like(perpimage, dtype = float).nan

    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 

    #obj_function = np.empty((perpimage.shape[1], perpimage.shape[2]))
    log_dict = {}
    
    limit_idx = None
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #anisotropy_fit[:,i,j] = np.nan
                #res_img[:,i,j] = np.nan 
                #ani_smooth[:,i,j] = np.nan
                #raw_decay[:,i,j] = np.nan
    
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                #tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]
                
                #perp = perpimage[:,i,j]
                
                tot = totimg[:,i,j]
                
                r = rimg[:,i,j]
                              
                #perp, par, peak, _ = align_peaks(perp, par, time, plot = False)
                
                peak = np.argmax(par)
                
                #peak_int.append(np.amax(par))
                                                
                #tot = par + 2 * G * perp
                
                #r = (par - G * perp) / tot
                
                if limit_idx is None:
                    plt.plot(time, r)
                    plt.title("Choose single point as limit")
                    plt.axvline(x=time[peak],color = 'r')
                    limit = plt.ginput(1)
                    limit_idx = find_nearest_neighbor_index(time[peak:], limit[0])
                    limit_idx = limit_idx[0]
                    limit_idx = peak + limit_idx
                    plt.close()
                    plt.plot(time[:(limit_idx-peak)],r[peak:limit_idx])
                    plt.title('Part that will be fit')
                    plt.pause(5)
                    plt.close()
                r = r[peak:limit_idx]
                
                #print(f'Limiting index here is: {limit_idx}')
                #raw_decay[peak:limit_idx,i,j] = r
                             
                window_size = 15
                polynomial= 2
                # smooth 
                
                # implement buttons to choose suitable and move on 
                # s should have same length as r - peak to limit chosen 
                if smooth:
                    try: 
                        s = savgol_filter(r, window_size, polynomial)
                        if total_attempts < 3:
                            plt.plot(time[peak:limit_idx], s, 'b')
                            plt.plot(time[peak:limit_idx], r)
                            #plt.axvline(time, time[peak], 'r')
                            plt.title(f"Smoothed decay with window {window_size} and poly {polynomial}")
                            plt.show()
                            plt.pause(5)
                            plt.close()
                    
                        #ani_smooth[peak:limit_idx,i,j] = s 
                            
                        weights = 1/np.sqrt(((1-s)*(1 + 2*s)*(1-s+G*(1+2*s)))/(3 * tot[peak:limit_idx]))
                    
                    except: 
                        weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
                        #ani_smooth[peak:limit_idx,i,j] = np.nan 

                    
                else:
                    weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
            
                y = r
                t = time[:(limit_idx-peak)]
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                
                model = lambda t, p1, p2, p3: (p1-p2)*np.exp(-t/p3)+p2
                                
                def error_func(x,t, y, w):
                    SS = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)
                    #MSE = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)/len(y)   
                    #MAE = np.sum(abs((y-model(t,x[0],x[1],x[2]))*w))/len(y)
                                           
                    log_dict.setdefault(f'{i}x{j}', []).append(SS)
                    #log_dict.setdefault(f'{i}x{j}', []).append(MSE)
                    #log_dict.setdefault(f'{i}x{j}', []).append(MAE)
                    
                    # for least squares - err function is the residuals 
                    return SS
                    #return MSE
                    #return MAE
                #bounds = ([0,-0.5,0],[1,1,100])

                #bounds = ((0, 1),(-0.5, 0.5),(0,100))
                
                method = method
                
                result = minimize(error_func, x0 = [p10,p20,p30], 
                                  method = method,
                                  args = (t, y, w), 
                                  bounds = None,
                                  options = {'dis': False})
                #print(result)
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                
                #obj_function[i,j] = result.fun 
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)

                if result.success is True:                 
                    successful_conv += 1
                    
                    fit = model(t, r0, rinf, theta)
                    
                    anisotropy_fit[peak:limit_idx,i,j] = fit
                    
                    red_chi_sq = np.sum(((y-fit)*w)**2)/(len(y)-len(result.x))
                    
                    chi2_img[i,j] = red_chi_sq
                    
                    #print(result.message)
                    #print(result.optimality)
                    #print(result.x)
                    
                    r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                    
                    residuals = (y - fit)*w
                
                    r0_img[i,j] = r0   # r0  - 2d image 
                    #B_img[i,j] = B
                    rinf_img[i,j] = rinf
                    theta_img[i,j] = theta
                    #res_img[peak:limit_idx,i,j] = residuals
                    
                    #if model == "BFGS" or model == "L-BFGS-B": 
                    #    try:
                    #        Jac = result.jac
                    #        cov = np.linalg.inv(Jac.T.dot(Jac))
                    #        perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted paramet
                    #        
                    #        r0_err_img[i,j] = perr[0]
                    #        rinf_err_img[i,j] = perr[1]
                    #        theta_err_img[i,j] = perr[1]
                    #        #successful_conv += 1
                    #    except:
                    ##        print("Jacobian not found")
                    #        anisotropy_fit[:,i,j] = np.nan
                    #        res_img[:,i,j] = np.nan 
                    #        ani_smooth[:,i,j] = np.nan
                    #        raw_decay[:,i,j] = np.nan
            
                     #       r0_img[i,j] = np.nan
                     #       rinf_img[i,j] = np.nan
                     #       theta_img[i,j] = np.nan
                     #       chi2_img[i,j] = np.nan
            
                     #       r0_err_img[i,j] = np.nan
                     #       rinf_err_img[i,j] = np.nan
                     #       theta_err_img[i,j] = np.nan
    
                    if total_attempts < 50:
                        plt.figure(2)
                        plt.cla()
                        plt.plot(t, y, label = 'raw')
                        plt.plot(t, fit, 'r', label = 'fit')
                        if smooth: 
                            plt.plot(t, s, 'm', label = 'smoothed')
                        plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                                  theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                        plt.pause(0.01)
                        
                        plt.figure(3)
                        plt.cla()
                        plt.plot(t, residuals)
                        plt.ylabel('residuals')
                        plt.pause(0.01)
                    else: 
                        plt.close()

                    print(f"""Estimated results 
                          r0 = {r0}
                          rinf = {rinf}
                          theta = {theta}
                          red chi sq = {red_chi_sq} """)
                
                else: 
                    print("convergence not reached")
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
                    #ani_smooth[:,i,j] = np.nan
                    #raw_decay[:,i,j] = np.nan
        
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
                
                        
                #res = error_func(result.x, t, y, w)/(len(y)-len([p10, p20]))
                                
                #par_conv_data[peak_idx:,i,j] = y 
                
                
    print(f'% successfuly convergence = {(successful_conv/total_attempts)*100}')
    
    print('% in bounds')
    print('r0')
    _ = percent_in_bounds(r0_img, [-0.3, 0.4])
    print('theta')
    _ = percent_in_bounds(theta_img, [1, 6]) 
    print('rinf')
    _ = percent_in_bounds(rinf_img, [-0.5, 0.3])
    print('chi2')
    _ = percent_in_bounds(chi2_img, [-1.5, 1.5])
          
    plt.figure(3)
    plt.imshow(rinf_img, clim = (-0.5,0.3) )
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    rinf_hist = rinf_img.ravel()[~np.isnan(rinf_img.ravel())]
    rinf_hist = rinf_hist[np.where(np.logical_and(rinf_hist >= -0.5, rinf_hist <= 0.3))]
    
    plt.figure(4)
    plt.hist(rinf_hist, bins = 100)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img, clim = (1,6))
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    theta_hist = theta_img.ravel()[~np.isnan(theta_img.ravel())]
    theta_hist = theta_hist[np.where(np.logical_and(theta_hist >= 1, theta_hist <= 6))]
    
    plt.figure(6)
    plt.hist(theta_hist, bins = 100)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img, clim = (-0.3, 0.4))
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    r0_hist = r0_img.ravel()[~np.isnan(r0_img.ravel())]
    r0_hist = r0_hist[np.where(np.logical_and(r0_hist >= -0.3, r0_hist <= 0.4))]
    
    
    plt.figure(8)
    plt.hist(r0_hist, bins = 100)
    plt.title("r0")
    plt.show()
    
    
    return rimg, r0_img, rinf_img, theta_img, anisotropy_fit, chi2_img, res_img, log_dict

def model_diff_evol_anisotropy_fit(perpimage, parimage, time, G, mask, r0_0, rinf_0, theta_0, strategy, smooth = False):
    '''
    Direct fit of the anisotroyp decay. 
    Binning essential ! 
    '''
    _,a,b = perpimage.shape
    p10,p20, p30 = r0_0, rinf_0, theta_0
    
    #totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    #peak_int = []
    
    perpimage, parimage, _ = align_image(perpimage,parimage, time)
    print('images aligned')
    totimg = parimage + 2 * G *perpimage
    print('total img created')

    rimg = (parimage - G * perpimage)/totimg
    print('r image created')

            
    #3D images to hold the decays and fits 
    anisotropy_fit = np.empty_like(perpimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float)   # residuals 
    
    #ani_smooth = np.empty_like(perpimage, dtype = float).nan
    #raw_decay = np.empty_like(perpimage, dtype = float).nan

    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 

    #obj_function = np.empty((perpimage.shape[1], perpimage.shape[2]))
    log_dict = {}
    
    limit_idx = None
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #anisotropy_fit[:,i,j] = np.nan
                #res_img[:,i,j] = np.nan 
                #ani_smooth[:,i,j] = np.nan
                #raw_decay[:,i,j] = np.nan
    
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                #tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]
                
                #perp = perpimage[:,i,j]
                
                tot = totimg[:,i,j]
                
                r = rimg[:,i,j]
                              
                #perp, par, peak, _ = align_peaks(perp, par, time, plot = False)
                
                peak = np.argmax(par)
                
                #peak_int.append(np.amax(par))
                                                
                #tot = par + 2 * G * perp
                
                #r = (par - G * perp) / tot
                
                if limit_idx is None:
                    plt.plot(time, r)
                    plt.title("Choose single point as limit")
                    plt.axvline(x=time[peak],color = 'r')
                    limit = plt.ginput(1)
                    limit_idx = find_nearest_neighbor_index(time[peak:], limit[0])
                    limit_idx = limit_idx[0]
                    limit_idx = peak + limit_idx
                    plt.close()
                    plt.plot(time[:(limit_idx-peak)],r[peak:limit_idx])
                    plt.title('Part that will be fit')
                    plt.pause(5)
                    plt.close()
                r = r[peak:limit_idx]
                #raw_decay[peak:limit_idx,i,j] = r
                             
                window_size = 15
                polynomial= 2
                # smooth 
                
                # implement buttons to choose suitable and move on 
                # s should have same length as r - peak to limit chosen 
                if smooth:
                    try: 
                        s = savgol_filter(r, window_size, polynomial)
                        if total_attempts < 3:
                            plt.plot(time[peak:limit_idx], s, 'b')
                            plt.plot(time[peak:limit_idx], r)
                            #plt.axvline(time, time[peak], 'r')
                            plt.title(f"Smoothed decay with window {window_size} and poly {polynomial}")
                            plt.show()
                            plt.pause(5)
                            plt.close()
                    
                        #ani_smooth[peak:limit_idx,i,j] = s 
                            
                        weights = 1/np.sqrt(((1-s)*(1 + 2*s)*(1-s+G*(1+2*s)))/(3 * tot[peak:limit_idx]))
                    
                    except: 
                        weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
                        #ani_smooth[peak:limit_idx,i,j] = np.nan 

                    
                else:
                    weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
            
                y = r
                t = time[:(limit_idx-peak)]
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                
                model = lambda t, p1, p2, p3: (p1-p2)*np.exp(-t/p3)+p2
                                
                def error_func(x,t, y, w):
                    #SS = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)
                    MSE = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)/len(y)   
                    #MAE = np.sum(abs((y-model(t,x[0],x[1],x[2]))*w))/len(y)
                                           
                    #log_dict.setdefault(f'{i}x{j}', []).append(SS)
                    log_dict.setdefault(f'{i}x{j}', []).append(MSE)
                    #log_dict.setdefault(f'{i}x{j}', []).append(MAE)
                    
                    # for least squares - err function is the residuals 
                    #return SS
                    return MSE
                    #return MAE
                #bounds = ([0,-0.5,0],[1,1,100])

                #bounds = ((0, 1),(-0.5, 0.5),(0,100))
                                
                result = differential_evolution(error_func, x0 = [p10,p20,p30], 
                                  method = method,
                                  args = (t, y, w), 
                                  bounds = None,
                                  strategy = strategy)
                #print(result)
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                
                #obj_function[i,j] = result.fun 
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)

                if result.success is True:                 
                    successful_conv += 1
                    
                    fit = model(t, r0, rinf, theta)
                    
                    anisotropy_fit[peak:limit_idx,i,j] = fit
                    
                    red_chi_sq = np.sum(((y-fit)*w)**2)/(len(y)-len(result.x))
                    
                    chi2_img[i,j] = red_chi_sq
                    
                    #print(result.message)
                    #print(result.optimality)
                    #print(result.x)
                    
                    r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                    
                    residuals = (y - fit)*w
                
                    r0_img[i,j] = r0   # r0  - 2d image 
                    #B_img[i,j] = B
                    rinf_img[i,j] = rinf
                    theta_img[i,j] = theta
                    #res_img[peak:limit_idx,i,j] = residuals

                     #       theta_err_img[i,j] = np.nan
    
                    if total_attempts < 50:
                        plt.figure(2)
                        plt.cla()
                        plt.plot(t, y, label = 'raw')
                        plt.plot(t, fit, 'r', label = 'fit')
                        if smooth: 
                            plt.plot(t, s, 'm', label = 'smoothed')
                        plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                                  theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                        plt.pause(0.01)
                        
                        plt.figure(3)
                        plt.cla()
                        plt.plot(t, residuals)
                        plt.ylabel('residuals')
                        plt.pause(0.01)
                    else: 
                        plt.close()

                    print(f"""Estimated results 
                          r0 = {r0}
                          rinf = {rinf}
                          theta = {theta}
                          red chi sq = {red_chi_sq} """)
                
                else: 
                    print("convergence not reached")
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
                    #ani_smooth[:,i,j] = np.nan
                    #raw_decay[:,i,j] = np.nan
        
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
                
                        
                #res = error_func(result.x, t, y, w)/(len(y)-len([p10, p20]))
                                
                #par_conv_data[peak_idx:,i,j] = y 
                
                
    print(f'% successfuly convergence = {(successful_conv/total_attempts)*100}')
    
    print('% in bounds')
    print('r0')
    _ = percent_in_bounds(r0_img, [-0.3, 0.4])
    print('theta')
    _ = percent_in_bounds(theta_img, [1, 6]) 
    print('rinf')
    _ = percent_in_bounds(rinf_img, [-0.5, 0.3])
    print('chi2')
    _ = percent_in_bounds(chi2_img, [-1.5, 1.5])
          
    plt.figure(3)
    plt.imshow(rinf_img, clim = (-0.5,0.3) )
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    rinf_hist = rinf_img.ravel()[~np.isnan(rinf_img.ravel())]
    rinf_hist = rinf_hist[np.where(np.logical_and(rinf_hist >= -0.5, rinf_hist <= 0.3))]
    
    plt.figure(4)
    plt.hist(rinf_hist, bins = 100)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img, clim = (1,6))
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    theta_hist = theta_img.ravel()[~np.isnan(theta_img.ravel())]
    theta_hist = theta_hist[np.where(np.logical_and(theta_hist >= 1, theta_hist <= 6))]
    
    plt.figure(6)
    plt.hist(theta_hist, bins = 100)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img, clim = (-0.5, 0.3))
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    r0_hist = r0_img.ravel()[~np.isnan(r0_img.ravel())]
    r0_hist = r0_hist[np.where(np.logical_and(r0_hist >= -0.5, r0_hist <= 0.3))]
    
    
    plt.figure(8)
    plt.hist(r0_hist, bins = 100)
    plt.title("r0")
    plt.show()
    
    return rimg, r0_img, rinf_img, theta_img, anisotropy_fit, chi2_img, res_img, log_dict

def model_dual_annealing_anisotropy_fit(perpimage, parimage, time, G, mask, smooth = False):
    '''
    Direct fit of the anisotroyp decay. 
    Binning essential ! 
    '''
    _,a,b = perpimage.shape
    
    #totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    #peak_int = []
    
    perpimage, parimage, _ = align_image(perpimage,parimage, time)
    print('images aligned')
    totimg = parimage + 2 * G *perpimage
    print('total img created')

    rimg = (parimage - G * perpimage)/totimg
    print('r image created')

            
    #3D images to hold the decays and fits 
    anisotropy_fit = np.empty_like(perpimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float)   # residuals 
    
    #ani_smooth = np.empty_like(perpimage, dtype = float).nan
    #raw_decay = np.empty_like(perpimage, dtype = float).nan

    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 

    #obj_function = np.empty((perpimage.shape[1], perpimage.shape[2]))
    log_dict = {}
    
    limit_idx = None
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #anisotropy_fit[:,i,j] = np.nan
                #res_img[:,i,j] = np.nan 
                #ani_smooth[:,i,j] = np.nan
                #raw_decay[:,i,j] = np.nan
    
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                #tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]
                
                #perp = perpimage[:,i,j]
                
                tot = totimg[:,i,j]
                
                r = rimg[:,i,j]
                              
                #perp, par, peak, _ = align_peaks(perp, par, time, plot = False)
                
                peak = np.argmax(par)
                
                #peak_int.append(np.amax(par))
                                                
                #tot = par + 2 * G * perp
                
                #r = (par - G * perp) / tot
                
                if limit_idx is None:
                    plt.plot(time, r)
                    plt.title("Choose single point as limit")
                    plt.axvline(x=time[peak],color = 'r')
                    limit = plt.ginput(1)
                    limit_idx = find_nearest_neighbor_index(time[peak:], limit[0])
                    limit_idx = limit_idx[0]
                    limit_idx = peak + limit_idx
                    plt.close()
                    plt.plot(time[:(limit_idx-peak)],r[peak:limit_idx])
                    plt.title('Part that will be fit')
                    plt.pause(5)
                    plt.close()
                r = r[peak:limit_idx]
                
                #print(f'Limiting index here is: {limit_idx}')
                #raw_decay[peak:limit_idx,i,j] = r
                             
                window_size = 15
                polynomial= 2
                # smooth 
                
                # implement buttons to choose suitable and move on 
                # s should have same length as r - peak to limit chosen 
                if smooth:
                    try: 
                        s = savgol_filter(r, window_size, polynomial)
                        if total_attempts < 3:
                            plt.plot(time[peak:limit_idx], s, 'b')
                            plt.plot(time[peak:limit_idx], r)
                            #plt.axvline(time, time[peak], 'r')
                            plt.title(f"Smoothed decay with window {window_size} and poly {polynomial}")
                            plt.show()
                            plt.pause(5)
                            plt.close()
                    
                        #ani_smooth[peak:limit_idx,i,j] = s 
                            
                        weights = 1/np.sqrt(((1-s)*(1 + 2*s)*(1-s+G*(1+2*s)))/(3 * tot[peak:limit_idx]))
                    
                    except: 
                        weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
                        #ani_smooth[peak:limit_idx,i,j] = np.nan 

                    
                else:
                    weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
            
                y = r
                t = time[:(limit_idx-peak)]
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                
                model = lambda t, p1, p2, p3: (p1-p2)*np.exp(-t/p3)+p2
                                
                def error_func(x,t, y, w):
                    #SS = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)
                    MSE = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)/len(y)   
                    #MAE = np.sum(abs((y-model(t,x[0],x[1],x[2]))*w))/len(y)
                                           
                    #log_dict.setdefault(f'{i}x{j}', []).append(SS)
                    log_dict.setdefault(f'{i}x{j}', []).append(MSE)
                    #log_dict.setdefault(f'{i}x{j}', []).append(MAE)
                    
                    # for least squares - err function is the residuals 
                    #return SS
                    return MSE
                    #return MAE
                #bounds = ([0,-0.5,0],[1,1,100])

                #bounds = ((0, 1),(-0.5, 0.5),(0,100))
                
                bounds = [[-0.5,0.5], [-0.5, 0.5], [0.5,100]]         
                
                try: 
                    result = dual_annealing(error_func,
                                  args = (t, y, w), 
                                  bounds = bounds)
                except ValueError as ve:
                    ## error, continue on to next pixel
                    
                    print('Value error, pixel fit aborted')
                    print(ve)
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
    
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
                        
                    continue
                #print(result)
                r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                
                #obj_function[i,j] = result.fun 
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)

                if result.success is True:                 
                    successful_conv += 1
                    
                    fit = model(t, r0, rinf, theta)
                    
                    anisotropy_fit[peak:limit_idx,i,j] = fit
                    
                    red_chi_sq = np.sum(((y-fit)*w)**2)/(len(y)-len(result.x))
                    
                    chi2_img[i,j] = red_chi_sq
                    
                    #print(result.message)
                    #print(result.optimality)
                    #print(result.x)
                    
                    r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                    
                    residuals = (y - fit)*w
                
                    r0_img[i,j] = r0   # r0  - 2d image 
                    #B_img[i,j] = B
                    rinf_img[i,j] = rinf
                    theta_img[i,j] = theta
                    #res_img[peak:limit_idx,i,j] = residuals
                    
    
                    if total_attempts < 50:
                        plt.figure(2)
                        plt.cla()
                        plt.plot(t, y, label = 'raw')
                        plt.plot(t, fit, 'r', label = 'fit')
                        if smooth: 
                            plt.plot(t, s, 'm', label = 'smoothed')
                        plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                                  theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                        plt.pause(0.01)
                        
                        plt.figure(3)
                        plt.cla()
                        plt.plot(t, residuals)
                        plt.ylabel('residuals')
                        plt.pause(0.01)
                    else: 
                        plt.close()

                    print(f"""Estimated results 
                          r0 = {r0}
                          rinf = {rinf}
                          theta = {theta}
                          red chi sq = {red_chi_sq} """)
                
                else: 
                    print("convergence not reached")
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
                    #ani_smooth[:,i,j] = np.nan
                    #raw_decay[:,i,j] = np.nan
        
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
                
                        
                #res = error_func(result.x, t, y, w)/(len(y)-len([p10, p20]))
                                
                #par_conv_data[peak_idx:,i,j] = y 
                
                
    print(f'% successfuly convergence = {(successful_conv/total_attempts)*100}')
    
    print('% in bounds')
    print('r0')
    _ = percent_in_bounds(r0_img, [-0.3, 0.4])
    print('theta')
    _ = percent_in_bounds(theta_img, [1, 6]) 
    print('rinf')
    _ = percent_in_bounds(rinf_img, [-0.5, 0.3])
    print('chi2')
    _ = percent_in_bounds(chi2_img, [-1.5, 1.5])
          
    plt.figure(3)
    plt.imshow(rinf_img, clim = (-0.5,0.3) )
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    rinf_hist = rinf_img.ravel()[~np.isnan(rinf_img.ravel())]
    rinf_hist = rinf_hist[np.where(np.logical_and(rinf_hist >= -0.5, rinf_hist <= 0.3))]
    
    plt.figure(4)
    plt.hist(rinf_hist, bins = 100)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img, clim = (1,6))
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    theta_hist = theta_img.ravel()[~np.isnan(theta_img.ravel())]
    theta_hist = theta_hist[np.where(np.logical_and(theta_hist >= 1, theta_hist <= 6))]
    
    plt.figure(6)
    plt.hist(theta_hist, bins = 100)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img, clim = (-0.4, 0.4))
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    r0_hist = r0_img.ravel()[~np.isnan(r0_img.ravel())]
    r0_hist = r0_hist[np.where(np.logical_and(r0_hist >= -0.5, r0_hist <= 0.3))]
    
    
    plt.figure(8)
    plt.hist(r0_hist, bins = 100)
    plt.title("r0")
    plt.show()
    
    return rimg, r0_img, rinf_img, theta_img, anisotropy_fit, chi2_img, res_img, log_dict


def model_shgo_anisotropy_fit(perpimage, parimage, time, G, mask, smooth = False):
    '''
    Direct fit of the anisotroyp decay. 
    Binning essential ! 
    '''
    _,a,b = perpimage.shape
    
    #totalimage = parimage + 2 * G * perpimage
    
    # make peak intensity distribution 
    #peak_int = []
    
    perpimage, parimage, _ = align_image(perpimage,parimage, time)
    print('images aligned')
    totimg = parimage + 2 * G *perpimage
    print('total img created')

    rimg = (parimage - G * perpimage)/totimg
    print('r image created')

            
    #3D images to hold the decays and fits 
    anisotropy_fit = np.empty_like(perpimage, dtype = float) # this will hold the fit 
    res_img = np.empty_like(perpimage, dtype = float)   # residuals 
    
    #ani_smooth = np.empty_like(perpimage, dtype = float).nan
    #raw_decay = np.empty_like(perpimage, dtype = float).nan

    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))

    successful_conv = 0 
    total_attempts = 0 

    #obj_function = np.empty((perpimage.shape[1], perpimage.shape[2]))
    log_dict = {}
    
    limit_idx = None
    plt.ion()
    for i in range(a):  # create anisotropy and lifetime images 
        for j in range(b):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #anisotropy_fit[:,i,j] = np.nan
                #res_img[:,i,j] = np.nan 
                #ani_smooth[:,i,j] = np.nan
                #raw_decay[:,i,j] = np.nan
    
                r0_img[i,j] = np.nan
                rinf_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
                
            else: # fit 
                print(f'now on pixel {i} x {j} out of {a} x {b} ')
                
                total_attempts += 1                                
                
                #par = parimage[:,i,j]
                #perp = perpimage[:,i,j]

                #tot = totalimage[:,i,j]
                                
                par = parimage[:,i,j]
                
                #perp = perpimage[:,i,j]
                
                tot = totimg[:,i,j]
                
                r = rimg[:,i,j]
                              
                #perp, par, peak, _ = align_peaks(perp, par, time, plot = False)
                
                peak = np.argmax(par)
                
                #peak_int.append(np.amax(par))
                                                
                #tot = par + 2 * G * perp
                
                #r = (par - G * perp) / tot
                
                if limit_idx is None:
                    plt.plot(time, r)
                    plt.title("Choose single point as limit")
                    plt.axvline(x=time[peak],color = 'r')
                    limit = plt.ginput(1)
                    limit_idx = find_nearest_neighbor_index(time[peak:], limit[0])
                    limit_idx = limit_idx[0]
                    limit_idx = peak + limit_idx
                    plt.close()
                    plt.plot(time[:(limit_idx-peak)],r[peak:limit_idx])
                    plt.title('Part that will be fit')
                    plt.pause(5)
                    plt.close()
                r = r[peak:limit_idx]
                
                #print(f'Limiting index here is: {limit_idx}')
                #raw_decay[peak:limit_idx,i,j] = r
                             
                window_size = 15
                polynomial= 2
                # smooth 
                
                # implement buttons to choose suitable and move on 
                # s should have same length as r - peak to limit chosen 
                if smooth:
                    try: 
                        s = savgol_filter(r, window_size, polynomial)
                        if total_attempts < 3:
                            plt.plot(time[peak:limit_idx], s, 'b')
                            plt.plot(time[peak:limit_idx], r)
                            #plt.axvline(time, time[peak], 'r')
                            plt.title(f"Smoothed decay with window {window_size} and poly {polynomial}")
                            plt.show()
                            plt.pause(5)
                            plt.close()
                    
                        #ani_smooth[peak:limit_idx,i,j] = s 
                            
                        weights = 1/np.sqrt(((1-s)*(1 + 2*s)*(1-s+G*(1+2*s)))/(3 * tot[peak:limit_idx]))
                    
                    except: 
                        weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
                        #ani_smooth[peak:limit_idx,i,j] = np.nan 

                    
                else:
                    weights = 1/np.sqrt(((1-r)*(1 + 2*r)*(1-r+G*(1+2*r)))/(3 * tot[peak:limit_idx]))
            
                y = r
                t = time[:(limit_idx-peak)]
                w = weights
                                
                '''
                p1 - r0 
                p2 - rinf 
                p3 - theta 
                '''
                
                model = lambda t, p1, p2, p3: (p1-p2)*np.exp(-t/p3)+p2
                                
                def error_func(x,t, y, w):
                    #SS = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)
                    MSE = np.sum(((y - model(t,x[0],x[1],x[2]))*w)**2)/len(y)   
                    #MAE = np.sum(abs((y-model(t,x[0],x[1],x[2]))*w))/len(y)
                                           
                    #log_dict.setdefault(f'{i}x{j}', []).append(SS)
                    log_dict.setdefault(f'{i}x{j}', []).append(MSE)
                    #log_dict.setdefault(f'{i}x{j}', []).append(MAE)
                    
                    # for least squares - err function is the residuals 
                    #return SS
                    return MSE
                    #return MAE
                #bounds = ([0,-0.5,0],[1,1,100])

                #bounds = ((0, 1),(-0.5, 0.5),(0,100))
                
                bounds = [[-0.5,0.5], [-0.5, 0.5], [0.5,100]]         
                
                try: 
                    result = shgo(error_func,
                                  args = (t, y, w), 
                                  bounds = bounds)
                except ValueError as ve:
                    ## error, continue on to next pixel
                    
                    print('Value error, pixel fit aborted')
                    print(ve)
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
    
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
                        
                    continue
                #print(result)
                
                #obj_function[i,j] = result.fun 
                
                #print(result.message)
                #print(result.optimality)
                #print(result.x)

                if result.success is True:         
                    successful_conv += 1
                    
                    r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                    
                    fit = model(t, r0, rinf, theta)
                    
                    anisotropy_fit[peak:limit_idx,i,j] = fit
                    
                    red_chi_sq = np.sum(((y-fit)*w)**2)/(len(y)-len(result.x))
                    
                    chi2_img[i,j] = red_chi_sq
                    
                    #print(result.message)
                    #print(result.optimality)
                    #print(result.x)

                    r0, rinf, theta = result.x[0], result.x[1], result.x[2]
                    
                    residuals = (y - fit)*w
                
                    r0_img[i,j] = r0   # r0  - 2d image 
                    #B_img[i,j] = B
                    rinf_img[i,j] = rinf
                    theta_img[i,j] = theta
                    #res_img[peak:limit_idx,i,j] = residuals
                    
    
                    if total_attempts < 50:
                        plt.figure(2)
                        plt.cla()
                        plt.plot(t, y, label = 'raw')
                        plt.plot(t, fit, 'r', label = 'fit')
                        if smooth: 
                            plt.plot(t, s, 'm', label = 'smoothed')
                        plt.title(f''' r0 = {r0:.2f}, rinf = {rinf:.2f}, 
                                  theta = {theta:.2f}, x2 = {red_chi_sq:.2f}''')
                        plt.pause(0.01)
                        
                        plt.figure(3)
                        plt.cla()
                        plt.plot(t, residuals)
                        plt.ylabel('residuals')
                        plt.pause(0.01)
                    else: 
                        plt.close()

                    print(f"""Estimated results 
                          r0 = {r0}
                          rinf = {rinf}
                          theta = {theta}
                          red chi sq = {red_chi_sq} """)
                
                else: 
                    print("convergence not reached")
                    anisotropy_fit[:,i,j] = np.nan
                    res_img[:,i,j] = np.nan 
                    #ani_smooth[:,i,j] = np.nan
                    #raw_decay[:,i,j] = np.nan
        
                    r0_img[i,j] = np.nan
                    rinf_img[i,j] = np.nan
                    theta_img[i,j] = np.nan
                    chi2_img[i,j] = np.nan
                
                        
                #res = error_func(result.x, t, y, w)/(len(y)-len([p10, p20]))
                                
                #par_conv_data[peak_idx:,i,j] = y 
                
                
    print(f'% successfuly convergence = {(successful_conv/total_attempts)*100}')
    
    print('% in bounds')
    print('r0')
    _ = percent_in_bounds(r0_img, [-0.3, 0.4])
    print('theta')
    _ = percent_in_bounds(theta_img, [1, 6]) 
    print('rinf')
    _ = percent_in_bounds(rinf_img, [-0.5, 0.3])
    print('chi2')
    _ = percent_in_bounds(chi2_img, [-1.5, 1.5])
          
    plt.figure(3)
    plt.imshow(rinf_img, clim = (-0.5,0.3) )
    plt.title("Rinf")
    plt.colorbar()
    plt.show()
    
    rinf_hist = rinf_img.ravel()[~np.isnan(rinf_img.ravel())]
    rinf_hist = rinf_hist[np.where(np.logical_and(rinf_hist >= -0.5, rinf_hist <= 0.3))]
    
    plt.figure(4)
    plt.hist(rinf_hist, bins = 100)
    plt.title("Rinf")
    plt.show()
    
    plt.figure(5)
    plt.imshow(theta_img, clim = (1,6))
    plt.title("Theta")
    plt.colorbar()
    plt.show()
    
    theta_hist = theta_img.ravel()[~np.isnan(theta_img.ravel())]
    theta_hist = theta_hist[np.where(np.logical_and(theta_hist >= 1, theta_hist <= 6))]
    
    plt.figure(6)
    plt.hist(theta_hist, bins = 100)
    plt.title("Theta")
    plt.show()
    
    plt.figure(7)
    plt.imshow(r0_img, clim = (-0.5, 0.3))
    plt.title("r0")
    plt.colorbar()
    plt.show()
    
    r0_hist = r0_img.ravel()[~np.isnan(r0_img.ravel())]
    r0_hist = r0_hist[np.where(np.logical_and(r0_hist >= -0.5, r0_hist <= 0.3))]
    
    
    plt.figure(8)
    plt.hist(r0_hist, bins = 100)
    plt.title("r0")
    plt.show()
    
    return rimg, r0_img, rinf_img, theta_img, anisotropy_fit, chi2_img, res_img, log_dict
