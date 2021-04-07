#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import torch
import gpytorch
import numpy as np
from matplotlib.patches import Polygon
from hpl_functions import vx_interp, ey_interp, t_interp, get_s_from_t
from sklearn.utils import shuffle


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class hplStrategy():
    def __init__(self, T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, retrain_flag):
        self.T = T # number of environment prediction samples --> training input will have length 21
        self.ds = ds # environment sample step (acts on s)
        self.N_mpc = N_mpc # number of timesteps (of length dt) for the MPC horizon 
        self.dt = dt # MPC sample time (acts on t). determines how far out we take our training label and env. forecast. (N*dt)
        self.fint = fint # s counter which determines start of new training data
        self.s_conf_thresh = s_conf_thresh # confidence threshold for del_s prediction
        self.ey_conf_thresh = ey_conf_thresh # confidence threshold for ey prediction
        self.map = map
        
        if retrain_flag:
            self.trainStrategy()       
        else:                
            self.s_model = pickle.load(open('AE:BE:US/s_model.pkl','rb'))
            self.s_model.double()
            self.s_model.eval()
            self.s_likelihood = pickle.load(open('AE:BE:US/s_likelihood.pkl','rb'))
            self.s_likelihood.eval()
            self.ey_model = pickle.load(open('AE:BE:US/ey_model.pkl','rb'))
            self.ey_model.double()
            self.ey_model.eval()
            self.ey_likelihood = pickle.load(open('AE:BE:US/ey_likelihood.pkl','rb'))
            self.ey_likelihood.eval()
        
    def trainStrategy(self):
        train_list = ['Tracks/CN.pkl', 'Tracks/AT.pkl', 'Tracks/MX.pkl', 'Tracks/HU.pkl', 'Tracks/CA.pkl', 'Tracks/IT.pkl','Tracks/JP.pkl']
        test_list = ['Tracks/US.pkl', 'Tracks/BE.pkl', 'Tracks/AE.pkl']
        
        [TrainD, TrainL] = self.make_GP_data(train_list)
        s_model, s_likelihood = self.train_s_GP(TrainD, TrainL)
        print('now training ey')
        ey_model, ey_likelihood = self.train_ey_GP(TrainD, TrainL)
        print('done')
        
        # save for future
        # pickle.dump(s_model,open('AE:BE:US/s_model.pkl','wb'))
        # pickle.dump(s_likelihood,open('AE:BE:US/s_likelihood.pkl','wb'))
        # pickle.dump(ey_model,open('AE:BE:US/ey_model.pkl','wb'))
        # pickle.dump(ey_likelihood,open('AE:BE:US/ey_likelihood.pkl','wb'))
        
        self.s_model = s_model
        self.s_model.double()
        self.s_model.eval()
        self.s_likelihood = s_likelihood
        self.s_likelihood.eval()
        self.ey_model = ey_model
        self.ey_model.double()
        self.ey_model.eval()
        self.ey_likelihood = ey_likelihood
        self.ey_likelihood.eval()
        
        return
        
    def make_GP_data(self, train_list):
        TrainD = np.array([])
        TrainL = np.array([])
        
        for file in train_list:
            db = pickle.load(open(file,"rb"))
            pframe= db['raceline']
            [TD, TL]=self.createTrainingData_theta(pframe)
            TrainD = np.vstack([TrainD,TD]) if np.size(TrainD) else TD
            TrainL = np.vstack([TrainL,TL]) if np.size(TrainL) else TL
            
            return TrainD, TrainL
        
    def createTrainingData_theta(self, pframe):
        # this function creates training data matrices with inputs (X) and outputs (Y).
        
        # inputs to GPs: [vx[0], d_theta[1], ..., d_theta[T]]
        # outputs of GPs: [s[N]-s[0]]
        
        # inputs to GPey: [ey[0], d_theta[1], ..., d_theta[T]]
        # outputs of GPey: [ey[N]]
        
        # input vector: [vx[0], ey[0], d_theta[1], ..., d_theta[T]]
        # output: [s[N]-s[0], ey[N]]
        
        TrainD = [] # training input
        TrainL = [] # training labels [s_pred, ey_pred]
            
        cur_s = 0
        while True:
            s_indices = np.arange(cur_s, cur_s + (self.ds * self.T), self.ds)    # the s-values that we evaluate theta at   
            s_indices = s_indices[1:]
            
            # check stopping conditions. do we have enough 's' left to predict?
            if s_indices[-1]>pframe.index[-1]:
                break 
            
            # first training data entry is the current ey. # collect curvature along the s_indices vector
            t_input = []
            cur_vx = vx_interp(pframe, cur_s)
            t_input.append(cur_vx/10) # normalize to be within (0,1)!!!
            cur_ey = ey_interp(pframe, cur_s)                      
            t_input.append(cur_ey)
            
            # the first angle is the angle from the current state to the next. deviation calculated from tangent.
            b = map.cs(cur_s,1)
            tmp_s = map.getGlobalPosition(cur_s,0,0)
            tmp_next_s = map.getGlobalPosition(s_indices[0],0,0) 
            f = [tmp_next_s[0]-tmp_s[0], tmp_next_s[1] - tmp_s[1]]
            
            angle_sign = np.sign(np.cross(b,f))
            angle_size = np.arccos(np.dot(b,f)/(np.linalg.norm(b) * np.linalg.norm(f)))
           
            # current angle of transition, at s?
            theta = angle_sign * angle_size
            t_input.append(theta)
            
            for s in s_indices:            
                tmp_s = map.getGlobalPosition(s,0,0)
                
                # vector pointing from prev_s to s?           
                tmp_prev_s = map.getGlobalPosition(s - self.ds,0,0)
                b = [tmp_s[0]-tmp_prev_s[0], tmp_s[1] - tmp_prev_s[1]] 
                
                # vector pointing from s to next_s? 
                tmp_next_s = map.getGlobalPosition(s + self.ds,0,0)
                f = [tmp_next_s[0]-tmp_s[0], tmp_next_s[1] - tmp_s[1]]             
               
                # sign of the angle?
                angle_sign = np.sign(np.cross(b,f))
                angle_size = np.arccos(np.dot(b,f)/(np.linalg.norm(b) * np.linalg.norm(f)))
               
                # current angle of transition, at s?
                theta = angle_sign * angle_size
    
                t_input.append(theta)      
                           
            # concatenate training data and add to list    
            TrainD.append(t_input)
            
            # training labels: d_s, ey
            cur_t = t_interp(pframe, cur_s)
            pred_t = cur_t + self.N*self.dt 
            pred_s = get_s_from_t(pframe,pred_t)
                    
            # check if the s that interpolates to the end state is still in range
            if pred_s>=pframe.index[-1]:
                break
            TrainL.append([pred_s-cur_s, ey_interp(pframe, pred_s)]) 
            
            cur_s += self.fint        
            
        return TrainD, TrainL
    
    def train_s_GP(self, TrainD, TrainL):
    
        # training data GP_s
        s_train_data = TrainD[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
        s_train_label = TrainL[:, 0]
        s_train_data_shuff, s_train_label_shuff = shuffle(s_train_data, s_train_label)
        s_train_data = torch.from_numpy(s_train_data_shuff).double()
        s_train_label = torch.from_numpy(s_train_label_shuff).double()
        
        # initialize likelihood and model
        s_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        s_model = ExactGPModel(s_train_data, s_train_label, s_likelihood)
        training_iter = 100
        s_model.double()
        s_model.train()
        s_likelihood.train()
        optimizer = torch.optim.Adam(s_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(s_likelihood, s_model)
        
        best_loss = 10
        best_lengthscale = 10
        best_noise = 10
        
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = s_model(s_train_data)
            # Calc loss and backprop gradients
            loss = -mll(output, s_train_label)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                s_model.covar_module.base_kernel.lengthscale.item(),
                s_model.likelihood.noise.item()
            ))
            # save the best value
            if loss.item() < best_loss:
                #print('updated parameters')
                best_loss = loss.item()
                best_lengthscale = s_model.covar_module.base_kernel.lengthscale.item(),
                best_noise = s_model.likelihood.noise.item()
                  
            optimizer.step()
        
        s_model.covar_module.base_kernel.lengthscale = best_lengthscale
        s_model.likelihood.noise = best_noise
        s_model.train()  # this clears any precomputed caches 
        
        return s_model, s_likelihood

    def train_ey_GP(TrainD, TrainL):
        # training data GP_ey
        ey_train_data = TrainD[:,1:]
        ey_train_data = torch.from_numpy(ey_train_data).double()
        ey_train_label = TrainL[:,1]
        ey_train_label = torch.from_numpy(ey_train_label).double()
        
        # initialize likelihood and model
        ey_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        ey_model = ExactGPModel(ey_train_data, ey_train_label, ey_likelihood)
        training_iter = 100
        ey_model.double()
        ey_model.train()
        ey_likelihood.train()
        optimizer = torch.optim.Adam(ey_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(ey_likelihood, ey_model)
        
        best_loss = 10
        best_lengthscale = 10
        best_noise = 10
        
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = ey_model(ey_train_data)
            # Calc loss and backprop gradients
            loss = -mll(output, ey_train_label)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                ey_model.covar_module.base_kernel.lengthscale.item(),
                ey_model.likelihood.noise.item()
            ))
            # save the best value
            if loss.item() < best_loss:
                #print('updated parameters')
                best_loss = loss.item()
                best_lengthscale = ey_model.covar_module.base_kernel.lengthscale.item(),
                best_noise = ey_model.likelihood.noise.item()
            
            optimizer.step()
                 
        ey_model.covar_module.base_kernel.lengthscale = best_lengthscale
        ey_model.likelihood.noise = best_noise
        ey_model.train()  # this clears any precomputed caches 
        
        return ey_model, ey_likelihood

    
    
    def get_gp_input(self, x):    
        cur_s = x[4]
        cur_ey = x[5]
        cur_vx = x[0]
        
        t_input = []
        t_input.append(cur_vx/10)
        t_input.append(cur_ey)
        
        s_indices = np.arange(cur_s, cur_s + (self.ds * self.T), self.ds)     
        s_indices = s_indices[1:]
    
        # the first angle is the angle from the current state to the next. deviation calculated from tangent.
        b = self.map.cs(cur_s,1)
        tmp_s = self.map.getGlobalPosition(cur_s,0,0)
        tmp_next_s = self.map.getGlobalPosition(s_indices[0],0,0) # CHANGED FROM [1] TO [0]
        f = [tmp_next_s[0]-tmp_s[0], tmp_next_s[1] - tmp_s[1]]
        
        angle_sign = np.sign(np.cross(b,f))
        angle_size = np.arccos(np.dot(b,f)/(np.linalg.norm(b) * np.linalg.norm(f)))
       
        # what's the current angle of transition, at s?
        theta = angle_sign * angle_size
        t_input.append(theta)
        
        # for the remaining states, we go 
        for s in s_indices:            
            # what's the current point?
            tmp_s = self.map.getGlobalPosition(s,0,0)
            
            # what's the vector pointing from prev_s to s?           
            tmp_prev_s = self.map.getGlobalPosition(s - self.ds,0,0)
            b = [tmp_s[0]-tmp_prev_s[0], tmp_s[1] - tmp_prev_s[1]] 
            
            # what's the vector pointing from s to next_s? 
            tmp_next_s = self.map.getGlobalPosition(s + self.ds,0,0)
            f = [tmp_next_s[0]-tmp_s[0], tmp_next_s[1] - tmp_s[1]]             
           
            # what's the sign of the angle?
            angle_sign = np.sign(np.cross(b,f))
            angle_size = np.arccos(np.dot(b,f)/(np.linalg.norm(b) * np.linalg.norm(f)))
           
            # what's the current angle of transition, at s?
            theta = angle_sign * angle_size
    
            t_input.append(theta) # the remaining training data entries are curvature forecasts      
            
        if len(t_input)>17:
            # sometimes precision causes this thing to have length 18
            t_input = t_input[0:17]
        
        t_input = np.reshape(t_input, (1,17)) 
        s_gp_input = t_input[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
        ey_gp_input = t_input[:,1:17]    
    
        return ey_gp_input, s_gp_input
        

    def evaluateStrategy(self, x_state):
        ey_gp_input, s_gp_input = self.get_gp_input(x_state)
        
        s_gp_input = torch.from_numpy(s_gp_input).double()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.s_likelihood(self.s_model(s_gp_input))
            lower, upper = observed_pred.confidence_region() # pm 2 standard deviations
            est_s = observed_pred.mean.numpy()
            std_s = (upper.numpy() - lower.numpy())
        
        ey_gp_input = torch.from_numpy(ey_gp_input).double()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.ey_likelihood(self.ey_model(ey_gp_input))
            lower, upper = observed_pred.confidence_region() # pm 2 standard deviations
            est_ey = observed_pred.mean.numpy()
            std_ey = (upper.numpy() - lower.numpy())/2  
            
        est_s += x_state[4] 
        est_s = est_s[0]
        est_ey = est_ey[0]
        std_s = std_s[0]
        std_ey = std_ey[0]
        
        # convert into current strategy set in global coordinates
        rect_pts_xy = np.array([self.map.getGlobalPosition(est_s - std_s, est_ey - std_ey,0)])
        rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(self.map.getGlobalPosition(est_s + std_s, est_ey - std_ey, 0), (1,-1))))
        rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(self.map.getGlobalPosition(est_s + std_s, est_ey + std_ey, 0), (1,-1))))
        rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(self.map.getGlobalPosition(est_s - std_s, est_ey + std_ey, 0), (1,-1))))
        strategy_set_xy = Polygon(rect_pts_xy, True, color = 'b',alpha = 0.3)
        
        # store the means and standard deviations
        strategy_info = np.array([est_s, std_s, est_ey, std_ey])
              
        return est_s, std_s, est_ey, std_ey, strategy_set_xy, strategy_info
        