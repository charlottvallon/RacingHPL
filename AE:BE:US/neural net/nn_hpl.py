#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:32:07 2021

@author: vallon2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:37:31 2020
Main file for running racing-HPL.
@author: vallon2
"""

import numpy as np
from pyomo.environ import *
from pyomo.dae import *
import sys
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct
import glob, os
from sklearn.model_selection import RandomizedSearchCV
import scipy 
from hpl_functions import *
from Track_new import *
from pyomo.util.infeasible import log_infeasible_constraints
import math
import torch
import gpytorch
from matplotlib.patches import Polygon

    

#%% Initialization 

# define the test environment from loaded pickle file
testfile = 'Tracks/AE.pkl'
[raceline_X,raceline_Y,Cur, map, race_time, pframe] = plotFromFile(testfile, lineflag=True)
# what is the initial state of the raceline? let's use this to start the MPC

# training data parameters for GP training and MPC
T = 15 # number of environment prediction samples --> training input will have length 21
ds = 2 # environment sample step (acts on s)
N_mpc = 20 # number of timesteps (of length dt) for the MPC horizon 
dt = 0.1 # MPC sample time (acts on t). determines how far out we take our training label and env. forecast. (N*dt)
fint = 0.5 # s counter which determines start of new training data

# hpl parameters
s_conf_thresh = 8 # confidence threshold for del_s prediction
ey_conf_thresh = 1.3 # confidence threshold for ey prediction

# safety-mpc parameters
gamma = 0.1
gamma = 0.01
vt = 0.5

net = pickle.load(open('neural_net.pkl','rb'))

# decide whether to use the strategy.
strategy_flag = True # 1 --> use strategy set, 0 --> only use safety controller
beta = False

#%% Load the learned safety classifier (SVM, or CVX)
# svm hyperplane info is used to construct the linear terminal set constraints
clf = pickle.load(open('SafetyControl/clf.pkl','rb'))

# scaler info is used to transform the states into standardized ones
scaler = pickle.load(open('SafetyControl/scaler.pkl','rb'))
#transformed = scaler.scale_*(np.delete(safe_pts,4,1) - scaler.mean_)/(scaler.var_)


#%% Functions

def strategy_NN(x, map, vmodel, dt, N, set_list, clf, scaler, beta):
    cv_pred = curv_pred_LMPC(x, map, x[0], dt, N) # use current vx to estimate positiont =
    
    # centers contains [s_pred, s_std, ey_pred, ey_std]
    s_pred, ey_pred = set_list[N-1]

    vt = 10
    du_weight = 0.5
        
    TrackHalfW = map.width
    TrackLength = map.TrackLength
    
    # define model parameters
    if vmodel == 'BARC':
        mass  = 1.98
        lf = 0.125
        lr = 0.125
        Iz = 0.024
        Df = 0.8 * mass * 9.81 / 2.0
        Cf = 1.25
        Bf = 1.0
        Dr = 0.8 * mass * 9.81 / 2.0
        Cr = 1.25
        Br = 1.0   
        model = m = ConcreteModel()
        m.sf = Param(initialize = TrackLength)
        m.t = RangeSet(0, N) 
        m.tnotN = m.t - [ N ]
        m.t_u = m.tnotN - [0]
        m.u0 = Var(m.tnotN, bounds=(-1,1), initialize=0)
        m.u1 = Var(m.tnotN, bounds=(-0.5,0.5), initialize=0)
        m.alpha_f = Var(m.t, bounds=(-2,2), initialize=0)
        m.alpha_r = Var(m.t, bounds=(-2,2), initialize=0)
        m.Fyf = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.t,bounds=(0,10), initialize=0.01) #vx - we can only move forward
        m.x1 = Var(m.t,bounds=(-10,10), initialize=0) #vy
        m.x2 = Var(m.t,bounds=(-0.5*3.14,0.5*3.14), initialize=0) #wz
        m.x3 = Var(m.t, bounds=(-0.3*3.1416,0.3*3.1416)) #epsi
        m.x4 = Var(m.t, bounds=(0,m.sf), initialize=0) #s
        m.x5 = Var(m.t, bounds=(-TrackHalfW,TrackHalfW), initialize=0) #ey
    
    elif vmodel == 'Genesis':
        mass  = 2303.1
        lf = 1.5213
        lr = 1.4987
        Iz = 5520.1
        Cr = 13.4851e4*2
        Cf = 7.6419e4*2
        
        Df = 0.8 * mass * 9.81 / 2.0
        Dr = 0.8 * mass * 9.81 / 2.0
        Br = 1.0   
        Bf = 1.0
        
        model = m = ConcreteModel()
        m.sf = Param(initialize = TrackLength)
        m.t = RangeSet(0, N) 
        m.tnotN = m.t - [ N ]
        m.u0 = Var(m.tnotN, bounds=(-4,4), initialize=0) # 4m/s2, internet
        m.u1 = Var(m.tnotN, bounds=(-0.5,0.5), initialize=0) # will keep the same, but unclear
        #m.alpha_f = Var(m.s, bounds=(-2,2), initialize=0)
        #m.alpha_r = Var(m.s, bounds=(-0.5,0.5), initialize=0)
        m.alpha_f = Var(m.t, initialize=0)
        m.alpha_r = Var(m.t, initialize=0)
        m.Fyf = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.Fyr = Var(m.t,bounds=(-mass*9.8,mass*9.8), initialize=0)
        m.x0 = Var(m.t,bounds=(0,67), initialize=0.01) #150mph, according to internet
        m.x1 = Var(m.t, initialize=0) #??
        m.x2 = Var(m.t, initialize=0)
        #m.x3 = Var(m.s, bounds=(-0.1*3.1416,0.1*3.1416))
        m.x3 = Var(m.t, bounds=(-0.3*3.1416,0.3*3.1416))
        m.x4 = Var(m.t, bounds=(0,20000), initialize=0)
        m.x5 = Var(m.t, bounds=(-TrackHalfW,TrackHalfW), initialize=0)
    
       
    #sideslip and lateral force
    def _alphafc(m, t):
        return m.alpha_f[t] == m.u1[t] - atan((m.x1[t] + lf * m.x2[t])/ (m.x0[t]))
    m.c4 = Constraint(m.tnotN, rule=_alphafc)
    def _alpharc(m, t):
        return m.alpha_r[t] == -atan((m.x1[t] - lr * m.x2[t])/ (m.x0[t]))
    m.c3 = Constraint(m.tnotN, rule=_alpharc)
    def _Fyfc(m, t):
        return m.Fyf[t] ==  2 * Df * sin( Cf * atan(Bf * m.alpha_f[t]) )
    m.c2 = Constraint(m.tnotN, rule=_Fyfc)
    def _Fyrc(m, t):
        return m.Fyr[t] ==  2 * Dr * sin( Cr * atan(Br * m.alpha_r[t]) )
    m.c1 = Constraint(m.tnotN, rule=_Fyrc)
        
    def _x0(m,t): #vx
        return m.x0[t+1] == m.x0[t] + dt * (m.u0[t] - 1 / mass * m.Fyf[t] * sin(m.u1[t]) + m.x2[t]*m.x1[t])
    m.x0constraint = Constraint(m.tnotN, rule=_x0)    
    def _x1(m,t): #vy
        return m.x1[t+1] == m.x1[t] + dt * (1 / mass * (m.Fyf[t] * cos(m.u1[t]) + m.Fyr[t]) - m.x2[t]*m.x0[t])
    m.x1constraint = Constraint(m.tnotN, rule =_x1)    
    def _x2(m,t): # wz
        return m.x2[t+1] == m.x2[t] + dt * (1 / Iz * (lf * m.Fyf[t] * cos(m.u1[t]) - lr * m.Fyr[t]))
    m.x2constraint = Constraint(m.tnotN, rule=_x2)    
    def _x3(m,t): #epsi
        cur = cv_pred[t]
        #cur = map.getCurvature(m.x4[t])
        return m.x3[t+1] == m.x3[t] + dt * (m.x2[t] - (m.x0[t] * cos(m.x3[t]) - m.x1[t]*sin(m.x3[t])) / (1 - cur*m.x5[t]) * cur)
    m.x3constraint = Constraint(m.tnotN, rule=_x3)    
    def _x4(m,t): #s
        #cur = map.getCurvature(m.x4[t])
        cur = cv_pred[t]
        return m.x4[t+1] == m.x4[t] + dt * ( (m.x0[t] * cos(m.x3[t]) - m.x1[t]*sin(m.x3[t])) / (1 - cur * m.x5[t]) )
    m.x4constraint = Constraint(m.tnotN, rule=_x4)    
    def _x5(m,t): #ey
        return m.x5[t+1] == m.x5[t] + dt * (m.x0[t] * sin(m.x3[t]) + m.x1[t]*cos(m.x3[t]))
    m.x5constraint = Constraint(m.tnotN, rule=_x5)        

    def _init(m):
        yield m.x0[0] == x[0]
        yield m.x1[0] == x[1]
        yield m.x2[0] == x[2] 
        yield m.x3[0] == x[3]
        yield m.x4[0] == x[4]
        yield m.x5[0] == x[5]
    m.init_conditions = ConstraintList(rule=_init)
    
    
    def _safetySet(m):
        w = clf.coef_[0]
        a = -w
        
        # terminal constraint - safety set (only use on the last state)
        # transformed = scaler.scale_*(np.delete(safe_pts,4,1) - scaler.mean_)/(scaler.var_)

        # np.matmul(testpts,a) - clf.intercept_[0] / clf.intercept_scaling  <= 0
        # if yy is negative, we get a label of 1 ("Feasible")
        
        yield (a[0] * scaler.scale_[0] * (m.x0[N] - scaler.mean_[0])/scaler.var_[0]) + (a[1] * scaler.scale_[1] * (m.x1[N] - scaler.mean_[1])/scaler.var_[1]) + (a[2] * scaler.scale_[2] * (m.x2[N] - scaler.mean_[2])/scaler.var_[2]) + (a[3] * scaler.scale_[3] * (m.x3[N] - scaler.mean_[3])/scaler.var_[3]) + (a[4] * scaler.scale_[4] * (m.x5[N] - scaler.mean_[4])/scaler.var_[4]) - clf.intercept_[0] <= 0
    
    if beta:
        m.safety_constraint = ConstraintList(rule=_safetySet)
    
    
    # Objective function - should minimize deviation from state to center of corresponding strategy set
    # can we get a list of indices of set_list that are empty? 
    # index_list = [f for f,b in enumerate(set_list) if b!=[] and f < N]
    index_list = [f for f,b in enumerate(set_list) if f < N and not np.array_equal(b, np.empty([0,]))]
    
    #m.obj = Objective(expr = du_weight * sum((m.u1[j] - m.u1[j-1])**2 for j in m.t_u) + sum((set_list[i][0] - m.x4[i])**2 + (set_list[i][2] - m.x5[i])**2 for i in index_list), sense=minimize)
    m.track_obj = sum((set_list[i][0] - m.x4[i+1])**2 + (set_list[i][1] - m.x5[i+1])**2 for i in index_list)
    m.obj = Objective(expr =  m.track_obj, sense=minimize) 
    
    
           
    # logging.getLogger('pyomo.core').setLevel(logging.ERROR)    
    solver = SolverFactory('ipopt')
    solver.options["print_level"] = 1
    # log_infeasible_constraints(model)
       
    try:
        results = solver.solve(m,tee=False)
        if results.solver.status == SolverStatus.ok:
            solver_flag = True
        else:
            solver_flag = False
            cv_pred
            return 0, 0, solver_flag
    except:
        print('weird solver problem')
        solver_flag = False
        cv_pred
        return 0, 0, solver_flag
    
    VX = value(m.x0[0])
    VY = value(m.x1[0])
    WZ = value(m.x2[0])
    EPSI = value(m.x3[0])
    S = value(m.x4[0])
    EY = value(m.x5[0])
    
    for t in range(1,N+1):
        VX = np.hstack((VX,(value(m.x0[t]))))
        VY = np.hstack((VY,(value(m.x1[t]))))
        WZ = np.hstack((WZ,(value(m.x2[t]))))
        EPSI = np.hstack((EPSI,(value(m.x3[t]))))
        S = np.hstack((S,(value(m.x4[t]))))
        EY = np.hstack((EY,(value(m.x5[t]))))
        
    x_pred = np.vstack((VX, VY, WZ, EPSI, S, EY))
    
    A = value(m.u0[0])
    DELTA = value(m.u1[0])
    
    for t in range(1,N):
        A = np.hstack((A, (value(m.u0[t]))))
        DELTA = np.hstack((DELTA, (value(m.u1[t]))))
 
    u_pred = np.vstack((A, DELTA))
    
    return x_pred, u_pred, solver_flag

def evaluateNetStrategy(x_state, map, nn_input, net):
    
    # convert nn_input into torch thing
    nn_input = torch.from_numpy(nn_input[0]).double()
    
    # evaluate GP, and get confidence measure
    preds = net(nn_input)
    est_s = preds.detach().numpy()[0]
    est_ey = preds.detach().numpy()[1]
        
    # for visualization purposes, can we plot where we think we should be? TO DO TODAY
    est_s += x_state[4] # add delta to current s
    
    return est_s, est_ey


#%% Neural network training

torch.manual_seed(1)    # reproducible

# torch can only train on Variable, so convert them to Variable
trainD = Variable(torch.from_numpy(TrainD).double())
trainL = Variable(torch.from_numpy(TrainL).double())

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(17, 45),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(45, 100),
        torch.nn.Linear(100, 45),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(45, 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 2),
    )

net.double()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 1000
EPOCH = 150

torch_dataset = Data.TensorDataset(trainD, trainL)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

# start training
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step == 1:
            # plot and show learning process
            print('Epoch: %d - Loss: %.4f' % (epoch, loss.data.numpy()))

# for saving
# pickle.dump(net,open('neural_net.pkl','wb'))

#%% Start the control loop

track_length = map.TrackLength
#map.getGlobalPosition(1,2,3)
# initialize the vehicle state
# safety-mpc parameters
gamma = 0.1
vt = 5
vx0 = vx_interp(pframe, 0)
ey0 = ey_interp(pframe,0)
x_state = np.array([vt*0.8, 0, 0, 0, 0, 0]) # vx, vy, wz, epsi, s, ey
x_state = np.array([8, 0, 0, 0, 0, ey0]) # vx, vy, wz, epsi, s, ey

# initialize vectors to save vehicle state and inputs
x_closedloop = np.reshape(x_state, (6, 1))
u_closedloop = np.array([[0],[0]])
x_pred = x_closedloop
x_pred_stored = np.empty((1,21))
u_pred = np.array([[0,0],[0,0]])


# initialize SetList to be empty - how should we represent these sets in Python? collections of constraints? vectors w/ max/mins? 
N_mpc = 20
N = N_mpc

model = 'BARC' # (or BARC)

# how can we keep the set list to always contain N sets? could we instantiate an empty set that gets added instead?
set_list = []
for i in range(N):
    set_list.append([])

# while the predicted s-state of the vehicle is less than track_length:
while x_pred[4, -1] < track_length:
    
    nn_input = get_nn_input(x_state, map, T, ds)
    
    # evaluate neural network
    est_s, est_ey = evaluateNetStrategy(x_state, map, nn_input, net)
    
    # add to the list
    set_list.pop(0)
    set_list.append([est_s, est_ey])
    
    # solve the tracking MPC (just try to be as close as possible to the predictions)
    x_pred, u_pred, solver_status = strategy_NN(x_state, map, model, dt, N_mpc, set_list, clf, scaler, beta)
    
    if solver_status == False:
        x_pred, u_pred, solver_status = lane_track_MPC(x_state, map, dt, vt, N_mpc, gamma, model)
        if solver_status == False:
            x_pred, u_pred, solver_status = lane_track_MPC(x_state, map, dt, vt, int(N_mpc/4), gamma, model)
            if solver_status == False:
                x_pred, u_pred, solver_status = lane_track_MPC(x_state, map, dt, vt, 1, gamma, model)
                if solver_status == False:
                    input('shortened horizon MPC was infeasible?')
    
    x_pred = np.hstack((x_pred, np.zeros((6, N_mpc + 1 - np.shape(x_pred)[1]))))
    
    x_pred_stored = np.vstack((x_pred_stored, x_pred))
    
    u = u_pred[:,0]
    u_closedloop = np.hstack((u_closedloop, np.reshape(u,(2,1))))
    
    # apply input to system 
    x_state = vehicle_model(x_state, u, dt, map, model)
    # round to avoid numerical disasters. not enough to round, needs to be below 1.57 (mainly x2 issue)
    eps = 1e-4
    x_state = np.array([round(i,3) for i in x_state])
    while abs(x_state[2])>=1.569:
        x_state[2] -= eps*sign(x_state[2])
    while abs(x_state[5])>=0.8:
        x_state[5] -= eps*sign(x_state[5])
    
    # save important quantities for next round (predicted inputs, predicted state)
    x_closedloop = np.hstack((x_closedloop, np.reshape(x_state,(6,1))))
       
    plot_closed_loop(map,x_closedloop,x_pred = x_pred[:,:N+1], offst=20) #20
    #plot_closed_loop(map,x_closedloop, offst=3)
    plt.show()
    
    print(x_pred[4,-1])
    

x_closedloop = np.hstack((x_closedloop, x_pred))
plot_closed_loop(map,x_closedloop,offst=1000)
    
# get rid of the first row of x_pred_stored
x_pred_stored = x_pred_stored[1:,:]
hpl_time = np.shape(x_closedloop)[1]*dt

# can we make a plot that compares vx0, ey as a function of s? for raceline vs. hpl (today, and then stop)
plt.figure()
plt.plot(x_closedloop[4,:], x_closedloop[5,:],'r',label='HPL')
plt.plot(x_closedloop[4,:], ey_interp(pframe, x_closedloop[4,:]),'b',label='Raceline')
plt.legend()
plt.xlabel('s')
plt.ylabel('e_y')
plt.show()

plt.figure()
plt.plot(x_closedloop[4,:], x_closedloop[0,:],'r',label='HPL')
plt.plot(x_closedloop[4,:], vx_interp(pframe, x_closedloop[4,:]),'b',label='Raceline')
plt.legend()
plt.xlabel('s')
plt.ylabel('v_x')
plt.show()


#%% Plot the closed loop in nice ways
# First, for each track, plot left/right the colormaps

testfile = 'Tracks/US.pkl'
[raceline_X,raceline_Y,Cur, map, race_time, pframe] = plotFromFile(testfile, lineflag=True)

#%%
import matplotlib.cm as cm

# to fix: how can we change the size of the figure?
# i think this formatting will depend on the shape of the plots. save this for last. 

# load closed loop
x_cl = pickle.load(open('nn_AE_x_closeloop.pkl','rb'))

X = []
Y = []
try:
    if len(x_cl):
        for i in range(0, np.shape(x_cl)[1]):
            [x,y] = map.getGlobalPosition(x_cl[4,i], x_cl[5,i],0)
            X = np.append(X,x)
            Y = np.append(Y,y)
except 'Value Error': 
    if bool(x_cl.any()):
        for i in range(0, np.shape(x_cl)[1]):
            [x,y] = map.getGlobalPosition(x_cl[4,i], x_cl[5,i],0)
            X = np.append(X,x)
            Y = np.append(Y,y)    
            
            
fig = plt.figure()
fig.set_size_inches(7.5, 5)


plt.subplot(2, 1, 1)


map.plot_map()    
plt.scatter(X,Y,1,c=x_cl[0,:],cmap = cm.turbo, vmin=1, vmax=10, zorder=2)
plt.colorbar(label='$v_x$ [m/s]', pad=-0.4)
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.axis('scaled')
plt.title('AE Track with HPL (NN) Control')
#plt.plot(X, Y, '-r')


plt.subplot(2, 1, 2)
map.plot_map()  
plt.scatter(raceline_X, raceline_Y, 1, c=pframe['x0'].values, cmap = cm.turbo, vmin=1, vmax=10, zorder=2)
plt.colorbar(label='$v_x$ [m/s]', pad=-0.4)
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
#plt.xlim([-90, 80])
plt.axis('scaled')
plt.title('AE Track Raceline')

plt.tight_layout()

fig.savefig('nn_AE_velcomp.eps', dpi=200)


#%% Plot vx/ey
import matplotlib.pyplot as plt

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#x_cl = pickle.load(open('x_closedloop.pkl','rb'))


fig = plt.figure()
fig.set_size_inches(7, 8)
# can we change the font size?
# can we change the background of the legends label? location and color?

plt.subplot(2, 1, 1)
plt.plot(x_cl[4,:], x_cl[5,:],'r',label='HPL')
plt.plot(x_cl[4,:], ey_interp(pframe, x_cl[4,:]),'b',label='Raceline')
plt.xlim([0, x_cl[4,-1]])
plt.ylim([-1,1])
#plt.legend(loc='best')
plt.xlabel('$s~ [m]$')
plt.ylabel('$e_y~ [m]$')
plt.title('AE (NN) Track Trajectory Comparison', pad=30)

plt.subplot(2,1,2)
plt.plot(x_cl[4,:], x_cl[0,:],'r',label='HPL (NN)')
plt.plot(x_cl[4,:], vx_interp(pframe, x_cl[4,:]),'b',label='Raceline')
plt.xlim([0, x_cl[4,-1]])
plt.ylim([0, 11])
plt.legend(loc='lower right')
plt.xlabel('$s~[m]$')
plt.ylabel('$v_x~ [m/s]$')

plt.tight_layout()

# font = {'family' : 'latex',
#         'size'   : 20}

# matplotlib.rc('font', **font)


plt.show()
fig.savefig('nn_AE_trajcomp.eps', dpi=200)


#%% Zooming in on a particular part of the trackmap.plot_map()
# maybe we can make this a subplot with three figures?
#testfile = 'Tracks/US.pkl'
#[raceline_X,raceline_Y,Cur, map, race_time, pframe] = plotFromFile(testfile, lineflag=True)

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

x_cl = pickle.load(open('x_closedloop.pkl','rb'))
X = []
Y = []
try:
    if len(x_cl):
        for i in range(0, np.shape(x_cl)[1]):
            [x,y] = map.getGlobalPosition(x_cl[4,i], x_cl[5,i],0)
            X = np.append(X,x)
            Y = np.append(Y,y)
except 'Value Error': 
    if bool(x_cl.any()):
        for i in range(0, np.shape(x_cl)[1]):
            [x,y] = map.getGlobalPosition(x_cl[4,i], x_cl[5,i],0)
            X = np.append(X,x)
            Y = np.append(Y,y)    
            
  #%%          

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig = plt.figure()

pts = 2000
span_s = np.linspace(0,map.TrackLength, int(pts))
middle = 0
outer = map.width 
inner = -outer
middle_data = np.array([map.getGlobalPosition(s,middle,0) for s in span_s])
outer_data  = np.array([map.getGlobalPosition(s,outer,0) for s in span_s])
inner_data  = np.array([map.getGlobalPosition(s,inner,0) for s in span_s])


plt.plot(outer_data[:,0], outer_data[:,1],'k',linewidth=3)
plt.plot(inner_data[:,0], inner_data[:,1],'k',linewidth=3)
plt.plot(X,Y,'r',linewidth=3)
plt.axis('scaled')
plt.axis([0,50, -170, 60]) 
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title('AE Track')

#fig.savefig('AE_ex2.eps', dpi=200)

#fig.canvas.draw()
# plt.subplot(3, 1, 2)
# map.plot_map()    
# plt.plot(X,Y,'r')
# plt.xlim([0, 50])
# plt.ylim([0,50])
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.title('AE Track Trajectory Comparison')

# plt.subplot(3, 1, 3)
# map.plot_map()    
# plt.plot(X,Y,'r')
# plt.xlim([0, 50])
# plt.ylim([0,50])
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.title('AE Track Trajectory Comparison')

#plt.tight_layout()
