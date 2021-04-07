#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the paper describing the control framework:
#
#     [1] Charlott Vallon and Francesco Borrelli. "Data-driven hierarchical predictive learning in unknown
#         environments." In IEEE CASE (2020).
#
#
# Attibution Information: Code developed by Charlott Vallon
# (for clarifiactions and suggestions please write to charlottvallon@berkeley.edu).
#
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from hpl_functions import plot_closed_loop, plotFromFile, vx_interp, ey_interp, vehicle_model
from Track_new import *
from matplotlib.patches import Polygon
from hplStrategy import hplStrategy
from hplControl import hplControl


#%% Initialization 

# define the test environment from loaded pickle file, and load previously determined raceline
testfile = 'Tracks/US.pkl'
[raceline_X,raceline_Y, Cur, map, race_time, pframe] = plotFromFile(testfile, lineflag=True)

# MPC parameters
T = 15 # number of environment prediction samples --> training input will have length 21
ds = 2 # environment sample step (acts on s)
N_mpc = 20 # number of timesteps (of length dt) for the MPC horizon 
dt = 0.1 # MPC sample time (acts on t). determines how far out we take our training label and env. forecast. (N*dt)
fint = 0.5 # s counter which determines start of new training data
model = 'BARC' # vehicle model, BARC or Genesis

# hpl parameters
s_conf_thresh = 8 # confidence threshold for del_s prediction
ey_conf_thresh = 1.3 # confidence threshold for ey prediction

# safety-mpc parameters
gamma = 0.1 # cost function weight (tracking vt vs. centerline)
vt = 5 # speed for lane-keeping safety controller

# flag for retraining
retrain_flag = False

# flag for plotting sets
plotting_flag = False

# flag for whether to incorporate safety constraint by measuring accepted risk level (1 = use safety, 0 = ignore safety)
beta = False

# flag whether to retrain new GPs
retrain_flag = False

AeBeUsStrat = hplStrategy(T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, retrain_flag)
HPLMPC = hplControl(T, ds, N_mpc, dt, fint, s_conf_thresh, ey_conf_thresh, map, vt, model, gamma, beta)


#%% Control Loop

# initialize the vehicle state
vx0 = vx_interp(pframe, 0)
ey0 = ey_interp(pframe,0)
x_state = np.array([vt*0.8, 0, 0, 0, 0, 0]) # vx, vy, wz, epsi, s, ey
x_state = np.array([vx0, 0, 0, 0, 0, ey0]) # vx, vy, wz, epsi, s, ey

# initialize vectors to save vehicle state and inputs
x_closedloop = np.reshape(x_state, (6, 1))
u_closedloop = np.array([[0],[0]])
x_pred = x_closedloop
x_pred_stored = np.empty((1,21))
u_pred = np.array([[0,0],[0,0]])

# while the predicted s-state of the vehicle is less than track_length:
while x_pred[4, -1] < map.TrackLength:
    
    # evaluate GPs
    est_s, std_s, est_ey, std_ey, strategy_set, centers = AeBeUsStrat.evaluateStrategy(x_state)
    
    # evaluate control
    x_pred, u_pred = HPLMPC.solve(x_state, std_s, std_ey, centers)
    
    # store predicted state signals
    x_pred_stored = np.vstack((x_pred_stored, x_pred))
    
    # append applied input
    u = u_pred[:,0]
    u_closedloop = np.hstack((u_closedloop, np.reshape(u,(2,1))))
    
    # apply input to system 
    x_state = vehicle_model(x_state, u, dt, map, model)
    # round to avoid numerical disasters
    eps = 1e-4
    x_state = np.array([round(i,3) for i in x_state])
    while abs(x_state[2])>=1.569:
        x_state[2] -= eps*sign(x_state[2])
    while abs(x_state[5])>=0.8:
        x_state[5] -= eps*sign(x_state[5])
    
    # save important quantities for next round (predicted inputs, predicted state)
    x_closedloop = np.hstack((x_closedloop, np.reshape(x_state,(6,1))))
    
    # plot the closedloop thus far 
    fig, ax = plt.subplots(1)
    ax.add_patch(strategy_set)
    
    if plotting_flag:
        for st in HPLMPC.set_list:
            if st != []:
                rect_pts_xy = np.array([map.getGlobalPosition(st[0] - st[1], st[2] - st[3],0)])
                rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(map.getGlobalPosition(st[0] + st[1], st[2] - st[3], 0), (1,-1))))
                rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(map.getGlobalPosition(st[0] + st[1], st[2] + st[3], 0), (1,-1))))
                rect_pts_xy = np.vstack((rect_pts_xy, np.reshape(map.getGlobalPosition(st[0] - st[1], st[2] + st[3], 0), (1,-1))))
                ax.add_patch(Polygon(rect_pts_xy, True, color = 'g',alpha = 0.3))
       
    plot_closed_loop(map,x_closedloop,x_pred = x_pred[:,:HPLMPC.N+1], offst=20)
    plt.show()

x_closedloop = np.hstack((x_closedloop, x_pred))
hpl_time = np.shape(x_closedloop)[1]*dt
x_pred_stored = x_pred_stored[1:,:]


#%% Plotting closed-loop behavior

plot_closed_loop(map,x_closedloop,offst=1000)
    
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
