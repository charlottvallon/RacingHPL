#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:40:47 2020

@author: vallon2
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np


def plotFromFile(testfile, lineflag = False, spacing = 0):
    
    db = pickle.load(open(testfile,"rb"))
    pframe= db['raceline']
    map = db['trackmap']
        
    # play back recorded inputs
    X=[0]
    Y=[pframe['x5'].values[0]]
    Cur=[0]
    EY=[pframe['x5'].values[0]]
    svec=list(sorted(pframe.index))
    # this gives us the X-Y coords to plot
    plt.figure()
    for j in range(1,len(svec)):
        sj=svec[j]
        ey=pframe['x5'].values[j]
        tmp = map.getGlobalPosition(sj, ey,0)
        cv = map.getCurvature(sj)
        EY.append(ey)
        Cur.append(cv)
        X.append(tmp[0])
        Y.append(tmp[1])
            
   # plot the track
    plt.figure() 
    if lineflag:
        plotTrajectory_newmap(map,X,Y)
    else:
        plotTrajectory_newmap(map,[],[])
        
    if spacing != 0:
        # also add markers along the plot every s points
        for j in np.arange(1,map.TrackLength,spacing):
            tmp = map.getGlobalPosition(j, 0, 0)
            plt.plot(tmp[0], tmp[1],'yo', markersize=5)
    plt.title(testfile)
    plt.gca().set_aspect('equal', adjustable='box')
                
    racetime = pframe['x4'][svec[-1]]
    return X, Y, Cur, map, racetime, pframe


def plotTrajectory_newmap(map, X,Y):

    map.plot_map()    
    plt.plot(X, Y, '-r')
    xmin, xmax, ymin, ymax = plt.axis()

    return [xmin, xmax, ymin, ymax]
    

def ey_interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    cur_ey = np.interp(new_index, df.index, df['x5'].values)
    return cur_ey


def vx_interp(df, new_index):
    cur_vx = np.interp(new_index, df.index, df['x0'].values)   
    return cur_vx


def t_interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    cur_t = np.interp(new_index, df.index, df['x4'].values)
    return cur_t

def get_s_from_t(df, new_index):
    s = np.interp(new_index, df['x4'].values, df.index)   
    return s

def get_t_from_s(df, new_index):
    t = np.interp(new_index, df.index, df['x4'].values)   
    return t


def plot_closed_loop(map,x_cl = [], offst=10, x_pred=[], ):
    # shape of closedloop is 6xSamples
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
    Xp = []
    Yp = []
    if len(x_pred):
        for i in x_pred.T:
            [x,y]=map.getGlobalPosition(i[4],i[5],0)
            Xp = np.append(Xp,x)
            Yp = np.append(Yp,y)
                
    plt.plot(Xp,Yp,'g')
    
    [xm, xx, ym, yx] = plotTrajectory_newmap(map, X,Y)    
    if offst == 1000:
        plt.axis('scaled')
    else:       
        plt.axis('scaled')
        plt.axis([X[-1]-offst, X[-1]+offst, Y[-1]-offst, Y[-1]+offst]) 
    

def vehicle_model(x, u, dt, map, model):
    # this function applies the chosen input to the discretized vehicle model 

    if model == 'BARC': 
        m  = 1.98 # mass of vehicle [kg]
        lf = 0.125 # distance from center of mass to front axle [m]
        lr = 0.125 # distance from center of mass to rear axle [m]
        Iz = 0.024 # presumably moment? [kg/m]
        Df = 0.8 * m * 9.81 / 2.0 # peak factor - is this a general formula? (maybe the 2 stays?)
        Cf = 1.25 # shape (a0)
        Bf = 1.0 # stiffness
        Dr = 0.8 * m * 9.81 / 2.0 
        Cr = 1.25
        Br = 1.0
        
    if model == 'Genesis':
        m  = 2303.1
        lf = 1.5213
        lr = 1.4987
        Iz = 5520.1
        Cr = 13.4851e4*2
        Cf = 7.6419e4*2
        
        Df = 0.8 * m * 9.81 / 2.0
        Dr = 0.8 * m * 9.81 / 2.0
        Br = 1.0   
        Bf = 1.0

    cur_x_next = np.zeros(x.shape[0])

    # Extract the value of the states
    delta = u[1]
    a     = u[0]

    vx    = x[0]
    vy    = x[1]
    wz    = x[2]
    epsi  = x[3]
    s     = x[4]
    ey    = x[5]

    alpha_f = delta - np.arctan2( vy + lf * wz, vx )
    alpha_r = - np.arctan2( vy - lr * wz , vx)

    # Compute lateral force at front and rear tire
    Fyf = 2 * Df * np.sin( Cf * np.arctan(Bf * alpha_f ) )
    Fyr = 2 * Dr * np.sin( Cr * np.arctan(Br * alpha_r ) )

    cur = map.getCurvature(s)
    cur_x_next[0] = vx   + dt * (a - 1 / m * Fyf * np.sin(delta) + wz*vy)
    cur_x_next[1] = vy   + dt * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
    cur_x_next[2] = wz   + dt * (1 / Iz *(lf * Fyf * np.cos(delta) - lr * Fyr) )
    cur_x_next[3] = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    cur_x_next[4] = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
    cur_x_next[5] = ey   + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))

    vx   = cur_x_next[0]
    vy   = cur_x_next[1]
    wz   = cur_x_next[2]
    epsi = cur_x_next[3]
    s    = cur_x_next[4]
    ey   = cur_x_next[5]

    return cur_x_next
    


        
    



