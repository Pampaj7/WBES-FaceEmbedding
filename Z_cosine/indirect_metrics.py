#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:12:16 2022

@author: sariyanide
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

li = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508] 
sdir = '/offline_data/face/synth/meshes/db1/'
sdir = '/offline_data/face/meshes/synthbbb_020/'
sdir = '/offline_data/face/meshes/synth_020/'
sdir = '/online_data/3Dfacebenchmark/synth_neutral/Rmeshes/BFM/p23470'
gdir = '/online_data/3Dfacebenchmark/synth_neutral/Gmeshes'
Nsubj = 99

Ms = {}
As = {}
Rs = {}

means = {}
meshes = {}

methods = ['3DIv2_050', 'Deep3DFace_050', 'Deep3DFace_001', 'INORig_050', 'SynergyNet_001', '3DDFAv2_050', '3DIv2_001',  'ground_truth']

valid_subjs = []

for subj_id in range(0,Nsubj):
    print(subj_id)
    isvalid = True
    for method in methods[:-1]:
        if not os.path.exists('%s/%s/id%04d.txt' % (sdir, method, subj_id)):
            isvalid = False
        else:
            if np.sum(np.isnan(np.loadtxt('%s/%s/id%04d.txt' % (sdir, method, subj_id)))):
                isvalid = False
                
    if not isvalid:
        continue
    
    valid_subjs.append(subj_id)
        


for method in methods:
    print(method)

    meshes[method] = {}
    if method != 'ground_truth':
        sdir_ = sdir + '/' + method
    else:
        sdir_ = gdir
        
    for subj_id in valid_subjs:
        meshes[method][subj_id] = np.loadtxt('%s/id%04d.txt' % (sdir_,  subj_id))#[li,:]
    
    means[method] = None
    for subj_id in valid_subjs:
        if means[method] is None:
            means[method] = copy.deepcopy(meshes[method][subj_id])
        else:
            means[method] += meshes[method][subj_id]
    
    means[method] /= len(meshes[method])


        
#%%
import scipy.stats

for method in methods[:-1]:

    perf = 0
    
    rs = []
    gs = []
    for s1 in valid_subjs:
        r1 = meshes[method][s1].flatten()
        g1 = meshes['ground_truth'][s1].flatten()
        for s2 in valid_subjs:
            if s2 <= s1:
                continue
            r2 = meshes[method][s2].flatten()
            g2 = meshes['ground_truth'][s2].flatten()
            
            theta_r = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
            theta_g = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
            
            if np.isnan(theta_r) or np.isnan(theta_g):
                continue
            rs.append(theta_r)
            gs.append(theta_g)
    
    print( '%s \t%.3f'  % (method, np.abs(scipy.stats.pearsonr(rs,gs)[0])))


#%%


for method in methods[:-1]:

    perf = 0
    
    rs = []
    gs = []
    for s1 in valid_subjs:
        r1 = meshes[method][s1].flatten()
        g1 = meshes['ground_truth'][s1].flatten()
        for s2 in valid_subjs:
            if s2 <= s1:
                continue

            r2 = meshes[method][s2].flatten()
            g2 = meshes['ground_truth'][s2].flatten()
            
            abs_r = np.linalg.norm(r1-r2)
            abs_g = np.linalg.norm(g1-g2)
            
            rs.append(abs_r)
            gs.append(abs_g)
            
    
    print( '%s \t%.3f'  % (method, np.abs(scipy.stats.pearsonr(rs,gs)[0])))
    # print(np.array(errs_a).T @ np.array(errs_b))


#%%

for method in methods[:-1]:

    perf = 0
    
    rs = []
    gs = []
    for s1 in valid_subjs:
        r1 = meshes[method][s1].flatten()
        g1 = meshes['ground_truth'][s1].flatten()
        for s2 in valid_subjs:
            if s2 <= s1:
                continue
            r2 = meshes[method][s2].flatten()
            g2 = meshes['ground_truth'][s2].flatten()
            
            # errs_a.append(np.linalg.norm(a-mua,2)*np.exp(1j*theta_a))

            theta_r = np.arccos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
            theta_g = np.arccos(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2)))
            
            abs_r = np.linalg.norm(r1-r2)
            abs_g = np.linalg.norm(g1-g2)
            
            rs.append(abs_r*np.exp(1j*theta_r))
            gs.append(abs_g*np.exp(1j*theta_g))
            
    # np.abs gives the magnitude
    print( '%s \t%.3f'  % (method, np.abs(scipy.stats.pearsonr(rs,gs)[0])))



    
    
    
