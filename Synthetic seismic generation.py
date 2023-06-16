# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:37:26 2022

@author: pli3
"""

from Inversion import *
from RockPhysics import *
import numpy as np
import matplotlib.pyplot as plt  # plotting
from Geostats import *
import scipy.io as sio

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cude:0'if torch.cuda.is_available() else 'cpu')

# Rock physics parameters

# solid phase
Kmat = 30
Gmat = 60
Rho_sol = 2.6
Vp_sol = 6

# solid phase (quartz and clay)
Kclay = 21
Kquartz = 33
Gclay = 15
Gquartz = 36
Rhoclay = 2.45
Rhoquartz = 2.65

# fluid phase
Kw = 2.5
Ko = 0.7
Rho_w = 1.03
Rho_o = 0.7
Rho_fl = 1
Vp_fl = 1.5
Sw = 0.8

# fluid phase (water and gas)
Kwater = 2.25
Kgas = 0.1
Rhowater = 1.05
Rhogas = 0.1
patchy = 0

# granular media theory parameters
critporo = 0.4
coordnum = 7
press = 0.04

# Seismic parameter
freq = 45
dt = 0.001
ntw = 64
err_frac= 0.1

# reflection angles: 
# Post stack
theta_poststack = np.array([0])

# prestack
theta_prestack = [15, 30, 45]


# Loading 1D dataset

# dataset_1d = sio.loadmat('1DdataWell.mat')   

# Depth_1d = dataset_1d['Depth']
# Time_1d = dataset_1d['Time']

# Phi_1d = dataset_1d['Phi']
# Sw_1d = dataset_1d['Sw']

# Vp_1d = dataset_1d['Vp']
# Vs_1d = dataset_1d['Vs']
# Rho_1d = dataset_1d['Rho']

# Loading 2D dataset
dataset_2D = sio.loadmat('Goliat2D_Torstein.mat') 
# Depth, trace, phi, sw, Vclay, Vp, Vs, Rho,

Depth_2D  = dataset_2D['depth']
Trace_2D  = dataset_2D['trace']

Phi_2D  = dataset_2D['Phi']
Sw_2D  = dataset_2D['Sw']
Vclay_2D  = dataset_2D['Vclay']

Vp_2D  = dataset_2D['Vp']
Vs_2D  = dataset_2D['Vs']
Rho_2D  = dataset_2D['Rho']

ntr = Phi_2D.shape[1]
ns = Phi_2D.shape[0]
t0 = 1.8


# Depth to time, assume t0 = 1.8, taking Phi as an example
# m in the shape of ns =m.shape[0], ntr = m.shape[1]

def DeptoTime(m,Depth,Vp,t0,dt):
    ns = m.shape[0]
    ntr = m.shape[1]     
    Time =np.zeros([ns-1,ntr],dtype = np.float64)
    
    for i in range(ntr):
      Time[:,i]= (t0 + np.cumsum(2*np.ediff1d(Depth)/(1000*(Vp[1:,i]+Vp[0:-1,i])/2)))
        
    TimeSeis = np.arange(np.max(Time[0,:]),np.min(Time[-1,:]),dt)
    TimeSeis = np.matlib.repmat(TimeSeis.reshape(len(TimeSeis),1),1,ntr)
    Time = np.concatenate((TimeSeis-dt/2,TimeSeis[-1:]+dt/2),axis=0)
    
    t1 = TimeSeis[0,0]-dt/2
    t2 = TimeSeis[-1,0]+dt/2
    t = np.arange(t1,t2,(t2-t1)/ns)
    
    mTime =np.array([[0]*ntr]*(Time.shape[0]),dtype = np.float64)

    for i in range(ntr):
        mTime[:,i] = np.interp(Time[:,i],t,m[:,i])

    return mTime

PhiTime = DeptoTime(Phi_2D,Depth_2D,Vp_2D,1.8,0.001)
SwTime = DeptoTime(Sw_2D,Depth_2D,Vp_2D,1.8,0.001)
VclayTime = DeptoTime(Vclay_2D,Depth_2D,Vp_2D, 1.8, 0.001)

# 2d array to 3D array
# a: np array
def Addaxis(a):
    from numpy import newaxis
    b = a[:, newaxis,:]      # newaxis can be anywhere
    return b

# example: aa = Addaxis(df.to_numpy())


def Normalize(x, mean_val, std_val):

    n = (x - mean_val) / std_val

    return n


def Denormalize(x, mean_val, std_val):
    dn = x * std_val + mean_val

    return dn


# Generating 2D grid for the use of SGS from SeReMpy
# (Xcoords, dcoords, dz, zmean, zvar, l, krigtype, krig)
def GenMeasureD(m,nt,ns):
    """
    Generate dcoordinate and dvalue for SGS
    """
    trace_idx = np.arange(nt).reshape(1,-1)
    trace_idxmat = np.tile(trace_idx,(ns,1))   # Construct an array by repeating A the number of times given by n.
    sample_idx = np.arange(ns).reshape(-1,1)
    sample_idxmat = np.tile(sample_idx,(1,nt))
    dx = np.reshape(trace_idxmat,-1)
    dy = np.reshape(sample_idxmat,-1)
    dcoords = np.transpose(np.vstack([dx.reshape(-1), dy.reshape(-1)]))
    dz = m.reshape(-1,1)
    
    return dcoords, dz

# dcoords, dz = GenMeasureD(Snear,85,153)


def EnlargeGrid(nt,ns,tlarge,slarge):
    
    X_trace = np.linspace(0,nt-1,(nt-1)*tlarge+1).reshape(1,-1)   # trace enlarge tlarge times
    X_trace_idxmat = np.tile(X_trace,(ns*slarge,1))
    X = np.reshape(X_trace_idxmat,-1)
    
    Y_sample = np.linspace(0,ns-1,ns*slarge).reshape(-1,1)   # sample enlarge slarge times
    Y_sample_mat= np.tile(Y_sample,(1,(nt-1)*tlarge+1))
    Y = np.reshape(Y_sample_mat,-1)
    Xcoords = np.transpose(np.vstack([X.reshape(-1), Y.reshape(-1)]))
    
    return Xcoords,X_trace.shape[1],Y_sample.shape[0]

# Xcoords, X_trace_num, Y_sample_num= EnlargeGrid(nt, ns,tlarge,slarge)

# using sgs to generate high resolution synthetic porosity
def ApplySGS(m,tlarge,slarge):
    """
    Applying SGS for enlargement
    
    Parameters
    ----------
    m : 2D array
        Shape[0] should be the samples.
        Shape[1] should be the traces
    tlarge : int
        the number of enlargement in trace direction.
    slarge : int
        the number of enlargement in sample direction.

    Returns
    -------
    sgs : 2D array
        the sgs results of 2D section.

    """
    
    ns = m.shape[0]  #number of samples
    nt = m.shape[1]   #number of traces
    
    dcoords, dz = GenMeasureD(m,nt,ns)
    
    xmean = np.mean(dz)
    xvar = np.var(dz)
    xstd = np.std(dz)
    l = 1
    krigtype = 'gau'
    krig = 1
        
    d = Normalize(dz, xmean, xstd)
    
    xcoords,X_trace_num,Y_sample_num = EnlargeGrid(nt, ns, tlarge, slarge)
    
    sim = SeqGaussianSimulation(xcoords, dcoords, d, xmean, xvar, l, krigtype, krig)
    
    d_out = Denormalize(sim, xmean, xstd)
    sgs = d_out.reshape(Y_sample_num,X_trace_num)       # column first, then row
    
    return sgs

# Example
# Phi_1var = ApplySGS(PhiTime, 4, 4)


# Synthetic 2D seismic with only one (Phi,in time domain) variable, and using softsand model and 0 incident angle

def SynSeisLinearVpZeroIncident(Phi,a,b,Rho_sol,Rho_fl,freq,dt,ntw,theta):
    
    ns = Phi.shape[0]
    ntr = Phi.shape[1]
    # Roch physics model
    Rho_syn = DensityModel(Phi, Rho_sol, Rho_fl)
    Vp_syn = a*Phi +b
    
    # Seismic model
    w, _ = RickerWavelet(freq, dt, ntw)
    
    Seis_syn = np.zeros((ns - 1, ntr))
    
    for i in range(ntr):
        Seis = SeismicModelZeroincidentAngle(Vp_syn[:, i].reshape(-1, 1), Rho_syn[:, i].reshape(-1, 1), theta, w)
        err = np.sqrt(err_frac * np.var(Seis.flatten())) * np.random.randn(len(Seis.flatten()))
        Seis_syn[:, i] = Seis.flatten() + err
    
    return Seis_syn

# Example:
# seis =SynSeisLinearVpZeroIncident(PhiTime,a=-8,b=8,Rho_sol=Rho_sol,Rho_fl=Rho_fl,freq=freq,dt=dt,ntw=ntw,theta=theta_poststack)

def SynSeisRaymerZeroIncident(Phi,Vp_sol,Vp_fl,Rho_sol,Rho_fl,freq, dt, ntw,theta):
    
    ns = Phi.shape[0]
    ntr = Phi.shape[1]
  
    # Roch physics model
    Rho_syn = DensityModel(Phi, Rho_sol, Rho_fl)
    Vp_syn = RaymerModel(Phi,Vp_sol,Vp_fl)
    
    # Seismic model
    w, _ = RickerWavelet(freq, dt, ntw)
    
    Seis_syn = np.zeros((ns - 1, ntr))
    
    for i in range(ntr):
        Seis = SeismicModelZeroincidentAngle(Vp_syn[:, i].reshape(-1, 1), Rho_syn[:, i].reshape(-1, 1), theta, w)
        err = np.sqrt(err_frac * np.var(Seis.flatten())) * np.random.randn(len(Seis.flatten()))
        Seis_syn[:, i] = Seis.flatten() + err
    
    return Seis_syn

# Example
# seis = SynSeisRaymerZeroIncident(PhiTime, Vp_sol=Vp_sol,Vp_fl=Vp_fl,Rho_sol=Rho_sol,Rho_fl=Rho_fl,freq=freq, dt=dt, ntw=ntw,theta=theta_poststack)

def SynSeisSoftsandZeroincidentAngle2D(Phi,Kmat,Gmat,Rho_sol,Kw,Ko,Rho_fl,Sw,critporo,coordnum,press,freq,dt,ntw,theta):
        
    ns = Phi.shape[0]
    ntr = Phi.shape[1]
    
    Kfl = Sw * Kw + (1 - Sw) * Ko
    
    Rho_syn = DensityModel(Phi, Rho_sol, Rho_fl)
    Vp_syn, _ = SoftsandModel(Phi, Rho_syn, Kmat, Gmat, Kfl, critporo, coordnum, press)
    
    w, _ = RickerWavelet(freq, dt, ntw)
    
    Seis_syn = np.zeros((ns - 1, ntr))
    
    for i in range(ntr):
        Seis = SeismicModelZeroincidentAngle(Vp_syn[:, i].reshape(-1, 1), Rho_syn[:, i].reshape(-1, 1), theta, w)
        err = np.sqrt(err_frac * np.var(Seis.flatten())) * np.random.randn(len(Seis.flatten()))
        Seis_syn[:, i] = Seis.flatten() + err
    
    return Seis_syn

# Example
# seis = SynSeisSoftsandZeroincidentAngle2D(PhiTime,Kmat=Kmat,Gmat=Gmat,Rho_sol=Rho_sol,Kw=Kw,Ko=Ko,Rho_fl=Rho_fl,Sw=Sw,critporo=critporo,coordnum=coordnum,press=press,freq=freq,dt=dt,ntw=ntw,theta=theta_poststack)


# based on granular media theory+softsand+ aki-richard approximation, the shape of Phi,Vclay,Sw, should be [ns,nt]
def SynSeisFullRockphysics2D(Phi,Vclay,Sw,Kclay,Kquartz,Gclay,Gquartz,Rhoclay,Rhoquartz,Kwater,Kgas,Rhowater,Rhogas,patchy,criticalporo,coordnumber,pressure,dt,freq,ntw,theta):   
   
    wavelet, _ = RickerWavelet(freq, dt, ntw)

    ## solid and fluid phases
    ns = Phi.shape[0]
    nt = Phi.shape[1]
    
    Kmat = np.zeros([ns,nt])
    Gmat = np.zeros([ns,nt])
    Rhomat =np.zeros([ns,nt])
    Kfl = np.zeros([ns,nt])
    Rhofl = np.zeros([ns,nt])
    
    for i in range(nt):
        Kmat[:,i], Gmat[:,i], Rhomat[:,i], Kfl[:,i], Rhofl[:,i] = MatrixFluidModel(np.array([Kclay, Kquartz]), np.array([Gclay, Gquartz]), np.array([Rhoclay,Rhoquartz]),np.array([Vclay[:,i], 1-Vclay[:,i]]).T, np.array([Kwater, Kgas]), np.array([Rhowater,Rhogas]), np.array([Sw[:,i],1-Sw[:,i]]).T, patchy)


    ## Density
    Rho = DensityModel(Phi, Rhomat, Rhofl)

    ## Soft sand model
    Vp, Vs = SoftsandModel(Phi, Rho, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure)

    ## Seismic
    ntheta = len(theta)
    Snear = np.zeros([ns-1,nt])
    Smid = np.zeros([ns-1,nt])
    Sfar = np.zeros([ns-1,nt])
    Seis_syn = np.zeros([ns-1,ntheta,nt])
    for i in range(nt):
        Seis = SeismicModelAkiRichard(Vp[:,i].reshape(-1, 1), Vs[:,i].reshape(-1, 1), Rho[:,i].reshape(-1, 1), theta, wavelet)
        err = np.sqrt(err_frac * np.var(Seis))*np.random.randn((ns-1)*ntheta)
        
        Seis = Seis + err.reshape(-1,1)

        Snear[:,i] = Seis[:ns-1].flatten() # ns-1 not include acorrding to python role
        Smid[:,i] = Seis[ns-1:2*(ns-1)].flatten() 
        Sfar[:,i] = Seis[2*(ns-1):].flatten() 
        
    Seis_syn = np.concatenate((Snear[:,None,:],Smid[:,None,:],Sfar[:,None,:]),axis=1)
    return Seis_syn

# Example
seis_pre = SynSeisFullRockphysics2D(PhiTime,VclayTime,SwTime,Kclay=Kclay,Kquartz=Kquartz,Gclay=Gclay,Gquartz=Gquartz,Rhoclay=Rhoclay,Rhoquartz=Rhoquartz,Kwater=Kwater,Kgas=Kgas,Rhowater=Rhowater,Rhogas=Rhogas,patchy=patchy,criticalporo=critporo,coordnumber=coordnum,pressure=press,dt=dt,freq=freq, ntw=ntw,theta=theta_prestack)


# SRN dB calculation
# measure the power P(x) of a signal x(n), P(x) = 1/N*(sum x(n)**2)
# SNR via power ratio, SNR = P_signal/P_noise
# SNR in dB, SNR_db = 10log10(P_signal/P_noise)

def SignalPower(d):
    return np.mean(d**2).astype('float64')

def SNR_db(d_noise,d_noise_free):
    S_pow = SignalPower(d_noise_free).astype('float64')
    N_pow = SignalPower(d_noise).astype('float64')
    SNR_dB = 10*np.log10(S_pow/(N_pow-S_pow))
    return SNR_dB

# Example
# snr = SNR_db(Seis_noise, Seis_noisefree)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Plotting
fs = 16
line = 0 # could be a arbitary line

# 1D plot
# fig1=plt.figure(figsize=(2.5,6)) 
# # plt.plot(seis[:,line],list(range(seis.shape[0])),c='k',label='Seismic')
# # plt.plot(PhiTime[:,line],list(range(PhiTime.shape[0])),c='k',label='Porosity')

# plt.gca().invert_yaxis()
# plt.grid()
# plt.legend(loc='upper left')
# fig1.tight_layout()
# plt.show()


# #  2D plot for post stack seismic 

# fig2 = plt.figure(figsize=(8,3))
# plt.imshow(seis, cmap='gray', aspect='auto',
#             interpolation='bilinear' , extent=[0,337,1.95363,1.80063] ) #Extent defines the left and right limits, and the bottom and top limits, e.g. extent=[horizontal_min,horizontal_max,vertical_min,vertical_max]
# plt.colorbar()
# # cor.ax.set_title('Sw')
# # plt.clim(vmin = 0, vmax=1)
# plt.xlabel('Traces')
# plt.ylabel('Time (s)')
# fig2.tight_layout()
# fig2.show()

# 1d prestack seismic trace
fig3=plt.figure(figsize=(8,8))
plt.subplot(1,3,1)
plt.plot(seis_pre[:,0,line],np.linspace(1.80063,1.95363,153),c='k',label='True')
plt.title("Near",fontsize = fs)
plt.ylabel('Time (s)',fontsize = fs)
plt.xlim([-0.3,0.3])
plt.ylim([1.8,1.95])
plt.gca().invert_yaxis()
plt.grid()
# plt.legend(loc='upper left')
plt.subplot(1,3,2)
plt.plot(seis_pre[:,1,line],np.linspace(1.80063,1.95363,153),c='k',label='True')
plt.title("Mid",fontsize = fs)
# plt.ylabel('Time (s)')
plt.xlim([-0.3,0.3])
plt.ylim([1.8,1.95])
plt.gca().invert_yaxis()
plt.xlabel('Seismic amplitude',fontsize = 20)
plt.grid()
plt.subplot(1,3,3)
plt.plot(seis_pre[:,2,line],np.linspace(1.80063,1.95363,153),c='k',label='True')
plt.title("Far",fontsize = fs)
# plt.ylabel('Time (s)')
plt.xlim([-0.3,0.3])
plt.ylim([1.8,1.95])
plt.gca().invert_yaxis()
plt.grid()
fig3.tight_layout()
fig3.show()

#  2D plot for pre stack seismic

fig4 = plt.figure(figsize=(8,8))
plt.subplot(3, 1, 1)
plt.imshow(seis_pre[:, 0], cmap='gray', aspect='auto',vmin=-0.3,
            vmax=0.3, extent=[0,337,1.95363,1.80063])
# plt.xlabel('Traces')
plt.ylabel('Time (s)',fontsize = 16)
plt.colorbar()
plt.title("Near",fontsize = 16)

plt.subplot(3, 1, 2)
plt.imshow(seis_pre[:, 1], cmap='gray', aspect='auto',vmin=-0.3,
            vmax=0.3,extent=[0,337,1.95363,1.80063])
# plt.xlabel('Traces')
plt.ylabel('Time (s)',fontsize = 16)
plt.title("Mid",fontsize = 16)
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(seis_pre[:, 2], cmap='gray', aspect='auto', vmin=-0.3,
            vmax=0.3, extent=[0, 337, 1.95363, 1.80063])
plt.xlabel('Traces',fontsize = 16)
plt.ylabel('Time (s)',fontsize = 16)
plt.colorbar()
plt.title("Far",fontsize = 16)
fig4.tight_layout()
fig4.show()

# fig4.savefig('Fig12b.tiff',dpi=400)