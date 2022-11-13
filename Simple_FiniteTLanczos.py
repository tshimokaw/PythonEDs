# coding:utf-8
from __future__ import print_function
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time
from matplotlib import pyplot

def ham_to_vec_nosz_conserv(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert):
    w = np.zeros(Nhilbert,dtype=float)

    for i in range(Nhilbert):  #loop for all basis state (from 0 to 2**N-1)
       ii = i                 #i-th basis
       for k in range(Nint):  #loop for all interaction (Nint = # of interaction J_ij)
        isite1 = list_isite1[k] # site i for J_ij S_i S_j
        isite2 = list_isite2[k] # site j for J_ij S_i S_j
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2     #up-up spin state only at i=isite1 and j=isite2 sites
        wght = 2.0*Jxx[k]   #The transverse J_ij
        diag = Jzz[k]        #The longitudinal J_ij
        ibit = ii & is12       #To check the spin config at i=isite1 and j=isite2 sites

        if (ibit==0 or ibit==is12):   #if i and j sites have up-up or down-down spin configs.
             w[i] += diag*v1[i]     #For S_i^z S_j^z term
        else:                        #if i and j sites have up-down or down-up spin configs.
            w[i] -= diag*v1[i]       #For S_i^z S_j^z term 
            iexchg = ii ^ is12       #Flip two spins and get to know the new basis number (decinal number)
            w[i] += wght*v1[iexchg]#For (S_i^+ S_j^- + S_i^- S_j^+) term 
    return w

def simple_FTL_nosz_conserv(R,M,Jxx,Jzz,list_isite1,list_isite2,N,Nint,seed0):

    Nhilbert = 2**N
    epsilons = np.zeros((R,M),dtype=float)
    v1psi = np.zeros((R,M),dtype=float)
    for r in range(R): # random sampling
        np.random.seed(seed=seed0+r)
               
        alphas = []                              #Diagonal parts of the trigonal matrix
        betas = [0.]                             #Off-diagonal parts of the trigonal matrix
        v1 = (1+1)*np.random.rand(Nhilbert)-1  #old Lanczos vector (real number vector)
        v1 /= np.linalg.norm(v1)                 #normalization
        v0 = np.zeros(Nhilbert, dtype=float)     #new Lanczos vector
        w  = np.zeros(Nhilbert, dtype=float)

        alpha = 0.
        beta = 0.
        pre_energy=0
    
        for iteration in range(0, M):
            w = ham_to_vec_nosz_conserv(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert)
            alpha = np.dot(v1,w)
            w = w -alpha*v1 -beta*v0
            v0 = np.copy(v1)
            beta = np.sqrt(np.dot(w,w))
            v1 = w/beta
            alphas.append(alpha)
            betas.append(beta)

        t_eigs,t_vecs = scipy.linalg.eigh_tridiagonal(alphas,betas[1:-1])
        epsilons[r,:]=t_eigs[:]/4 
        minene = min(t_eigs)/4
        v1psi[r,:]=t_vecs[0,:] #1st component of the eigenvector, <V_r | Psi_j^r>

    return epsilons,v1psi,minene

def calc_Tdep_ene(R,M,epsilons,v1psi,minene):
     
     TdepEne = np.zeros((2000,2),dtype=float)
     epsilons0 = epsilons - minene
     for T0 in range(1,2000): 
         T= T0*0.01
         beta = 1.0/T
         PartZ = 0.0
         ene = 0.0

         for r in range(R):
             for i0 in range(M):
                 PartZ += np.exp(-beta*epsilons0[r,i0])*abs(v1psi[r,i0])**2      
                 ene  +=  epsilons0[r,i0]*np.exp(-beta*epsilons0[r,i0])*abs(v1psi[r,i0])**2

         TdepEne[T0,:] =T, ene/PartZ+minene 

     return TdepEne

def calc_Tdep_C(R,M,epsilons,v1psi,minene,TdepEne):
 
    TdepC = np.zeros((2000,2),dtype=float)
    epsilons0 = epsilons - minene
    for T0 in range(1,2000):
        T = T0*0.01
        beta = 1.0/T
        PartZ = 0.0
        C = 0.0

        for r in range(R):
            for i0 in range(M):
                PartZ += np.exp(-beta*epsilons0[r,i0])*abs(v1psi[r,i0])**2      
                C  +=  (abs(epsilons[r,i0])**2)*np.exp(-beta*epsilons0[r,i0])*abs(v1psi[r,i0])**2

        TdepC[T0,:] = T, C/PartZ/T/T - abs(TdepEne[T0,1])**2/T/T 

    return TdepC


def make_lattice_chain(N,J1,J2):
    Jxx = []
    Jzz = []
    list_isite1 = []
    list_isite2 = []
    Nint = 0
    for i in range(N):
        site1 = i
        site2 = (i+1)%N
        site3 = (i+2)%N
#
        list_isite1.append(site1)
        list_isite2.append(site2)
        Jxx.append(J1)
        Jzz.append(J1)
        Nint += 1
#
        list_isite1.append(site1)
        list_isite2.append(site3)
        Jxx.append(J2)
        Jzz.append(J2)
        Nint += 1
    return Jxx, Jzz, list_isite1, list_isite2, Nint


def main(seed0):
    N=10
    J1=1.00
    J2=0.4
    R=10 # # of random sample
    M=50 # Lanczos itereation

    print("J1=",J1)
    print("J2=",J2)
    print("N=",N)
    print("R=",R)
    print("M=",M)

    Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice_chain(N,J1,J2)
    print (Jxx)
    print (Jzz)
    print (list_isite1)
    print (list_isite2)
    print("Nint=",Nint)

    start = time.time()
    epsilons, v1psi, minene = simple_FTL_nosz_conserv(R,M,Jxx,Jzz,list_isite1,list_isite2,N,Nint,seed0)
    TdepEne = calc_Tdep_ene(R,M,epsilons,v1psi,minene)
    TdepC = calc_Tdep_C(R,M,epsilons,v1psi,minene,TdepEne)
    TdepEne[:,1] = TdepEne[:,1]/N #per site
    TdepC[:,1] = TdepC[:,1]/N     #per site
    end = time.time()
    print (end - start)

    return TdepEne, TdepC 

if __name__ == "__main__":
    TdepEne = np.zeros((2000,2,3),dtype=float)
    TdepC = np.zeros((2000,2,3),dtype=float)

    # three different seeds
    TdepEne[:,:,0], TdepC[:,:,0] = main(12345)
    TdepEne[:,:,1], TdepC[:,:,1] = main(22345)
    TdepEne[:,:,2], TdepC[:,:,2] = main(32345)

    # standard error
    TdepEneErr = np.zeros((2000,3), dtype=float)
    TdepEneErr[:,0:2] = np.average(TdepEne,axis=2)[:,:]
    TdepEneErr[:,2] = (np.std(TdepEne,axis=2)/np.sqrt(3.0))[:,1]

    TdepCErr = np.zeros((2000,3), dtype=float)
    TdepCErr[:,0:2] = np.average(TdepC,axis=2)[:,:]
    TdepCErr[:,2] = (np.std(TdepC,axis=2)/np.sqrt(3.0))[:,1]      

    pyplot.errorbar(TdepEneErr[1:100,0], TdepEneErr[1:100,1], TdepEneErr[1:100,2])
    pyplot.errorbar(TdepCErr[1:100,0], TdepCErr[1:100,1], TdepCErr[1:100,2])
