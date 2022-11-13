# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time
import matplotlib.pyplot as plt


def snoob(x):
    next = 0
    if(x>0):
        smallest = x & -(x)
        ripple = x + smallest
        ones = x ^ ripple
        ones = (ones >> 2) // smallest
        next = ripple | ones
    return next

def binomial(n,r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

def count_bit(n):
    count = 0
    while (n): 
        count += n & 1
        n >>= 1
    return count 

def init_parameters(N,Sz):
    Nup = N//2 + Sz
    Nhilbert = binomial(N,Nup)
    ihfbit = 1 << (N//2)
    irght = ihfbit-1
    ilft = ((1<<N)-1) ^ irght
    iup = (1<<(N-Nup))-1   #all up state from 0-th to (N-Nup)-th site
    return Nup, Nhilbert, ihfbit, irght, ilft, iup

def make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup):
    list_1 = np.zeros(Nhilbert,dtype=int)
    list_ja = np.zeros(ihfbit,dtype=int)
    list_jb = np.zeros(ihfbit,dtype=int)
    ii = iup
    ja = 0
    jb = 0
    ia_old = ii & irght
    ib_old = (ii & ilft) // ihfbit
    list_1[0] = ii
    list_ja[ia_old] = ja
    list_jb[ib_old] = jb
    ii = snoob(ii)
    for i in range(1,Nhilbert):
        ia = ii & irght
        ib = (ii & ilft) // ihfbit
        if (ib == ib_old):
            ja += 1
        else:
            jb += ja+1
            ja = 0
        list_1[i] = ii
        list_ja[ia] = ja
        list_jb[ib] = jb
        ia_old = ia
        ib_old = ib
        ii = snoob(ii)
    return list_1, list_ja, list_jb

def get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb):
    ia = ii & irght
    ib = (ii & ilft) // ihfbit
    ja = list_ja[ia]
    jb = list_jb[ib]
    return ja+jb


def ham_to_vec_complex(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    w = np.zeros(Nhilbert,dtype=complex)

    for i in range(Nhilbert):
       ii = list_1[i]        # i-th basis   
       for k in range(Nint): # loop for all interaction
        isite1 = list_isite1[k]
        isite2 = list_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        wght = 2.0*Jxx[k]
        diag = Jzz[k]
        ibit = ii & is12

        if (ibit==0 or ibit==is12): #up-up or down-down
             w[i] += diag*v1[i]
        else:  
            w[i] -= diag*v1[i]
            iexchg = ii ^ is12 
            newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
            w[i] += wght*v1[newcfg] 
    return w/4.0 

def continued_fraction(norm,salphas,sbetas,itr_dsf,eta,minene):

    print(norm,minene,eta)
    Sqw = np.zeros((1000,2),dtype=float)
    for omega in range(0,1000):
        Z = omega*0.01 + 1.0j*eta + minene
        tmp = 0.0 + 0.0j
        for itr in reversed(range(1,itr_dsf)):
            tmp = Z - salphas[itr] - tmp
            tmp = (sbetas[itr]**2)/tmp
        tmp = Z - salphas[0] - tmp
        Sqw[omega,:]=omega*0.01, -((norm/tmp).imag)/np.pi
        
    return Sqw




def sz_spin_dynamics(itr_dsf,qx,GS,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):

    salphas = []                         
    sbetas = [0.]                        

    #calculate |v1> = Sq^z |GS>
    v1 = np.zeros(Nhilbert, dtype=complex)
    v0 = np.zeros(Nhilbert, dtype=complex)
    w = np.zeros(Nhilbert,dtype=complex)
    tmp = np.zeros(2,dtype=float)
    for i in range(Nhilbert):
         ii = list_1[i]
         tmp[0:2]=0.0

         for k in range(N):
           is1 = 1<<k
           ibit = ii & is1

           if(ibit==is1): #site-k has up spin
              tmp[0] += np.cos(qx*k)
              tmp[1] += np.sin(qx*k)
           else:
              tmp[0] -= np.cos(qx*k)
              tmp[1] -= np.sin(qx*k)   

         v1[i] = GS[i]*tmp[0]/2.0 + 1.0j*GS[i]*tmp[1]/2.0 

    v1 = v1/np.sqrt(N)
    norm = np.vdot(v1,v1)
    v1 /= np.linalg.norm(v1)

    salpha=0. 
    sbeta=0. 

    for itr in range(0,itr_dsf):
        w = ham_to_vec_complex(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
        salpha = np.vdot(v1,w)
        w = w - salpha*v1 -sbeta*v0
        v0 = np.copy(v1)
        sbeta = np.sqrt(np.vdot(w,w))
        v1 = w/sbeta
        salphas.append(salpha)
        sbetas.append(sbeta)

    return salphas, sbetas, norm

def ham_to_vec(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    w = np.zeros(Nhilbert,dtype=float)

    for i in range(Nhilbert):
       ii = list_1[i]        # i-th basis   
       for k in range(Nint): # loop for all interaction
        isite1 = list_isite1[k]
        isite2 = list_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        wght = 2.0*Jxx[k]
        diag = Jzz[k]
        ibit = ii & is12

        if (ibit==0 or ibit==is12): #up-up or down-down
             w[i] += diag*v1[i]
        else:  
            w[i] -= diag*v1[i]
            iexchg = ii ^ is12 
            newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
            w[i] += wght*v1[newcfg] 
    return w


def simple_lanczos(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb,itr_max,eps):
    alphas = []                          #Diagonal parts of the trigonal matrix
    betas = [0.]                         #Off-diagonal parts of the trigonal matrix
    np.random.seed(seed=12345)
    v1 = np.random.rand(Nhilbert)       #old Lanczos vector (real number vector)
    v1 /= np.linalg.norm(v1)             #normalization
    v0 = np.zeros(Nhilbert, dtype=float)  #new Lanczos vector
    w  = np.zeros(Nhilbert, dtype=float)

    alpha = 0.
    beta = 0.
    pre_energy=0
    
    for iteration in range(0, itr_max):
        w = ham_to_vec(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
        alpha = np.dot(v1,w)
        w = w -alpha*v1 -beta*v0
        v0 = np.copy(v1)
        beta = np.sqrt(np.dot(w,w))
        v1 = w/beta
        alphas.append(alpha)
        betas.append(beta)

        t_eigs,t_vecs = scipy.linalg.eigh_tridiagonal(alphas,betas[1:-1])
        print(min(t_eigs)/4)

        if np.abs(min(t_eigs)-pre_energy) < eps:
          print("Lanczos converged in", iteration, "iterations")
          conv_itr=iteration
          print("The lowest 5 energy", t_eigs[0:4]/4)
          break

        pre_energy = min(t_eigs)


    #calcu GS eigenvector
    np.random.seed(seed=12345)
    v1 = np.random.rand(Nhilbert)
    v1 /= np.linalg.norm(v1)
    v0 = np.zeros(Nhilbert, dtype=float)
    w = np.zeros(Nhilbert, dtype=float)
    alpha = 0.
    beta = 0.
          
    GS = t_vecs[0,0]*v1  # GS wavefunction from 0-th eigenvector of the tridiagonal matrix

    for itr in range(0,conv_itr-1):
        w = ham_to_vec(w,v1,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
        alpha = np.dot(v1,w)
        w = w -alpha*v1 -beta*v0
        v0 = np.copy(v1)
        beta = np.sqrt(np.dot(w,w))
        v1 = w/beta

        GS = GS + t_vecs[itr+1,0]*v1
        print("eigenvector iteretion", itr)


    return t_eigs, np.array(alphas), np.array(betas[1:]), t_vecs, GS


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


def make_lattice(Lx,Ly,J1,J2):
    Jxx = []
    Jzz = []
    list_isite1 = []
    list_isite2 = []
    Nint = 0
    for iy in range(Ly):
        for ix in range(Lx):
            site1 = ix + Lx*iy
            site1x = (ix+1)%Lx + Lx*iy
            site1y = ix + Lx*((iy+1)%Ly)
            site1xpy = (ix+1)%Lx + Lx*((iy+1)%Ly)
            site1xmy = (ix+1)%Lx + Lx*((iy-1+Ly)%Ly)
#
            list_isite1.append(site1)
            list_isite2.append(site1x)
            Jxx.append(J1)
            Jzz.append(J1)
            Nint += 1
#
            list_isite1.append(site1)
            list_isite2.append(site1y)
            Jxx.append(J1)
            Jzz.append(J1)
            Nint += 1
#
            list_isite1.append(site1)
            list_isite2.append(site1xpy)
            Jxx.append(J2)
            Jzz.append(J2)
            Nint += 1
#
            list_isite1.append(site1)
            list_isite2.append(site1xmy)
            Jxx.append(J2)
            Jzz.append(J2)
            Nint += 1
    return Jxx, Jzz, list_isite1, list_isite2, Nint

def main():
    N=10
    Sz=0
    J1=1.00
    J2=0.4
    qx=np.pi # for spin dynamics
    eta=0.02
    itr_dsf=200

    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)
    binirght = np.binary_repr(irght,width=N)
    binilft = np.binary_repr(ilft,width=N)
    biniup = np.binary_repr(iup,width=N)
    print("J1=",J1)
    print("J2=",J2)
    print("N=",N)
    print("Sz=",Sz)
    print("Nup=",Nup)
    print("Nhilbert=",Nhilbert)
    print("ihfbit=",ihfbit)
    print("irght,binirght=",irght,binirght)
    print("ilft,binilft=",ilft,binilft)
    print("iup,biniup=",iup,biniup)
    start = time.time()
    list_1, list_ja, list_jb = make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup)
    end = time.time()
    print (end - start)
    print("")


    Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice_chain(N,J1,J2)#make_lattice(Lx,Ly,J1,J2)
    print (Jxx)
    print (Jzz)
    print (list_isite1)
    print (list_isite2)
    print("Nint=",Nint)

    start = time.time()
    eps=1e-12
    itr_max=1000
    eigs, alphas, betas, t_vecs, GS = simple_lanczos(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb,itr_max,eps)
    end = time.time()
    print (end - start)
    start = time.time()

    ene = eigs/4
    end = time.time()
    print (end - start)
    print ("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])

    print("")

    print("Dynamical spin structure factor (z-comp. only)")
    print("qx=",qx)
    print("eta=",eta)
    print("itr_dsf=",itr_dsf)
   
    salphas, sbetas, norm = sz_spin_dynamics(itr_dsf,qx,GS,Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    Sqw = continued_fraction(norm,salphas,sbetas,itr_dsf,eta,ene[0])

    return salphas, sbetas, norm, Sqw

if __name__ == "__main__":
   salphas, sbetas, norm, Sqw= main()
   plt.plot(Sqw[:,0],Sqw[:,1])
