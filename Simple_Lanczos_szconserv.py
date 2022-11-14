# coding:utf-8
from __future__ import print_function
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time

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

def make_list(Nup,Nhilbert,ihfbit,irght,ilft,iup):
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
    for n in range(1,Nhilbert):
        ia = ii & irght
        ib = (ii & ilft) // ihfbit
        if (ib == ib_old):
            ja += 1
        else:
            jb += ja+1
            ja = 0
        list_1[n] = ii
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


def ham_to_vec(w,v1,Jxx,Jzz,list_isite1,list_isite2,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    w = np.zeros(Nhilbert,dtype=float) #output vector

    for n in range(Nhilbert):
       ii = list_1[n]           # n-th basis   
       for ij in range(Nint):    # loop for all interaction
        isite1 = list_isite1[ij]  # site i
        isite2 = list_isite2[ij]  # site j
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2       # up-up state only at site i and j
        wght = 2.0*Jxx[ij]
        diag = Jzz[ij]
        ibit = ii & is12

        if (ibit==0 or ibit==is12): #up-up or down-down
             w[n] += diag*v1[n]
        else:  
            w[n] -= diag*v1[n]
            iexchg = ii ^ is12 
            newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
            w[n] += wght*v1[newcfg] 
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
    
    for k in range(0, itr_max):
        w = ham_to_vec(w,v1,Jxx,Jzz,list_isite1,list_isite2,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
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
          print("Lanczos converged in", k, "iterations")
          conv_itr=k #M value in the tutorial slide
          print("The lowest 5 energy", t_eigs[0:4]/4)
          break

        pre_energy = min(t_eigs)


    #calcu GS eigenvector
    np.random.seed(seed=12345)    #set the same seed value we used above
    v1 = np.random.rand(Nhilbert)
    v1 /= np.linalg.norm(v1)
    v0 = np.zeros(Nhilbert, dtype=float)
    w = np.zeros(Nhilbert, dtype=float)
    alpha = 0.
    beta = 0.
          
    GS = t_vecs[0,0]*v1  # GS wavefunction from 0-th eigenvector of the tridiagonal matrix

    for k in range(0,conv_itr-1):
        w = ham_to_vec(w,v1,Jxx,Jzz,list_isite1,list_isite2,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
        alpha = np.dot(v1,w)
        w = w -alpha*v1 -beta*v0
        v0 = np.copy(v1)
        beta = np.sqrt(np.dot(w,w))
        v1 = w/beta

        GS = GS + t_vecs[k+1,0]*v1
        print("eigenvector iteretion", k)


    return t_eigs, np.array(alphas), np.array(betas[1:]), t_vecs, GS


def calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,phi,list_1):
    szz = np.zeros(Ncorr,dtype=float)
    for ij in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[ij] #site i
        isite2 = list_corr_isite2[ij] #site j
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for n in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[n]
            ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): factor = +1
                factor = +1.0
            else: # if (spin1,spin2) = (01) or (10): factor = -1
                factor = -1.0
            corr += factor*phi[n]**2 # phi[n]: real
        szz[ij] = 0.25 * corr
        if (isite1==isite2):
            szz[ij] = 0.25
    return szz

def calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,phi,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    sxx = np.zeros(Ncorr,dtype=float)
    for ij in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[ij] #site i
        isite2 = list_corr_isite2[ij] #site j
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for n in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[n]
            ibit = ii & is12 # find sgmz.sgmz|ud> = -|ud> or sgmz.sgmz|du> = -|du>
            if (ibit==is1 or ibit==is2): # if (spin1,spin2) = (10) or (01)
                iexchg = ii ^ is12 # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                corr += phi[n]*phi[newcfg] # phi[n]: real
        sxx[ij] = 0.25 * corr
        if (isite1==isite2):
            sxx[ij] = 0.25
    return sxx

def make_lattice_chain(N,J1,J2): #J1-J2 chain
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


def make_lattice(Lx,Ly,J1,J2): #J1-J2 square-lattice
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
    N=4
    Sz=0
    J1=1.00
    J2=0.4
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
    list_1, list_ja, list_jb = make_list(Nup,Nhilbert,ihfbit,irght,ilft,iup)
    print(list_1)
    end = time.time()
    print (end - start)
    print("")

    Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice_chain(N,J1,J2)#make_lattice(Lx,Ly,J1,J2)
    print (Jxx)
    print (Jzz)
    print (list_isite1)
    print (list_isite2)
    print("Nint=",Nint)

    eps=1e-12
    itr_max=1000
    start = time.time()
    eigs, alphas, betas, t_vecs, GS = simple_lanczos(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb,itr_max,eps)
    end = time.time()
    print (end - start)
    ene = eigs/4


    print ("#GS energy:",ene[0],ene[1],ene[2],ene[3],ene[4])

    print("")
    Ncorr = N # number of total correlations
    list_corr_isite1 = [0 for k in range(Ncorr)] # site 1
    list_corr_isite2 = [k for k in range(Ncorr)] # site 2
    print (list_corr_isite1)
    print (list_corr_isite2)

    #Start correlation function calculation 
    start = time.time()
    szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,GS,list_1)
    sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,GS,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    ss = szz+sxx+sxx
    stot2 = N*np.sum(ss)
    end = time.time()
    print (end - start)
    print("")
    print ("# <S^z_i S^z_j>:",szz)
    print ("# <S^x_i S^x_j>:",sxx)
    print ("# <S S>:",ss)
    print ("# S^tot(S^tot+1):",stot2)

if __name__ == "__main__":
    main()
