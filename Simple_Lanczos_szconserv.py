# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#import scipy.linalg
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
        #alphas.append(alpha)
        #betas.append(beta)
        print("eigenvector iteretion", itr)


    return t_eigs, np.array(alphas), np.array(betas[1:]), t_vecs, GS


    

def make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    listki = np.zeros((Nint+1)*Nhilbert,dtype=int)
    loc = np.zeros((Nint+1)*Nhilbert,dtype=int)
    elemnt = np.zeros((Nint+1)*Nhilbert,dtype=float)
    listki = [i for k in range(Nint+1) for i in range(Nhilbert)]
    for k in range(Nint): # loop for all interactions
        isite1 = list_isite1[k]
        isite2 = list_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        wght = 2.0*Jxx[k]
        diag = Jzz[k]
## calculate elements of
## H_loc = Jzz sgmz.sgmz + Jxx (sgmx.sgmx + sgmy.sgmy)
##       = Jzz sgmz.sgmz + 2*Jxx (S+.S- + S-.S+)
        for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[i]
            ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
            loc[Nint*Nhilbert+i] = i # store diag index
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): sgmz.sgmz only
                elemnt[Nint*Nhilbert+i] += diag # store +Jzz
#                print("# diag k(interactions) i(Hilbert)",k,i)
#                print("# diag ii  ",np.binary_repr(ii,width=N))
#                print("# diag is12",np.binary_repr(is12,width=N))
#                print("# diag ibit",np.binary_repr(ibit,width=N))
            else: # if (spin1,spin2) = (01) or (10): sgmz.sgmz and (S+.S- or S-.S+)
                elemnt[Nint*Nhilbert+i] -= diag # store -Jzz
                iexchg = ii ^ is12 # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                elemnt[k*Nhilbert+i] = wght # store 2*Jxx
                loc[k*Nhilbert+i] = newcfg # store offdiag index
#                print("# offdiag k(interactions) i(Hilbert)",k,i)
#                print("# offdiag ii  ",np.binary_repr(ii,width=N))
#                print("# offdiag is12",np.binary_repr(is12,width=N))
#                print("# offdiag iexc",np.binary_repr(iexchg,width=N))
    HamCSR = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert))
    return HamCSR

def calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,list_1):
    szz = np.zeros(Ncorr,dtype=float)
    for k in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[k]
        isite2 = list_corr_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[i]
            ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): factor = +1
                factor = +1.0
            else: # if (spin1,spin2) = (01) or (10): factor = -1
                factor = -1.0
            corr += factor*psi[i]**2 # psi[i]: real
        szz[k] = 0.25 * corr
        if (isite1==isite2):
            szz[k] = 0.25
    return szz

def calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    sxx = np.zeros(Ncorr,dtype=float)
    for k in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[k]
        isite2 = list_corr_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[i]
            ibit = ii & is12 # find sgmz.sgmz|ud> = -|ud> or sgmz.sgmz|du> = -|du>
            if (ibit==is1 or ibit==is2): # if (spin1,spin2) = (10) or (01)
                iexchg = ii ^ is12 # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                corr += psi[i]*psi[newcfg] # psi[i]: real
        sxx[k] = 0.25 * corr
        if (isite1==isite2):
            sxx[k] = 0.25
    return sxx

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
   # args = parse_args()
   # Lx = args.Lx
   # Ly = args.Ly
   # N = Lx*Ly
   # Sz = args.Sz
   # J1 = args.J1
   # J2 = args.J2
  #  Lx=4
  #  Ly=4
  #  N=Lx*Ly
    N=4
    Sz=0
    J1=1.00
    J2=0.4
    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)
    binirght = np.binary_repr(irght,width=N)
    binilft = np.binary_repr(ilft,width=N)
    biniup = np.binary_repr(iup,width=N)
  #  print("Lx=",Lx)
  #  print("Ly=",Ly)
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
    print(list_1)
    end = time.time()
    print (end - start)
#    print("list_1=",list_1)
#    print("list_ja=",list_ja)
#    print("list_jb=",list_jb)
    print("")
#    print("i ii binii ja+jb")
#    for i in range(Nhilbert):
#        ii = list_1[i]
#        binii = np.binary_repr(ii,width=N)
#        ind = get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb)
#        print(i,ii,binii,ind)

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
  ##  HamCSR = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    end = time.time()
    print (end - start)
#    print (HamCSR)
    start = time.time()
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,k=5)
  ##  ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
  ##  ene = ene/N/4
    ene = eigs/4
    end = time.time()
    print (end - start)
    #print ("# GS energy:",ene[0])
    print ("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
#    vec_sgn = np.sign(np.amax(vec[:,0]))
#    print ("# GS wave function:")
#    for i in range (Nhilbert):
#        ii = list_1[i]
#        binii = np.binary_repr(ii,width=N)
#        print (i,vec[i,0]*vec_sgn,binii)
#
    print("")
    Ncorr = N # number of total correlations
    list_corr_isite1 = [0 for k in range(Ncorr)] # site 1
    list_corr_isite2 = [k for k in range(Ncorr)] # site 2
    print (list_corr_isite1)
    print (list_corr_isite2)
    #psi = vec[:,0] # choose the ground state
    start = time.time()
    szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,GS,list_1)
    sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,GS,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    ss = szz+sxx+sxx
    stot2 = N*np.sum(ss)
    end = time.time()
    print (end - start)
    print ("# szz:",szz)
    print ("# sxx:",sxx)
    print ("# ss:",ss)
    print ("# stot(stot+1):",stot2)

if __name__ == "__main__":
    main()
