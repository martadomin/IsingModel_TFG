# -*- coding: utf-8 -*-
"""
Created on Mon May 13 23:24:49 2024

@author: marta
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys

#%% Necessary Vectors and Matrices

sigz = np.array([[1,0],[0,-1]])
sigx = np.array([[0,1],[1,0]])
ident2 = np.identity(2)
ident4 = np.identity(4)
ident8 = np.identity(8)
vec0 = np.array([[1],[0]])
vec1 = np.array([[0],[1]])

#%% Eigenvectors for sigz 

up4 = np.kron(vec0, np.kron(vec0, np.kron(vec0, vec0)))
up3down = np.kron(vec0, np.kron(vec0, np.kron(vec0, vec1)))
up2downup = np.kron(vec0, np.kron(vec0, np.kron(vec1, vec0)))
up2down2 = np.kron(vec0, np.kron(vec0, np.kron(vec1, vec1)))
updownup2 = np.kron(vec0, np.kron(vec1, np.kron(vec0, vec0)))
updowndupdown = np.kron(vec0, np.kron(vec1, np.kron(vec0, vec1)))
updown2up = np.kron(vec0, np.kron(vec1, np.kron(vec1, vec0)))
updown3 = np.kron(vec0, np.kron(vec1, np.kron(vec1, vec1)))
downup3 = np.kron(vec1, np.kron(vec0, np.kron(vec0, vec0)))
downup2down = np.kron(vec1, np.kron(vec0, np.kron(vec0, vec1)))
downupdownup = np.kron(vec1, np.kron(vec0, np.kron(vec1, vec0)))
downupdown2 = np.kron(vec1, np.kron(vec0, np.kron(vec1, vec1)))
down2up2 = np.kron(vec1, np.kron(vec1, np.kron(vec0, vec0)))
down2updown = np.kron(vec1, np.kron(vec1, np.kron(vec0, vec1)))
down3up = np.kron(vec1, np.kron(vec1, np.kron(vec1, vec0)))
down4 = np.kron(vec1, np.kron(vec1, np.kron(vec1, vec1)))

vecpar = np.array([up4,up2down2,updowndupdown,updown2up,downup2down,downupdownup,down2up2,down4])
vecsen = np.array([up3down,up2downup,updownup2,updown3,downup3,downupdown2,down2updown,down3up ])

#%% Projectors for even and odd parity states

Projecpar = np.zeros([16,16])
Projecsen = np.zeros([16,16])

for i in range(0,8):
    Projecpar += np.multiply(vecpar[i,::,::],np.transpose(vecpar[i,::,::]))
    Projecsen += np.multiply(vecsen[i,::,::],np.transpose(vecsen[i,::,::]))
    
#%%
#Function that gives us the matrix for the Ising Hamiltonian

# N --> Number of lattices for the Ising Model
# l --> Strength of the magnetic field
# par --> Parity of the boundary conditions (Must be 1 or -1)

def HIsing(N, l, par):
    
    if par != -1 and par != 1:
        print('Not a valid parity for the boundary condition')
        sys.exit()
    
    #Our final Hamiltonian matrix
    Hising = np.zeros([2**N,2**N])
    
    #Necessary operators for all tensorial products
    #X term
    listmat = np.zeros([2,2,N])
    listmattemp = np.zeros([2,2,N])

    #Z term
    listmatz = np.zeros([2,2,N])
    listmattempz = np.zeros([2,2,N])

    for i in range(0, N):
        listmat[::,::,i] = np.identity(2)
        listmatz[::,::,i] = np.identity(2)
        
    #Tensorial products
    for j in range(0, N):
        
        listmattemp[::,::,::] = listmat[::,::,::]
        listmattempz[::,::,::] = listmatz[::,::,::]
        
        if j == (N-1):
            #X boundary term
            listmattemp[::,::,j] = sigx[::,::]
            listmattemp[::,::,0] = sigx[::,::]
            #Z term
            listmattempz[::,::,j] = sigz
            
        else:     
            #X term
            listmattemp[::,::,j] = sigx[::,::]
            listmattemp[::,::,j+1] = sigx[::,::]
            #Z term
            listmattempz[::,::,j] = sigz
        
        matinterx = listmattemp[::,::,0]
        matinterz = listmattempz[::,::,0]
        for k in range(0, N-1):
            if j == N-1:
                matdefx = par *np.kron(matinterx, listmattemp[::,::,k+1])
                matdefz = np.kron(matinterz, listmattempz[::,::,k+1])
                matinterx = matdefx
                matinterz = matdefz
            else:      
                matdefx = np.kron(matinterx, listmattemp[::,::,k+1])
                matdefz = np.kron(matinterz, listmattempz[::,::,k+1])
                matinterx = matdefx
                matinterz = matdefz
        #We sum the terms
        Hising += matdefx + l*matdefz
    
    return Hising
#%% Finite Hamiltonain Matrix for N=4
N = 4
eigenvalues = np.zeros((41,2**N))
eigenvectors = np.zeros((2**N,41, 2**N))
Htot = np.zeros([16,16,41])

l = -2
for i in range(0,41, 1):
    Hpar = np.matmul(Projecpar, np.matmul(HIsing(N, l, -1), Projecpar))
    Hsen = np.matmul(Projecsen, np.matmul(HIsing(N, l, 1), Projecsen))
    Htot[::,::,i] = Hpar + Hsen
    eigenvalues[i,::] = np.linalg.eigh(Htot[::,::,i])[0]
    eigenvectors[::,i,::] = np.linalg.eigh(Htot[::,::,i])[1]
    l += 0.1

#%%
x = np.linspace(-3, 3, 101)
k = np.array([-1,0,1,2])
w1neg = np.zeros([101])
w0 = np.zeros([101])
w1 = np.zeros([101])
w2 = np.zeros([101])
w = np.zeros([101,32768])

mat1neg = np.zeros((2,2, 101))
mat0 = np.zeros((2,2,101))
mat1 = np.zeros((2,2,101))
mat2 = np.zeros((2,2,101))
Hdiag = np.zeros([16,16,101])

eigenvalues = np.zeros((101,16))
eigenvectors = np.zeros((16,101, 16))

for i in range(0,101):
    kn = (np.pi)/(2)*k
    w1neg[i] = np.sqrt((np.cos(kn[0])-x[i])**2 + (np.sin(kn[0]))**2)
    w0[i] = np.sqrt((np.cos(kn[1])-x[i])**2 + (np.sin(kn[1]))**2)
    w1[i] = np.sqrt((np.cos(kn[2])-x[i])**2 + (np.sin(kn[2]))**2)
    w2[i] = np.sqrt((np.cos(kn[3])-x[i])**2 + (np.sin(kn[3]))**2)
    
    #Hdiag
    Hdiag[:,:,i] = np.kron(-w1neg[i]*sigz, ident8) + np.kron(ident2,np.kron(-w0[i]*sigz, ident4)) + np.kron(ident4,np.kron(-w1[i]*sigz,ident2))+ np.kron(ident8,-w2[i]*sigz)
    
    #Matrices
    mat1neg[0,0,i] = -w1neg[i]
    mat1neg[1,1,i] = -w1neg[i]
    
    mat0[0,0,i] = -w0[i]
    mat0[1,1,i] = w0[i]
    
    mat1[0,0,i] = -w1[i]
    mat1[1,1,i] = w1[i]
    
    mat2[0,0,i] = -w2[i]
    mat2[1,1,i] = w2[i]
    
    #Diagonalize the Hdiag
    eigenvalues[i,::]  = np.linalg.eigh(Hdiag[:,:,i])[0]
    eigenvectors[::,i,::] = np.linalg.eigh(Hdiag[:,:,i])[1]
    
Ediag = np.zeros([101,16])



# 4 partículas
Ediag[::,0] = w1neg + w0 + w1 + w2

# 3 partícula

Ediag[::,1] = w1neg + w0 + w1 - w2
Ediag[::,2] = w1neg + w0 - w1 + w2
Ediag[::,3] = w1neg - w0 + w1 + w2
Ediag[::,4] = - w1neg + w0 + w1 + w2


#2 partículas

Ediag[::,5] = w1neg + w0 - w1 - w2
Ediag[::,6] = w1neg - w0 + w1 - w2
Ediag[::,7] = - w1neg + w0 + w1 - w2
Ediag[::,8] = w1neg - w0 - w1 + w2
Ediag[::,9] = - w1neg + w0 - w1 + w2
Ediag[::,10] = - w1neg - w0 + w1 + w2

# 1 partículas

Ediag[::,11] = w1neg - w0 - w1 - w2
Ediag[::,12] = - w1neg + w0 - w1 - w2
Ediag[::,13] = - w1neg - w0 + w1 - w2
Ediag[::,14] = - w1neg - w0 - w1 + w2

# 0 partícula

Ediag[::,15] = - w1neg - w0 - w1 - w2
    
#%% GATES FOR THE QUANTUM CIRCUIT

#JW TRANSFORM
#Swap gate but taking into account anticommutation relations for fermions
fSWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,-1]])

#FOURIER TRANSFORM
#2qbit Fourier transform
# n --> Number of lattices of our Ising Hamiltonian
# k --> Momentum (...)

def fourier2q(n,k):
    
    phase = np.exp(-1j*(2.*np.pi*k)/(n))
    f2k = np.array([[1,0,0,0],
                    [0,-phase/np.sqrt(2),1/np.sqrt(2),0],
                    [0,phase/np.sqrt(2), 1/np.sqrt(2),0],
                    [0,0,0,-phase]])
    
    return f2k

#Examples:
n = 4
k = 1
f21 = fourier2q(n, k)

n = 4
k = 0
f20 = fourier2q(n, k)

#BOGOLIUBOV TRANSFORM
#2qbit Bogoliubov transform
# n --> Number of lattices of our Ising Hamiltonian
# k --> Momentum (...)
# lamb --> Strength of the magnetic Field

def bogo2q(n,k,lamb):
    
    kn = (2*np.pi*k)/(n)
    
    if lamb == 1:
        thetak = -np.pi/2.
    else:
        thetak = np.arctan(np.sin(kn)/(np.cos(kn)-lamb))
    
    b2k = np.array([[np.cos(thetak/2),0,0,(1j*np.sin(thetak/2))],
                    [0,1,0,0],
                    [0,0,1,0],
                    [1j*np.sin(thetak/2),0,0,np.cos(thetak/2)]])
    return b2k

#Examples:
lamb = 0

n = 4
k = 2
b20 = bogo2q(n,k,lamb)

n = 4 
k = 1
b21 = bogo2q(n,k,lamb)
#%%
vectrial = np.kron(vec0, np.kron(vec0,np.kron(vec0, vec0)))
Had = (1./np.sqrt(2))*np.array([[1,1],[1,-1]])

vecequi = np.matmul(np.kron(Had,np.kron(Had, np.kron(Had, Had))), vectrial)

print(np.matmul(np.kron(ident2, np.kron(fSWAP, ident2)), vecequi))

#%% 

lamb = -2.0
vec0 = np.array([[1],[0]])
vec1 = np.array([[0],[1]])
res1 = np.zeros((16,16,41),dtype = 'complex_')
res2 = np.zeros((16,41),dtype = 'complex_')

ident4 = np.identity(8)

for i in range(0, 41):
        
    #bogo gate
    k = 0
    b20 = bogo2q(n,k,lamb)
    k = 1
    b21 = bogo2q(n,k,lamb)
    
    #Primer paso swap
    primer = np.kron(ident2, np.kron(fSWAP, ident2))
    #segundo paso 2 fourier
    segundo = np.kron(f20, f20)
    #tercer paso swap
    tercer = np.kron(ident2, np.kron(fSWAP, ident2))
    #cuarto paso fourier
    cuarto = np.kron(f20, f21)
    #quinto paso bogoulibov
    quinto = np.kron(b20, b21)
    
    #res
    Udis = np.matmul(primer,np.matmul(segundo,np.matmul(tercer,np.matmul(cuarto,quinto))))
    Udisdag = np.linalg.inv(Udis)
    
    res1[::,::,i] = np.real(np.round(np.matmul(Udisdag, np.matmul(Htot[:,:,i],Udis)),10))
    print(lamb, scipy.sparse.dia_matrix(res1[::,::,i]))
    
    lamb += 0.1
    
#%% Monoparticular energies of the Diagonal Hamiltonian
x = np.linspace(-2, 2, 41)
k = np.linspace(((-N//2)+1),(N//2) , N)
kn = np.zeros([N])
w = np.zeros([41,N])

for i in range(0,41):
    for j in range(0,N):
        kn = ((np.pi)/(2))*k[j]
        w[i,j] = np.sqrt((np.cos(kn)-x[i])**2 + (np.sin(kn))**2)

wgs = np.zeros([41])

for i in range(0,41):
    for j in range(0,N):
        wgs[i] += -w[i,j]
        
#%% Groundstate

E = np.zeros([41,16])
x = np.linspace(-2,2,41)
for i in range(0,16):
    for j in range(0,41):
        E[j,i] = res1[i,i,j]
    
#%%    
#Plot of the Groundstate Energy
import matplotlib.font_manager as font_manager

x2 = np.linspace(-3, 3, 101)

plt.figure(figsize = [16,14])
plt.grid(True)

plt.ylabel('E', fontsize = 40, fontname="Cambria")
plt.xlabel(r'$\lambda$', fontsize = 40, fontname="Cambria")



plt.plot(x2[16:84],  Ediag[16:84,0], color = 'red', label = r'with $\epsilon_{k}$')
plt.plot(x2[16:84], Ediag[16:84,:], color = 'red')

plt.plot(x,E[::,0], linestyle = 'none', marker = 'o', markersize = 6, color ='blue', label = r'with $U_{dis}$')
plt.plot(x,E, linestyle = 'none', marker = 'o', markersize = 6, color ='blue')

plt.xticks(fontsize = 40, fontname="Cambria")
plt.yticks(fontsize = 40, fontname="Cambria")

font = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=38
                                   )

#Save the fig
plt.legend(prop=font, loc='upper center')



