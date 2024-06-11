# IsingModel_TFG

Here you can find the code I developed for my Bachelor's thesis "Ising Model in a Quantum Computer" using python.

$H = \sum_{j}{\sigma_{j}^{x}\sigma_{j+1}^{x}} + \lambda \sum_{j}{\sigma_{j}^{z}}$

There are two files of code:

- TFG_IsingModel.ipynb is the code used to run the quantum circuit for the simulation of the Ising Hamiltonian. If you run this code you get the Magnetitzation of the Ground state of the Ising Model for different values for the polarization field $(\lambda)$. To run this code you need to have qiskit updated to version 1.1 and an IBM Quantum Platform account to run it on one of their Quantum Machines.

- DiagonalizingtheH.py is the code used to run the quantum circuit for the diagonalization of the Ising Hamiltonian. If you run this code you get the energy spectrum for the $n=4$ Ising Hamiltonian.
