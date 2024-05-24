#!/usr/bin/env python
# coding: utf-8
"""
Plot the spin-up and spin-down components of the matrix density D(r)
along the x-axis.
"""
import json
import matplotlib.pyplot as plt
import numpy
import pyscf.fci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf

from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF


if __name__ == "__main__":
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # bond length (in Å), the scan variable
        'bond_length': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
    }

    # To fix the signs of the off-diagonal elements of the matrix densities,
    # so that we can plot smooth, continuous curves, the global phases of
    # the wavefunctions have to be aligned with the phases at the previous
    # scan point (the reference).
    msmd_ref = None
    
    # Geometry of lithium-hydride, LiH
    #
    # The LiH bond length is 1.596 Å.
    bond_length = 1.596
    mol = pyscf.gto.M(
        atom = f"""
        Li  {-bond_length/2} 0 0
        H  { bond_length/2} 0 0
        """,
        # A large basis set is needed.
        ### DEBUG
        #basis = 'aug-cc-pvqz',
        basis = '6-31g', #'cc-pvdz',
        ###
        charge = 0,
        # singlet
        spin = 0,
        # use symmetry
        symmetry = True)
    print(f"Molecular symmetry is used: {mol.topgroup}")
    
    hf = pyscf.scf.RHF(mol)
    # supress printing of SCF energy
    #hf.verbose = 0
    # compute self-consistent field
    hf.kernel()

    # CAS: 6 electrons in 21 orbitals
    ### DEBUG
    #ncas = 21
    #nelecas = 6
    #ncas = 13
    #nelecas = 4
    ncas = 9
    nelecas = 4
    ###
    casscf = pyscf.mcscf.CASSCF(hf, ncas, nelecas)

    # In the HF ground state the occupancies for the orbitals by irrep are
    # irrep:     A1  E1x  E1y  E2x  E2y  E3x  E3y  E4x  E4y
    # orbitals   2    0    0    0    0    0    0    0    0

    # Number of active orbitals in each irrep.
    ### DEBUG
    #cas_irrep_nocc = {'A1': 9, 'E1x': 6, 'E1y': 6}
    #cas_irrep_nocc = {'A1': 7, 'E1x': 3, 'E1y': 3}
    cas_irrep_nocc = {'A1': 5, 'E1x': 2, 'E1y': 2}
    ###
    # Construct the initial guess for the CASSCF orbitals.
    mo_guess = pyscf.mcscf.sort_mo_by_irrep(casscf, hf.mo_coeff, cas_irrep_nocc)
    
    # Compute lowest 2 singlet states with in the Σ+ irrep.
    nstate = 2
    casscf.nstate = nstate
    # Specify the spatial symmetry of the wavefunction.
    # LiH has C∞v symmetry, the Σ+ irrep is called A1.
    casscf.fcisolver.wfnsym = 'A1'
    # Each state is given equal weight in the CAS energy.
    weights = numpy.array([1.0/nstate] * nstate)
    casscf = casscf.state_average(weights)
    # States with undesired spins are shifted up in energy.
    casscf.fix_spin(shift=0.5)
    print("CASSCF ...")
    casscf.kernel(mo_guess)

    # Output all determinants coefficients.
    for state in range(0, nstate):
        active_orbitals = range(0, ncas)
        occslst = pyscf.fci.cistring.gen_occslst(active_orbitals, nelecas//2)
        print(f'State {state}  Energy= {casscf.e_states[state]}')
        print('   Determinants    CI coefficients')
        for i,occsa in enumerate(occslst):
            for j,occsb in enumerate(occslst):
                # Only the dominant CI coefficients are printed.
                ci_coefficient = casscf.ci[state][i,j]
                if abs(ci_coefficient) < 0.01:
                    continue
                # The occupation string shows which orbitals are doubly
                # occupied (2), singly occupied (a or b) or empty (.)
                # e.g.: '222ab...' for a HOMO-LUMO excited determinant.
                occupation_string = ''
                for o in active_orbitals:
                    if o in occsa and o in occsb:
                        # orbital is doubly occupied
                        occupation_string += '2'
                    elif o in occsa:
                        # orbital is singly occupied by spin-up electron
                        occupation_string += 'a'
                    elif o in occsb:
                        # orbital is singly occupied by spin-down electron
                        occupation_string += 'b'
                    else:
                        # orbital is unoccupied
                        occupation_string += '.'
                print('     %s      %+10.7f' % (occupation_string, ci_coefficient))

    # Construct matrix density D(r) from the CASSCF wavefunctions.
    msmd = MultistateMatrixDensityCASSCF(mol, casscf)

    # Plot D(x,0,0), cut along x-axis
    Ncoord = 5000
    coords = numpy.zeros((Ncoord, 3))
    r = numpy.linspace(-10.0, 10.0, Ncoord)
    coords[:,0] = r

    # Evaluate D(r) on the integration grid.
    D, _, _ = msmd.evaluate(coords)

    nspin,nstate,_,_ = D.shape

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    spin_to_string = {0 : r'\uparrow', 1: r'\downarrow'}
    spin_to_linestyle = {0: '-', 1: '--'}
    
    # State densities
    axes[0].set_ylabel(r"state density $D^{\sigma}_{II}(x,0,0)$ / $a_0^{-3}$")
    axes[0].set_xlabel(r"x / $\AA$")

    for spin in range(0, nspin):
        for i in range(0, nstate):
            axes[0].plot(
                r, D[spin,i,i,:],
                lw=2, ls=spin_to_linestyle[spin],
                label=rf"$D^{{{spin_to_string[spin]}}}_{{{i},{i}}}$")

    axes[0].legend(title="$\mathbf{(a)}$ diagonal")

    # Transition densities
    axes[0].set_ylabel(r"transition density $D_{IJ}(x,0,0)$ / $a_0^{-3}$")
    axes[0].set_xlabel(r"x / $\AA$")


    for spin in range(0, nspin):
        for i in range(0, nstate):
            for j in range(i+1, nstate):
                axes[1].plot(
                    r, D[spin,i,j,:],
                    lw=2, ls=spin_to_linestyle[spin],
                    label=rf"$D^{{{spin_to_string[spin]}}}_{{{i},{j}}}$")

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(b)}$ off-diagonal")
    
    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    #plt.savefig("spin_density.svg")
    #plt.savefig("spin_density.png", dpi=300)

    plt.show()
