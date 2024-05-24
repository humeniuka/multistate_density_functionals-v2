#!/usr/bin/env python
# coding: utf-8
"""
Electronic structure of beryllium dimer (Be2) with CASSCF

References
----------
[Be2] Merritt, Jeremy M., Vladimir E. Bondybey, and Michael C. Heaven.
      "Beryllium dimer—caught in the act of bonding."
      Science 324.5934 (2009): 1548-1551.
"""
import json
import numpy
import pyscf.fci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
import pyscf.tools.molden as molden

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
    
    # Geometry of beryllium-dimer, Be2
    #
    # The Be2 bond length is 2.45 Å (see [Be2]).
    # The bond length is scanned from 2.0 to 8.0 Å.
    bond_lengths = numpy.array(
        [
            2.0,
            # finer grid around equilibrium geometry
            2.2, 2.4, 2.45, 2.5, 2.6, 2.8,
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        ])
    for bond_length in bond_lengths:
        print(rf"* Be-Be bond length : {bond_length:4f} Å")
        mol = pyscf.gto.M(
            atom = f"""
            Be  {-bond_length/2} 0 0
            Be  { bond_length/2} 0 0
            """,
            # A large basis set is needed.
            #basis = 'aug-cc-pvqz',
            ### DEBUG
            basis = 'aug-cc-pvdz',
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
        # Show orbital energies, occupancies etc., requires hf.verbose > 0.
        hf.analyze()
        pyscf.tools.molden.dump_scf(hf, '/tmp/hf.molden', ignore_h=False)

        # CAS: 4 electrons in 19 orbitals
        ncas = 19
        nelecas = 4
        casscf = pyscf.mcscf.CASSCF(hf, ncas, nelecas)

        # Wave-function symmetry = Dooh
        # In the HF ground state the occupancies for the orbitals by irrep are
        # occupancy for each irrep:
        # A1g E1gx E1gy  A1u E1uy E1ux E2gx E2gy E3gx E3gy E2uy E2ux E3uy E3ux E4gx E4gy E4uy E4ux
        # 2    0    0    2    0    0    0    0    0    0    0    0    0    0    0    0    0    0

        # The HF/aug-cc-pvdz energies for the first 20 MOs at the equlibrium geometry are:
        # **** MO energy ****
        # MO #1 (A1g #1), energy= -4.73157902835041 occ= 2
        # MO #2 (A1u #1), energy= -4.73154814389738 occ= 2
        # MO #3 (A1g #2), energy= -0.395945560290537 occ= 2
        # MO #4 (A1u #2), energy= -0.242831740482202 occ= 2
        # MO #5 (A1g #3), energy= 0.00601879717358026 occ= 0
        # MO #6 (E1uy #1), energy= 0.00696067364876207 occ= 0
        # MO #7 (E1ux #1), energy= 0.00696067364876207 occ= 0
        # MO #8 (A1u #3), energy= 0.0142496988127067 occ= 0
        # MO #9 (E1gx #1), energy= 0.0219157082425647 occ= 0
        # MO #10 (E1gy #1), energy= 0.0219157082425647 occ= 0
        # MO #11 (A1g #4), energy= 0.0235525125063983 occ= 0
        # MO #12 (E1uy #2), energy= 0.0308827317404859 occ= 0
        # MO #13 (E1ux #2), energy= 0.0308827317404859 occ= 0
        # MO #14 (A1g #5), energy= 0.042984312571257 occ= 0
        # MO #15 (A1u #4), energy= 0.0564794427247009 occ= 0
        # MO #16 (E1gx #2), energy= 0.0863416020960436 occ= 0
        # MO #17 (E1gy #2), energy= 0.0863416020960436 occ= 0
        # MO #18 (A1u #5), energy= 0.133039266138266 occ= 0
        # MO #19 (A1g #6), energy= 0.152186673402173 occ= 0
        # MO #20 (E2gx #1), energy= 0.1525406562696 occ= 0

        # Number of active orbitals in each irrep.
        cas_irrep_nocc = {'A1g': 6, 'A1u': 5, 'E1gx': 2, 'E1gy': 2, 'E1ux': 2, 'E1uy': 2}
        # Construct the initial guess for the CASSCF orbitals.
        mo_guess = pyscf.mcscf.sort_mo_by_irrep(casscf, hf.mo_coeff, cas_irrep_nocc)

        # Compute lowest 2 singlet states with in the Σ+g irrep.
        nstate = 2
        casscf.nstate = nstate
        # Specify the spatial symmetry of the wavefunction.
        # Be2 has D∞h symmetry, the Σ+g irrep is called A1g.
        casscf.fcisolver.wfnsym = 'A1g'
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

        # Save data for later plotting
        scan_data['bond_length'].append(bond_length)
        scan_data['eigenenergies'].append(casscf.e_states.tolist())

        # Save intermediate results, in case the calculation is stopped early.
        with open('eigenenergies.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)

        # Save geometry and orbitals in CAS in molden format.
        with open(f"/tmp/beryllium-dimer_{bond_length:3.1f}.molden", "w") as molden_file:
            pyscf.tools.molden.header(mol, molden_file, ignore_h=False)
            pyscf.tools.molden.orbital_coeff(
                mol, molden_file, casscf.mo_coeff, ene=casscf.mo_energy, occ=casscf.mo_occ)
