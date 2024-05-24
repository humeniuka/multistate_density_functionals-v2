#!/usr/bin/env python
# coding: utf-8
"""
Electronic structure of ethene with CASSCF and a small basis set.
"""
import json
import numpy
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
import pyscf.tools.molden as molden

if __name__ == "__main__":
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # torsion around carbon-carbon double bond (in °), the scan variable
        'torsion_angle': [],
        # pyramidalization angle of one CH2 group
        # (see Fig. 2 in Barbatti et al. (2004) https://doi.org/10.1063/1.1807378 )
        'pyramidalization_angle': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
    }

    # Geometry of ethene
    #
    # Compute lowest 3 singlet states.
    nstate = 3
    # To fix the signs of the off-diagonal elements of the matrix densities,
    # so that we can plot smooth, continuous curves, the global phases of
    # the wavefunctions have to be aligned with the phases at the previous
    # scan point (the reference).
    msmd_ref = None
    # The experimental HCH angle is 117.6°.
    angleHCH = 117.6 * numpy.pi/180.0
    # The experimental CH bond length is 1.086 Å.
    rCH = 1.086
    # The experimental carbon-carbon bond length of ethene (NIST) is 1.339 Å.
    rCC = 1.339
    # C=C torsion angle
    torsion_angle = 90.0 * numpy.pi/180.0
    # The pyramidalization angle is scanned from 60° to 180° (cf. Fig. 5 in Barbatti et al.)
    pyramidalization_angles = numpy.linspace(60.0, 180.0, 13) * numpy.pi/180.0
    for pyramidalization_angle in pyramidalization_angles:
        print(rf"* torsion angle : {torsion_angle*180.0/numpy.pi:4f}°")
        print(rf"  pyramidalization angle : {pyramidalization_angle*180.0/numpy.pi:4f}°")
        # The hydrogen atoms of the fixed H2C= group lie in the xy plane.
        xH = rCH * numpy.cos(angleHCH/2.0)
        yH = rCH * numpy.sin(angleHCH/2.0)
        # The C=C bond is parallel to the x-axis.
        # The =CH2 group is rotated around the x-axis.
        # The pyramidalized CH2 group is rotated around the y-axis.
        mol = pyscf.gto.M(
            atom = f"""
            C {-rCC/2} 0 0
            C { rCC/2} 0 0
            H { rCC/2+xH} { numpy.cos(torsion_angle) * yH} { numpy.sin(torsion_angle) * yH}
            H { rCC/2+xH} {-numpy.cos(torsion_angle) * yH} {-numpy.sin(torsion_angle) * yH}
            H {-rCC/2+numpy.cos(pyramidalization_angle) * xH} { yH} {-numpy.sin(pyramidalization_angle) * xH}
            H {-rCC/2+numpy.cos(pyramidalization_angle) * xH} {-yH} {-numpy.sin(pyramidalization_angle) * xH}
            """,
            # small basis set
            basis = '6-31g*',
            charge = 0,
            # singlet
            spin = 0)

        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()

        weights = numpy.array([1.0/nstate] * nstate)
        # CAS: 2 electrons in 2 orbitals
        ncas = 2
        nelecas = 2
        casscf = pyscf.mcscf.CASSCF(hf, ncas, nelecas).state_average(weights)
        casscf.nstate = nstate
        casscf.fix_spin_()
        casscf.run()

        # Output all determinants coefficients.
        for state in range(0, nstate):
            active_orbitals = range(0, ncas)
            occslst = pyscf.fci.cistring.gen_occslst(active_orbitals, nelecas//2)    
            print(f'State {state}  Energy= {casscf.e_states[state]}')
            print('   Determinants    CI coefficients')
            for i,occsa in enumerate(occslst):
                for j,occsb in enumerate(occslst):
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
                    print('     %s      %+10.7f' % (occupation_string, casscf.ci[state][i,j]))

        # Save data for later plotting
        scan_data['torsion_angle'].append(torsion_angle*180.0/numpy.pi)
        scan_data['pyramidalization_angle'].append(pyramidalization_angle*180.0/numpy.pi)
        scan_data['eigenenergies'].append(casscf.e_states.tolist())

        # Save intermediate results, in case the calculation is stopped early.
        with open('eigenenergies_ethene_pyramidalization.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)

        # Save geometry in molden format.
        angle = pyramidalization_angle*180.0/numpy.pi
        with open(f"/tmp/ethene_pyramidalization_{angle:3.1f}.molden", "w") as molden_file:
            pyscf.tools.molden.header(mol, molden_file, ignore_h=False)
