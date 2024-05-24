#!/usr/bin/env python
# coding: utf-8
"""
compare the exact electron repulsion energy matrix between electronic states

    Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with the multi-state local-density approximation

    Wᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ
        = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|
          - Cₓ ∫ [D(r)⁴ᐟ³]ᵢⱼ dr
          + a ∫ log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ ) D(r) dr
          + SIC δᵢⱼ

for a range of Be-Be bond lengths.
The matrix density Dᵢⱼ(r) is calculated using CASSCF.

The self-interaction correction for the core electrons is a constant that neither depends
on the geometry nor on the electronic state.

    SIC = - 2 x [ 1/2 (ρ₁ₛᵅ|ρ₁ₛᵅ) - 2¹ᐟ³ Cₓ ∫ ρ₁ₛᵅ(r)⁴ᐟ³ dr  + ∫ ρ₁ₛᵅ(r) εᶜ(ρ₁ₛᵅ) dr ] δᵢⱼ

"""
import json
import numpy

import pyscf.fci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
import pyscf.tools.molden as molden

from msdft.ElectronRepulsionOperators import HartreeLikeFunctional
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
from msdft.KineticOperatorFunctional import LDAThomasFermiFunctional
from msdft.KineticOperatorFunctional import LDAVonWeizsaeckerFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF
from msdft.SelfInteractionCorrection import CoreSelfInteractionCorrection


if __name__ == "__main__":
    print("This calculation can take a while. Please be patient ...")
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # The bond length (in Å), the scan variable
        'bond_length': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
        # electron repulsion matrix
        'W_exact': [],
        # approximate electron repulsion matrix and its constituents.
        'W_approximate': [],
        'J_Hartree': [],
        'K_LDA': [],
        'C_LDA': [],
        # The self-interaction correction for the core orbitals, a constant
        # that does not depend on the geometry and is the same for all
        # electronic states.
        'self_interaction_correction': [],
        # kinetic energy
        'T_exact': [],
        # approximate kinetic energy
        'T_approximate': [],
        'T_ThomasFermi': [],
        'T_vonWeizsaecker': [],
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
            Be {-bond_length/2} 0 0
            Be { bond_length/2} 0 0
            """,
            # A large basis set is needed.
            basis = 'aug-cc-pvqz',
            charge = 0,
            # singlet
            spin = 0,
            # use symmetry
            symmetry = True)
        print(f"Molecular symmetry is used: {mol.topgroup}")

        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()
        # Show orbital energies, occupancies etc., requires hf.verbose > 0.
        #hf.analyze()
        pyscf.tools.molden.dump_scf(hf, '/tmp/hf.molden', ignore_h=False)

        # CAS: 4 electrons in 30 orbitals
        ncas = 30
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
        cas_irrep_nocc = {
            'A1g': 7, 'A1u': 7,
            'E1gx': 2, 'E1gy': 2, 'E1ux': 2, 'E1uy': 2,
            'E2gx': 2, 'E2gy': 2, 'E2ux': 2, 'E2uy': 2
        }

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
        print("CASSCF calculation ...")
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

        # Remove the self-interaction error (SIE) of the electrons in the core orbitals.
        SIE = CoreSelfInteractionCorrection(
            mol,
            exchange_functional_class=LDAExchangeLikeFunctional,
            correlation_functional_class=LDACorrelationLikeFunctional
        ).total_self_interaction_error()
        # The correction should remove the self-interaction error, therefore the minus sign.
        SIC = -SIE

        # Align the signs of the wavefunctions with those at the privious scan point.
        # At the first scan point there is no reference.
        if msmd_ref is not None:
            msmd.align_phases(msmd_ref)
        # Functionals for parts of electron repulsion.
        # J[D]
        hartree_functional = HartreeLikeFunctional(mol)
        # K[D]
        exchange_functional = LDAExchangeLikeFunctional(mol)
        # C[D]
        correlation_functional = LDACorrelationLikeFunctional(mol)
        # kinetic energy
        # T[D] = T_{TF}[D] + 1/9 T_{vW}[D]
        kinetic_functional_TF = LDAThomasFermiFunctional(mol)
        kinetic_functional_vW = LDAVonWeizsaeckerFunctional(mol)

        # exact electron repulsion
        # Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)
        W_exact = msmd.exact_electron_repulsion()
        # approximate multi-state LDA electron repulsion
        # Wᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ
        print("calculating J[D] ...")
        J_Hartree = hartree_functional(msmd)
        print("calculating K[D] ...")
        K_LDA = exchange_functional(msmd)
        print("calculating C[D] ...")
        C_LDA = correlation_functional(msmd)
        W_approximate = J_Hartree - K_LDA + C_LDA + SIC * numpy.eye(nstate)

        # Compare approximate and exact electron repulsion.
        print("=== Electron Repulsion Matrices ===")
        print("W_exact")
        print(W_exact)
        print("W_approximate")
        print(W_approximate)
        print("J Hartree")
        print(J_Hartree)
        print("-K_LDA")
        print(-K_LDA)
        print("C_LDA")
        print(C_LDA)
        print("self-interaction correction for core orbitals")
        print(SIC)

        # relative errors
        relative_errors = abs(W_approximate - W_exact)/(abs(W_exact) + 1.0e-8)
        print("=== Relative Errors ===")
        print("|W(approximate)-W(exact)|/|W(exact)|")
        print(relative_errors)

        # Orbital-free kinetic functionals.
        print("calculating T_FH[D] ...")
        T_TF = kinetic_functional_TF(msmd)
        print("calculating T_vW[D] ...")
        T_vW = kinetic_functional_vW(msmd)
        # The factor 1/9 is explained in section 6.7 "Conventional gradient correction"
        # in Parr & Yang's book.
        T_approximate = T_TF + 1.0/9.0 * T_vW

        # Compare approximate and exact kinetic energy.
        T_exact = msmd.exact_kinetic_energy()
        print("=== Kinetic Energy Matrices ===")
        print("T_exact")
        print(T_exact)
        print("T_approximate")
        print(T_approximate)
        print("T_TF")
        print(T_TF)
        print("T_vW")
        print(T_vW)

        # relative errors
        relative_errors = abs(T_approximate - T_exact)/(abs(T_exact) + 1.0e-8)
        print("=== Relative Errors ===")
        print("|T(approximate)-T(exact)|/|T(exact)|")
        print(relative_errors)

        # Save data for later plotting
        scan_data['bond_length'].append(bond_length)
        scan_data['eigenenergies'].append(casscf.e_states.tolist())
        scan_data['W_exact'].append(W_exact.tolist())
        scan_data['W_approximate'].append(W_approximate.tolist())
        scan_data['J_Hartree'].append(J_Hartree.tolist())
        scan_data['K_LDA'].append(K_LDA.tolist())
        scan_data['C_LDA'].append(C_LDA.tolist())
        scan_data['self_interaction_correction'].append(SIC)
        scan_data['T_exact'].append(T_exact.tolist())
        scan_data['T_approximate'].append(T_approximate.tolist())
        scan_data['T_ThomasFermi'].append(T_TF.tolist())
        scan_data['T_vonWeizsaecker'].append(T_vW.tolist())

        # This scan point becomes the reference for the next scan point.
        msmd_ref = msmd

        # Save intermediate results, in case the calculation is stopped early.
        with open('electron_repulsion_energies.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)
