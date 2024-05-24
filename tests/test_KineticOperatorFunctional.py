#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod

import numpy
import numpy.linalg as la
import numpy.testing

import pyscf.dft
import pyscf.dft.libxc
import pyscf.fci
import pyscf.gto
import pyscf.scf

from tqdm import tqdm
import unittest

from msdft.BasisTransformation import BasisTransformation
from msdft.KineticOperatorFunctional import EigendecompositionKineticFunctional
from msdft.KineticOperatorFunctional import EigendecompositionKineticFunctionalII
from msdft.KineticOperatorFunctional import GGALeeLeeParr91KineticFunctional
from msdft.KineticOperatorFunctional import KineticOperatorFunctional
from msdft.KineticOperatorFunctional import LDAThomasFermiFunctional
from msdft.KineticOperatorFunctional import LSDAThomasFermiFunctional
from msdft.KineticOperatorFunctional import LDAVonWeizsaeckerFunctional
from msdft.KineticOperatorFunctional import LSDAVonWeizsaeckerFunctional
from msdft.KineticOperatorFunctional import MatrixSquareRootKineticFunctional
from msdft.KineticOperatorFunctional import LDAVonWeizsaecker1eFunctional
from msdft.KineticOperatorFunctional import LSDAVonWeizsaecker1eFunctional
from msdft.KineticOperatorFunctional import LSDAVonWeizsaecker1eFunctionalII
from msdft.MultistateMatrixDensity import MultistateMatrixDensity
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


class LSDAVonWeizsaeckerFunctionalSingleState(object):
    """
    The von Weizsäcker density functional of the kinetic energy:

                        (∇ρ(↑↑))²         (∇ρ(↓↓))²
           T[ρ] = ∫ 1/8 --------- + ∫ 1/8 ---------
                          ρ(↑↑)             ρ(↓↓)

    Note that the kinetic energy is calculated separately for the spin-up and
    the spin-down densities and then summed.
    """
    def __init__(self, mol, level=8):
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        Compute the von Weizsaecker kinetic energy for the density of a single
        electronic state.

        :param msmd: A multistate density matrix with only a single electronic state
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return kinetic_energy: A 1x1 matrix with the scalar kinetic energy.
        :rtype kinetic_energy: numpy.ndarray of shape (1,1)
        """
        # number of electronic states
        nstate = msmd.number_of_states
        assert nstate == 1, \
           "The von Weizsaecker functional is only defined for a single electronic state."
        # up or down spin
        nspin = 2

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(self.grids.coords)
        # Trace out electronic states to get tr(D)(r)
        trace_D = numpy.einsum('siir->sr', D)

        # matrix element of the kinetic energy operator <i|Top|j>
        kinetic_matrix = numpy.zeros((nstate,nstate))

        # Loop over spins. For kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue

            # von Weizsäcker
            KED = 1.0/8.0 * (
                numpy.einsum('ikar,kjar->ijr', grad_D[s,...], grad_D[s,...]) /
                numpy.expand_dims(trace_D[s,...], axis=(0,1)))

            # The matrix of the kinetic energy operator in the subspace is obtained
            # by integration T_{i,j}(r) over space
            #
            #  ∫ -1/2 ϕᵢ*(r) ∇²ϕⱼ(r) = ∫ 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
            #
            kinetic_matrix += numpy.einsum('r,ijr->ij', self.grids.weights, KED)

        return kinetic_matrix


class LDAVonWeizsaeckerFunctionalSingleState(object):
    """
    The von Weizsäcker density functional of the kinetic energy:

                        (∇ρ)²
           T[ρ] = ∫ 1/8 -----
                          ρ

    Note that the kinetic energy is calculated from the total density ρ = ρ(↑↑) + ρ(↓↓).
    """
    def __init__(self, mol, level=8):
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        Compute the von Weizsaecker kinetic energy for the density of a single
        electronic state.

        :param msmd: A multistate density matrix with only a single electronic state
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return kinetic_energy: A 1x1 matrix with the scalar kinetic energy.
        :rtype kinetic_energy: numpy.ndarray of shape (1,1)
        """
        # number of electronic states
        nstate = msmd.number_of_states
        assert nstate == 1, \
           "The von Weizsaecker functional is only defined for a single electronic state."

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(self.grids.coords)

        # Sum over spins to get total D = Dᵅ(r) + Dᵝ(r)
        total_density = D[0,...] + D[1,...]
        # and its gradient
        grad_total_density = grad_D[0,...] + grad_D[1,...]

        # Trace out electronic states to get tr(D)(r)
        trace_total_density = numpy.einsum('iir->r', total_density)

        # matrix element of the kinetic energy operator <i|Top|j>
        kinetic_matrix = numpy.zeros((nstate,nstate))

        # von Weizsäcker
        KED = 1.0/8.0 * (
            numpy.einsum('ikar,kjar->ijr', grad_total_density, grad_total_density) /
            numpy.expand_dims(trace_total_density, axis=(0,1))
        )

        # The matrix of the kinetic energy operator in the subspace is obtained
        # by integration T_{i,j}(r) over space
        #
        #  ∫ -1/2 ϕᵢ*(r) ∇²ϕⱼ(r) = ∫ 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
        #
        kinetic_matrix += numpy.einsum('r,ijr->ij', self.grids.weights, KED)

        return kinetic_matrix


class KineticFunctionalTests(ABC):
    """
    Abstract base class for all kinetic energy functional tests.
    It contains functions needed by all tests.
    """
    @property
    @abstractmethod
    def kinetic_functional_class(self):
        """
        The subclass of :class:`~.KineticOperatorFunctional` for which
        the respective unit test is written.
        """
        pass

    def create_test_molecules_1electron(self):
        """ dictionary with 1-electron molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            'hydrogen atom (large basis set)': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = 'aug-cc-pvtz',
                # doublet
                spin = 1),
            'hydrogen molecular ion': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 1,
                # doublet
                spin = 1),
        }
        return molecules

    def create_test_molecules(self):
        """ dictionary with molecules to run the tests on """
        molecules = {
            # H2
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 -0.375; H 0 0 0.375',
                basis = '6-31g',
                # singlet
                spin = 0),
            # 3-electron systems, one unpaired spin
            'lithium atom': pyscf.gto.M(
                atom = 'Li 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            # 4-electron system, closed shell
            'lithium hydride': pyscf.gto.M(
                atom = 'Li 0 0 0; H 0 0 1.60',
                basis = '6-31g',
                # singlet
                spin = 0),
        }
        return molecules

    def create_test_molecules_closed_shell(self):
        """ dictionary with molecules to run the tests on """
        molecules = {
            # H2
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 -0.375; H 0 0 0.375',
                basis = '6-31g',
                # singlet
                spin = 0),
            # 4-electron system, closed shell
            'lithium hydride': pyscf.gto.M(
                atom = 'Li 0 0 0; H 0 0 1.60',
                basis = '6-31g',
                # singlet
                spin = 0),
        }
        return molecules

    def create_matrix_density(self, mol, nstate=4):
        """
        Compute multistate matrix density for the lowest few excited states
        of a small molecule using full configuration interaction.

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of excited states to calculate
        :type nstate: positive int

        :return: multistate matrix density
        :rtype: MultistateMatrixDensity
        """
        # call static method
        return MultistateMatrixDensityFCI.create_matrix_density(
            mol, nstate=nstate, spin_symmetry=False, raise_error=False)

    def check_exact_kinetic_energy(self, mol, nstate=1):
        """
        For molecules with a single electron, the kinetic energy functional
        should yield the exact kinetic energy.

        :param mol: A test molecule with only one electron.
        :type mol: gto.Mole

        :param nstate: Number of electronic states in the subspace.
           The full CI problem is solved for the lowest nstate states.
        :type nstate: int > 0
        """
        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.kinetic_functional_class, KineticOperatorFunctional)
        # These tests are expected to work only for one-electron systems.
        assert sum(mol.nelec) == 1, "This test only works for 1-electron systems."
        assert nstate > 0, "The number of electronic states has to be > 0."

        # functional for kinetic operator, T[D(r)]
        kinetic_functional = self.kinetic_functional_class(mol, level=5)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=nstate)

        # Evaluate T[D(r)]
        T_msdft = kinetic_functional(msmd)

        # The exact kinetic energy matrix is calculated by contracting the (transition)
        # density matrices in the AO basis with the kinetic energy matrix.
        T_exact = msmd.exact_1e_operator(intor='int1e_kin')

        numpy.testing.assert_almost_equal(T_msdft, T_exact, decimal=5)

    def check_chunk_size(self, mol):
        """
        Compute kinetic matrix with different chunk sizes.
        """
        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.kinetic_functional_class, KineticOperatorFunctional)
        # functional for kinetic operator, T[D(r)]
        kinetic_functional = self.kinetic_functional_class(mol, level=1)

        msmd = self.create_matrix_density(mol, nstate=3)
        # Tij with default chunk size
        kinetic_matrix_ref = kinetic_functional(msmd)
        # Increase the number of chunks by reducing the available memory
        # per chunk to 2**22 (~4 Mb) or 2**23 (~ 8Mb) bytes.
        for memory in [2**22, 2**23]:
            kinetic_matrix = kinetic_functional(msmd, available_memory=memory)

            numpy.testing.assert_almost_equal(
                kinetic_matrix, kinetic_matrix_ref)

    def test_chunk_size(self):
        """
        Check that the kinetic energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)

    def check_transformation(self, mol, nstate=1):
        """
        As an analytical matrix density functional, the kinetic energy T[D(r)]
        should transform under a basis transformation L as

          T[L D(r) Lᵗ] = L T[D(r)] Lᵗ
        """
        assert nstate > 0
        # First the electronic eigenstates are determined using
        # full configuration interaction.
        rhf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        rhf.verbose = 0
        # compute self-consistent field
        rhf.kernel()

        fci = pyscf.fci.FCI(mol, rhf.mo_coeff)
        # Solve for one state more than requested to avoid
        # problems when nstate == 1.
        fci.nroots = nstate+1
        fci_energies, fcivecs = fci.kernel()
        # Remove the additional state again.
        if len(fcivecs) == nstate+1:
            fcivecs = fcivecs[:-1]
        # For small basis sets, there can be fewer states than requested.
        nstate = len(fcivecs)

        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.kinetic_functional_class, KineticOperatorFunctional)
        # functional for the kinetic energy matrix, T[D(r)]
        kinetic_functional = self.kinetic_functional_class(mol, level=1)

        # random transformation L
        basis_transformation = BasisTransformation.random(nstate)

        # The multistate density matrix D(r)
        msmd = MultistateMatrixDensityFCI(mol, rhf, fci, fcivecs)
        # Evaluate T[D(r)] by integration on the grid.
        kinetic_matrix = kinetic_functional(msmd)
        # Transform the operator, L XC[D(r)] Lᵗ
        kinetic_matrix_transformed = basis_transformation.transform_operator(kinetic_matrix)

        # To compute L D(r) Lᵗ we apply the basis transformation to the CI vectors.
        fcivecs_transformed = basis_transformation.transform_vector(fcivecs)
        # The multistate density matrix L D(r) Lᵗ in the transformed basis
        msmd_transformed = MultistateMatrixDensityFCI(mol, rhf, fci, fcivecs_transformed)
        # Evaluate T[L D(r) Lᵗ] by integration on the grid.
        kinetic_matrix_from_transformed_D = kinetic_functional(msmd_transformed)

        numpy.testing.assert_almost_equal(kinetic_matrix_from_transformed_D, kinetic_matrix_transformed)

        # Finally, check that T[D(r)] is a symmetric matrix.
        numpy.testing.assert_almost_equal(kinetic_matrix, kinetic_matrix.T)

    def test_transformation(self):
        """
        Verify that the kinetic energy matrix transforms correctly under basis changes.
        """
        for name, mol in tqdm(
                self.create_test_molecules_1electron().items()):
            for nstate in tqdm([2,3]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_transformation(mol, nstate=nstate)


class TestLSDAVonWeizsaecker1eFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LSDAVonWeizsaecker1eFunctional

    def test_1electron_systems(self):
        """ Check that the kinetic energy functional is exact for one-electron systems """
        for name, mol in tqdm(
                self.create_test_molecules_1electron().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_kinetic_energy(mol, nstate=nstate)

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LSDAVonWeizsaecker1eFunctional(mol)
                kinetic_functional_single = LSDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestLDAVonWeizsaecker1eFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LDAVonWeizsaecker1eFunctional

    def test_1electron_systems(self):
        """ Check that the kinetic energy functional is exact for one-electron systems """
        for name, mol in tqdm(
                self.create_test_molecules_1electron().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_kinetic_energy(mol, nstate=nstate)

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LDAVonWeizsaecker1eFunctional(mol)
                kinetic_functional_single = LDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestLSDAVonWeizsaeckerFunctional(KineticFunctionalTests, unittest.TestCase):
    """
    NOTE: This von-Weizsaecker-like kinetic energy function is NOT exact
          for 1-electron systems but it performs better for many electron systems.
    """
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LSDAVonWeizsaeckerFunctional

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LSDAVonWeizsaeckerFunctional(mol)
                kinetic_functional_single = LSDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestLDAVonWeizsaeckerFunctional(KineticFunctionalTests, unittest.TestCase):
    """
    NOTE: This von-Weizsaecker-like kinetic energy function is NOT exact
          for 1-electron systems.
    """
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LDAVonWeizsaeckerFunctional

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LDAVonWeizsaeckerFunctional(mol)
                kinetic_functional_single = LDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestLSDAVonWeizsaecker1eFunctionalII(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LSDAVonWeizsaecker1eFunctionalII

    def test_1electron_systems(self):
        """ Check that the kinetic energy functional is exact for one-electron systems """
        for name, mol in tqdm(
                self.create_test_molecules_1electron().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_kinetic_energy(mol, nstate=nstate)

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LSDAVonWeizsaecker1eFunctionalII(mol)
                kinetic_functional_single = LSDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class LDAThomasFermiFunctionalSingleState(object):
    """
    The Thomas-Fermi density functional of the kinetic energy:

           T[ρ] = 3/10 (3π²)²ᐟ³ ∫ ρ(r)⁵ᐟ³ dr

    """
    def __init__(self, mol, level=8):
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        Compute the Thomas-Fermi kinetic energy for the density of a single
        electronic state.

        :param msmd: A multistate density matrix with only a single electronic state
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return kinetic_matrix: A 1x1 matrix with the scalar kinetic energy.
        :rtype kinetic_energy: numpy.ndarray of shape (1,1)
        """
        # number of electronic states
        nstate = msmd.number_of_states
        assert nstate == 1, \
           "The von Weizsaecker functional is only defined for a single electronic state."

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(self.grids.coords)

        # Sum over spins.
        Dtot = D.sum(axis=0)

        # Thomas-Fermi kinetic energy density.
        KED = 3.0/10.0 * pow(3.0*numpy.pi**2, 2.0/3.0) * pow(Dtot, 5.0/3.0)

        # The matrix of the kinetic energy operator in the subspace is obtained
        # by integration KED_{i,j}(r) over space.
        kinetic_matrix = numpy.einsum('r,ijr->ij', self.grids.weights, KED)

        return kinetic_matrix


class TestLSDAThomasFermiFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LSDAThomasFermiFunctional

    def test_Thomas_Fermi_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the Thomas-Fermi functional for a closed shell molecule.
        """
        for name, mol in tqdm(self.create_test_molecules_closed_shell().items()):
            # scalar D(r) from single electronic state
            msmd = self.create_matrix_density(mol, nstate=1)

            # functionals for kinetic operator, T[D(r)]
            kinetic_functional_multi = LSDAThomasFermiFunctional(mol)
            kinetic_functional_single = LDAThomasFermiFunctionalSingleState(mol)

            # Compare the multistate and the single-state TF functionals.
            kinetic_matrix_multi = kinetic_functional_multi(msmd)
            kinetic_matrix_single = kinetic_functional_single(msmd)

            with self.subTest(molecule=name):
                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestLDAThomasFermiFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return LDAThomasFermiFunctional

    def test_Thomas_Fermi_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the Thomas-Fermi functional for a closed shell molecule.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            # scalar D(r) from single electronic state
            msmd = self.create_matrix_density(mol, nstate=1)

            # functionals for kinetic operator, T[D(r)]
            kinetic_functional_multi = LDAThomasFermiFunctional(mol)
            kinetic_functional_single = LDAThomasFermiFunctionalSingleState(mol)

            # Compare the multistate and the single-state TF functionals.
            kinetic_matrix_multi = kinetic_functional_multi(msmd)
            kinetic_matrix_single = kinetic_functional_single(msmd)

            with self.subTest(molecule=name):
                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestEigendecompositionKineticFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return EigendecompositionKineticFunctional

    def test_1electron_systems(self):
        """ Check that the kinetic energy functional is exact for one-electron systems """
        for name, mol in tqdm(
                self.create_test_molecules_1electron().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_kinetic_energy(mol, nstate=nstate)

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LSDAVonWeizsaecker1eFunctional(mol)
                kinetic_functional_single = LSDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)

    def test_chunk_size(self):
        """
        Check that the kinetic energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        # This functional only works if there are no repeated eigenvalues with
        # repeated eigenvalue derivatives.
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)


class TestEigendecompositionKineticFunctionalII(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return EigendecompositionKineticFunctionalII

    def test_1electron_systems(self):
        """ Check that the kinetic energy functional is exact for one-electron systems """
        for name, mol in tqdm(
                self.create_test_molecules_1electron().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_kinetic_energy(mol, nstate=nstate)

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LSDAVonWeizsaecker1eFunctional(mol)
                kinetic_functional_single = LSDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)

    def test_chunk_size(self):
        """
        Check that the kinetic energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        # This functional only works if there are no repeated eigenvalues with
        # repeated eigenvalue derivatives.
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)


class TestMatrixSquareRootKineticFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return MatrixSquareRootKineticFunctional

    def test_von_Weizsaecker_functional(self):
        """
        Check that for a single electronic state the multistate kinetic energy functional
        reduces to the von Weizsäcker functional.
        """
        for name, mol in tqdm(self.create_test_molecules_1electron().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for kinetic operator, T[D(r)]
                kinetic_functional_multi = LSDAVonWeizsaecker1eFunctional(mol)
                kinetic_functional_single = LSDAVonWeizsaeckerFunctionalSingleState(mol)

                # Compare the multistate and the single-state vW functionals.
                kinetic_matrix_multi = kinetic_functional_multi(msmd)
                kinetic_matrix_single = kinetic_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    kinetic_matrix_multi, kinetic_matrix_single)


class TestGGALeeLeeParr91KineticFunctional(KineticFunctionalTests, unittest.TestCase):
    @property
    def kinetic_functional_class(self):
        """ The functional to be tested. """
        return GGALeeLeeParr91KineticFunctional

    def test_leeleeparr91_kinetic_functional_implementation(self):
        """
        Check that the implementation of Lee, Lee & Parr's 1991 GGA kinetic functional
        gives the same energy as the libxc library (GGA_K_LLP) for a range of electron densities.

        References
        ----------
        [libxc] S. Lehtola et al. (2018), Software X 7, 1-5,
            "Recent developments in libxc —
            A comprehensive library of functionals for density functional theory"
            https://doi.org/10.1016/j.softx.2017.11.002
        """
        for name, mol in tqdm({
                # combine all test molecules into a single dictionary
                **self.create_test_molecules_1electron(),
                **self.create_test_molecules()}.items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # Evaluate the kinetic energy, T[D(r)], using the multi-state functional.
                kinetic_functional_multi = self.kinetic_functional_class(mol)
                kinetic_matrix_multi = kinetic_functional_multi(msmd)

                # Evaluate the GGA kinetic energy of a single state using the
                # implementation of libxc.
                grids = pyscf.dft.gen_grid.Grids(mol)
                grids.level = 8
                grids.build()
                # number of grid points
                ncoord = grids.coords.shape[0]

                D, grad_D, _ = msmd.evaluate(grids.coords)
                # rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
                # For a spin-polarized GGA functional we have to provide
                # rho_ud = ((den_u,grad_xu,grad_yu,grad_zu,0,0)
                #           (den_d,grad_xd,grad_yd,grad_zd,0,0))
                rho_ud = numpy.zeros((2, 6, ncoord))
                rho_ud[:,0,:] = D[:,0,0,:]
                rho_ud[:,1:4,:] = grad_D[:,0,0,:,:]

                # The LLP functional is spin-polarized
                t, _, _, _ = pyscf.dft.libxc.eval_xc('GGA_K_LLP,', rho_ud, spin=1)
                # Integrate over space.
                # libxc divides the kinetic energy per particle by the total spin-summed density,
                #   t[ρᵅ,ρᵝ] = 1/ρ * (ρᵅ t[ρᵅ] + ρᵝ t[ρᵝ]),
                # so that the total exchange energy is calculated as
                #   T[ρ] = ∫ ρ t[ρᵅ,ρᵝ] dr.
                # (see Eqn. (4) in [libxc])
                rho = rho_ud[0,0,:] + rho_ud[1,0,:]
                kinetic_matrix_single = numpy.array([[
                        numpy.sum(grids.weights * (rho * t))
                    ]])

                # Compare the multistate and the single-state (libxc) kinetic functionals.
                numpy.testing.assert_almost_equal(
                    kinetic_matrix_single, kinetic_matrix_multi, decimal=6)


if __name__ == "__main__":
    unittest.main(failfast=True)
