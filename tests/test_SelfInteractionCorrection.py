#!/usr/bin/env python
# coding: utf-8
import numpy
import numpy.testing

import pyscf.gto

import unittest

from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.SelfInteractionCorrection import CoreSelfInteractionCorrection


class TestCoreSelfInteractionCorrection(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = 'sto-3g',
                # doublet
                spin = 1),
            # many electrons
            'oxygen atom': pyscf.gto.M(
                atom = 'O 0 0 0',
                basis = 'sto-3g',
                # singlet
                spin = 0),
            'oxygen molecule': pyscf.gto.M(
                atom = 'O 0.0 0.0 0.0;  O 0.0 0.0 1.21',
                basis = 'sto-3g',
                # singlet
                spin = 0),
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
        }
        return molecules

    def test_total_self_interaction_error(self):
        """
        Check that the self-interaction of the cores of different atoms are added correctly.
        """
        molecules = self.create_test_molecules()
        SIE_water = CoreSelfInteractionCorrection(
            molecules['water']).total_self_interaction_error()
        SIE_oxygen_molecule = CoreSelfInteractionCorrection(
            molecules['oxygen molecule']).total_self_interaction_error()
        SIE_oxygen_atom = CoreSelfInteractionCorrection(
            molecules['oxygen atom']).total_self_interaction_error()
        SIE_hydrogen_atom = CoreSelfInteractionCorrection(
            molecules['hydrogen atom']).total_self_interaction_error()

        # Hydrogen has no core electrons
        self.assertAlmostEqual(SIE_hydrogen_atom, 0.0)
        # SIE is determined by element type, the geometry does not matter.
        self.assertAlmostEqual(2*SIE_oxygen_atom, SIE_oxygen_molecule)
        self.assertAlmostEqual(SIE_oxygen_atom, SIE_water)

        # Check the actual value (in Hartree)
        self.assertAlmostEqual(0.5013, SIE_oxygen_atom, places=3)

    def test_total_self_interaction_error_LiF(self):
        """
        Check that the self-interaction error of the core orbitals for LiF
        with the default LDA exchange-correlation functional
        (Dirac exchange and Chachyo correlation) is as expected.
        """
        mol = pyscf.gto.M(
            atom = 'Li 0.0 0.0 0.0; F 0.0 0.0 1.564',
            basis = '6-31g')

        SIE_lithium_fluoride = CoreSelfInteractionCorrection(mol).total_self_interaction_error()

        # Check the actual value (in Hartree)
        self.assertAlmostEqual(0.5769, SIE_lithium_fluoride, places=3)

    def test_total_self_interaction_error_lda_vs_lsda(self):
        """
        Check that the same self-interaction error is obtained for
        LDA and LSDA.
        """
        mol = pyscf.gto.M(
            atom = 'Li 0.0 0.0 0.0; F 0.0 0.0 1.564',
            basis = '6-31g')

        SIE_lsda = CoreSelfInteractionCorrection(
            mol,
            exchange_functional_class = LSDAExchangeLikeFunctional,
            correlation_functional_class = LDACorrelationLikeFunctional
        ).total_self_interaction_error()
        SIE_lda = CoreSelfInteractionCorrection(
            mol,
            exchange_functional_class = LDAExchangeLikeFunctional,
            correlation_functional_class = LDACorrelationLikeFunctional
        ).total_self_interaction_error()

        self.assertAlmostEqual(SIE_lsda, SIE_lda, places=3)

    def test_self_interaction_energy_of_core(self):
        """
        Test the static function
        `CoreSelfInteractionCorrection.self_interaction_energy_of_core`
        """
        molecules = self.create_test_molecules()
        self_interaction_correction = CoreSelfInteractionCorrection(molecules['oxygen atom'])
        SIE_oxygen_atom = self_interaction_correction.total_self_interaction_error()

        SIEs_core_orbitals = self_interaction_correction.self_interaction_energy_of_core(
            'O', 'sto-3g')
        # Oxygen has one 1s core orbital
        self.assertEqual(1, len(SIEs_core_orbitals))
        self.assertAlmostEqual(SIE_oxygen_atom, numpy.sum(SIEs_core_orbitals))

        # Check heavy atoms with multiple core orbitals.
        # In Silicon the 1s,2s,2px,2py,2pz orbitals are part of the core
        SIEs_core_orbitals = self_interaction_correction.self_interaction_energy_of_core(
            'Si', 'sto-3g')
        self.assertEqual(5, len(SIEs_core_orbitals))

    def test_raises_exception(self):
        """ Check that an exception is raise if the functional arguments have the wrong type. """
        mol = self.create_test_molecules()['oxygen atom']
        # correlation functional has to be a subclass of ExchangeCorrelationLikeFunctional,
        # not a string.
        with self.assertRaises(ValueError):
            CoreSelfInteractionCorrection(mol, correlation_functional_class=str)
        # exchange functional has to be a subclass of ExchangeCorrelationLikeFunctional,
        # not a string.
        with self.assertRaises(ValueError):
            CoreSelfInteractionCorrection(mol, exchange_functional_class=str)


if __name__ == "__main__":
    unittest.main(failfast=True)
