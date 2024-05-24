#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In the mean field approximation there is a spurious interaction of an electron with its own
copy in the total charge density. In reality it should only interact with the remaining n-1
electrons. In the core region and at the tails of the electron density outside of the molecule
the self-interaction error is particularly bad.

Since in most functionals the self-interaction in the Hartree-term is not cancelled properly
by the exchange/correlation term, the self-interaction error (SIE) leads to a increase of the
electron repulsion in DFA relative to the full CI reference. This constant shift can be removed
by simply subtracting the SIE for each orbital from the electron repulsion energies for each
electronic state.

Let ρ₁ₛ be the density of a 1s core orbital. Then the SIE from that orbital is

  SIE(1s) = 2 x ( J[ρ₁ₛᵅ] - K[ρ₁ₛᵅ(r)] + C[ρ₁ₛᵅ] )

J[ρ₁ₛᵅ], -K[ρ₁ₛᵅ] and C[ρ₁ₛᵅ] are the Hartree-, exchange- and correlation-functionals,
respectively. The factor two comes from the double occupancy of the core orbital.

For LDA the SIE becomes

  SIE(1s)^LDA = 2 x [ 1/2 (ρ₁ₛᵅ|ρ₁ₛᵅ) - 2¹ᐟ³ Cₓ ∫ ρ₁ₛᵅ(r)⁴ᐟ³ dr + ∫ ρ₁ₛᵅ(r) εᶜ(ρ₁ₛᵅ) dr ]

where (ρ₁ₛᵅ|ρ₁ₛᵅ)=(1sᵅ1sᵅ|1sᵅ1sᵅ) is the electrostatic interaction of the density with itself.
"""
import becke
import collections
import numpy

import pyscf.data
from pyscf.dft import numint
import pyscf.scf

from msdft.ElectronRepulsionOperators import ExchangeCorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import HartreeLikeFunctional
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.ElectronRepulsionOperators import POLARIZED, UNPOLARIZED
from msdft.MultistateMatrixDensity import CoreOrbitalDensities


class CoreSelfInteractionCorrection(object):
    def __init__(
            self,
            mol : pyscf.gto.Mole,
            exchange_functional_class = LSDAExchangeLikeFunctional,
            # A single orbital is fully spin-polarizated, therefore the ferromagnetic
            # correlation energy should be used. However, since the difference between the two
            # is less than 10%, the paramagnetic LDA correlation is used to keep things simple.
            correlation_functional_class = LDACorrelationLikeFunctional,
            level=8):
        """
        The self-interaction error of the core orbitals.

        :param mol: The molecule contains the information about the elements
            and the basis set. The self-interaction corrections from all core
            orbitals in the molecule are added.
        :type mol: pyscf.gto.Mole

        :param exchange_functional_class: The effect of this exchange functional
            on the core electrons is removed.
        :type exchange_functional_class: subclass of :class:`~.ExchangeCorrelationLikeFunctional`

        :param correlation_functional_class: The effect of this correlation functional
            on the core electrons is removed.
        :type correlation_functional_class: subclass of class:`~.ExchangeCorrelationLikeFunctional`

        :param level: The level (3-8) controls the number of grid points
            in the integration grid.
        :type level: int
        """
        # The SIE of the core orbitals only depends on the elemental composition of the molecule,
        # the geometry is irrelevant.
        self.mol = mol
        # Check types.
        if not issubclass(exchange_functional_class, ExchangeCorrelationLikeFunctional):
            raise ValueError(
                "Argument `exchange_functional_class` has to be a subclass "
                "of `ExchangeCorrelationLikeFunctional`.")
        self.exchange_functional_class = exchange_functional_class
        if not issubclass(correlation_functional_class, ExchangeCorrelationLikeFunctional):
            raise ValueError(
                "Argument `correlation_functional_class` has to be a subclass "
                "of `ExchangeCorrelationLikeFunctional`.")
        self.correlation_functional_class = correlation_functional_class

    def self_interaction_energy_of_core(self, element : str, basis: str):
        """
        compute the self-interaction energy of the core electrons for a single
        element using the supplied basis set.

        The core orbitals are determined by a Hartree-Fock calculation on the
        isolated atom.

        :param element: The name of the element (e.g. 'C' or 'N')
            for which the self-interaction of the core electrons
            should be calculated.
        :type element: str

        :param basis: The basis set (e.g. 'sto-3g')
        :type basis: str

        :return:
            The self-interaction energy in Hartree for each core orbital.
            The factor 2 from the double occupation is already included.
        :rtype: float numpy.ndarray of shape (ncore,)
        """
        # Compute the density of each core orbital, Dᵢᵢ(r) = |ϕᵢ(r)|²
        core_densities = CoreOrbitalDensities(element, basis)

        # `number_of_states` is used to store the number of core orbitals.
        if core_densities.number_of_states == 0:
            # There are no core orbitals.
            return numpy.array([])

        # Hartree-part of self-interaction,
        # J[ρᵅ] = 1/2 (1sᵅ1sᵅ|1sᵅ1sᵅ)
        #       = 1/2 ∫∫' ρᵅ(r) ρᵅ(r') / |r-r'|
        #       = 1/2 ∫ ρᵅ(r) Vᵅ(r)
        hartree_functional = HartreeLikeFunctional(core_densities.mol)
        self_interaction_energy_J = hartree_functional(core_densities)

        # Exchange-part of self-interaction, -K[ρᵅ]
        exchange_functional = self.exchange_functional_class(core_densities.mol)
        # The density of a singly occupied core orbital is fully spin-polarized.
        # If an unpolarized functional is used (e.g. LDA instead of LSDA) a factor
        # of 2¹ᐟ³ needs to be inserted, since
        #   K^{LDA}[ρᵅ]  = Cₓ ∫ ρᵅ(r)⁴ᐟ³
        #   K^{LSDA}[ρᵅ] = 2¹ᐟ³ Cₓ ∫ ρᵅ(r)⁴ᐟ³.
        if exchange_functional.spin_type == UNPOLARIZED:
            spin_factor = pow(2.0, 1.0/3.0)
        else:
            spin_factor = 1.0
        self_interaction_energy_X = -spin_factor * exchange_functional(core_densities)

        # Correlation-part of self-interaction, C[ρᵅ].
        correlation_functional = self.correlation_functional_class(core_densities.mol)
        self_interaction_energy_C = correlation_functional(core_densities)

        # Combine contributions from direct and indirect part of Coulomb energy.
        # The factor 2 comes from the fact that the core electron is doubly occupied.
        # In the density functional approximation, the interaction of each electron
        # with itself is not excluded.
        self_interaction_energy = 2 * (
            self_interaction_energy_J +
            self_interaction_energy_X +
            self_interaction_energy_C
        )

        # The diagonal elements of the matrix (J - K + C) contain the self interaction
        # energies for each core orbital, the off-diagonal elements are zero.
        # Extract diagonal.
        self_interaction_energies = numpy.diag(self_interaction_energy)

        return self_interaction_energies

    def total_self_interaction_error(self) -> float:
        """
        The self-interaction energies (SIE) from all core orbitals in the molecule are
        calculated and summed. The basis set attached to the molecule is used.

        :return: Total core self-interaction energy (in Hartree)
        :rtype: float
        """
        elements = [
            self.mol.atom_pure_symbol(atom_index) for atom_index in range(0, self.mol.natm)
        ]
        # Count how often an element occurs in the molecule
        element_counts = collections.Counter(elements)

        total_core_SIE = 0.0
        # Loop over unique elements in the molecule.
        for element, count in element_counts.items():
            # SIE for all core orbitals of one element.
            core_SIEs = self.self_interaction_energy_of_core(element, self.mol.basis)
            # Sum contributions from all atoms of the same element.
            total_core_SIE += count * numpy.sum(core_SIEs)

        return total_core_SIE
