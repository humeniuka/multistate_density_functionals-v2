#!/usr/bin/env python
# coding: utf-8
"""
plot the eigenenergies of the lowest few electronic states
as a function of the Be-Be bond length.

References
----------
[Be2] Merritt, Jeremy M., Vladimir E. Bondybey, and Michael C. Heaven.
      "Beryllium dimer—caught in the act of bonding."
      Science 324.5934 (2009): 1548-1551.
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy

from pyscf.data.nist import HARTREE2WAVENUMBER

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # Load scan data
    with open('electron_repulsion_energies.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Bond lengths in Å.
    bond_length = numpy.array(scan_data['bond_length'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])

    # Compute the binding energy of the dimer as
    #  De = E(r=r_max) - E(r=r_eq))
    # The experimental well depth of the potential energy curve is
    #  De = 900 cm-1 (see Fig. 3 of [Be2])
    dissociation_limit = eigenenergies[-1,0]
    binding_energy = (dissociation_limit - eigenenergies[:,0].min())
    print(f"Binding energy De = {binding_energy * HARTREE2WAVENUMBER:6.0f} cm-1")

    # number of electronic states
    nstate = eigenenergies[0].shape[0]

    fig = plt.figure(figsize=(4.8,4.8))
    axes = plt.gca()
    
    # Zoom into the bonding region.
    axes_inset = plt.gca().inset_axes([0.35, 0.2, 0.6, 0.4])
    axes_inset.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    axes_inset.set_xlim((2.2, 5.0))
    axes_inset.set_ylim((-500.0, 0.0))

    plt.ylabel(r"adiab. energies / cm$^{-1}$")
    plt.xlabel(r"bond length / $\AA$")

    # All states belong to the Σ+g irrep.
    symmetry_label = [r'$1^1\!\Sigma^+_g$', r'$2^1\!\Sigma^+_g$']
    for i in range(0, nstate):
        line, = plt.plot(
            bond_length,
            (eigenenergies[:,i] - dissociation_limit) * HARTREE2WAVENUMBER,
            lw=2, alpha=0.5,
            label=symmetry_label[i]
        )
        # Zoom in on avoided crossing.
        axes_inset.plot(
            bond_length,
            (eigenenergies[:,i] - dissociation_limit) * HARTREE2WAVENUMBER,
            lw=2, alpha=0.5,
            label=symmetry_label[i],
            color=line.get_color()
        )

        
    plt.legend(loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.25))

    # Otherwise the x-labels are partly cut off.
    plt.tight_layout()

    #plt.savefig("beryllium-dimer_potential_energies.svg")
    #plt.savefig("beryllium-dimer_potential_energies.png", dpi=300)

    plt.show()
