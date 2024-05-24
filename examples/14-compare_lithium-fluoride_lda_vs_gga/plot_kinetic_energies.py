#!/usr/bin/env python
# coding: utf-8
"""
plot the exact matrix elements of the kinetic energy operator

  Tᵢⱼ = <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψⱼ>

and the multi-state LDA and GGA approximations.

The  the diagonal (i=j) and off-diagonal (i≠j) elements for all
scan geometries are plotted separately.
"""
import json
import matplotlib.lines
import matplotlib.pyplot as plt

import numpy

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # Load scan data
    with open('electron_repulsion_energies-lda.json', 'r') as filehandle:
        scan_data_lda = json.load(filehandle)
    with open('electron_repulsion_energies-gga.json', 'r') as filehandle:
        scan_data_gga = json.load(filehandle)

    # Bond lengths are in Angstrom.
    bond_length_lda = numpy.array(scan_data_lda['bond_length'])
    bond_length_gga = numpy.array(scan_data_gga['bond_length'])
    # All energies are in Hartree.
    T_exact = numpy.array(scan_data_lda['T_exact'])
    T_approximate_lda = numpy.array(scan_data_lda['T_approximate'])
    T_approximate_gga = numpy.array(scan_data_gga['T_approximate'])

    # Only relative energies are important.
    # The diagonal elements are shifted rigidly so that the energy
    # at the last scan point is 0.
    shift_exact = -T_exact[-1,0,0]
    shift_lda = -T_approximate_lda[-1,0,0]
    shift_gga = -T_approximate_gga[-1,0,0]
    
    # number of electronic states
    nstate = T_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of kinetic energy operator.
    axes[0].set_ylabel(r"kinetic energy $T_{II}$ (shifted) / $E_h$")
    axes[0].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            bond_length_lda, T_exact[:,i,i] + shift_exact,
            lw=2, alpha=0.5,
            label=rf"$T_{{{i},{i}}}$")
        axes[0].plot(
            bond_length_lda, T_approximate_lda[:,i,i] + shift_lda,
            ls="--", color=line.get_color())
        axes[0].plot(
            bond_length_gga, T_approximate_gga[:,i,i] + shift_gga,
            ls="-.", color=line.get_color())

    axes[0].legend(title="$\mathbf{(e)}$ diagonal")

    # Off-diagonal elements kinetic energy operator
    axes[1].set_ylabel(r"kinetic energy $T_{IJ}$ / $E_h$")
    axes[1].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                bond_length_lda, T_exact[:,i,j],
                lw=2, alpha=0.5,
                label=rf"$T_{{{i},{j}}}$")
            axes[1].plot(
                bond_length_lda, T_approximate_lda[:,i,j],
                ls="--", color=line.get_color())
            axes[1].plot(
                # The global phase of the wavefunctions is arbitary.
                # The phase of the off-diagonal matrix element is chosen so
                # that it agrees with the (random) phase of the exact results.
                bond_length_gga, (-1) * T_approximate_gga[:,i,j],
                ls="-.", color=line.get_color())

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(f)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
    dotdashed_line = matplotlib.lines.Line2D([], [], ls="-.", color="black")
    fig.legend(
        [solid_line, dashed_line, dotdashed_line],
        [
            r"$T_{IJ}^{\text{exact}}$", # = \langle \Psi_I \vert -\frac{1}{2} \sum_a \nabla_a^2 \vert \Psi_J \rangle$ (exact)",
            r"$T_{IJ}^{\text{LDA}}$", # = T_{TF}[\mathbf{D}]_{IJ} + 1/9 T_{vW}[\mathbf{D}]_{IJ}$"
            r"$T_{IJ}^{\text{GGA}}$", # = T_{LLP91}[\mathbf{D}]_{IJ}$"
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=3
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    #plt.savefig("kinetic_energies.svg")
    #plt.savefig("kinetic_energies.png", dpi=300)

    plt.show()
