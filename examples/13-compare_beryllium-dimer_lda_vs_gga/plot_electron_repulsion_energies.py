#!/usr/bin/env python
# coding: utf-8
"""
plot the exact matrix elements of the electron repulsion operator

  Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

and their multi-state LDA and GGA approximations

  Wᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ

for the diagonal (i=j) and off-diagonal (i≠j) elements for all
scan geometries.
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
    W_exact = numpy.array(scan_data_lda['W_exact'])
    W_approximate_lda = numpy.array(scan_data_lda['W_approximate'])
    W_approximate_gga = numpy.array(scan_data_gga['W_approximate'])

    # Only relative energies are important.
    # The diagonal elements are shifted rigidly so that the energy
    # at the last scan point is 0.
    shift_exact = -W_exact[-1,0,0]
    shift_lda = -W_approximate_lda[-1,0,0]
    shift_gga = -W_approximate_gga[-1,0,0]
    
    # number of electronic states
    nstate = W_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of electron repulsion operator
    # (~ classical Coulomb energies of electronic states)
    axes[0].set_ylabel(r"electron repulsion $W_{II}$ (shifted)/ $E_h$")
    axes[0].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            bond_length_lda, W_exact[:,i,i] + shift_exact,
            lw=2, alpha=0.5,
            label=rf"$W_{{{i},{i}}}$")
        axes[0].plot(
            bond_length_lda, W_approximate_lda[:,i,i] + shift_lda,
            ls="--", color=line.get_color())
        axes[0].plot(
            bond_length_gga, W_approximate_gga[:,i,i] + shift_gga,
            ls="-.", color=line.get_color())

    axes[0].legend(title="$\mathbf{(a)}$ diagonal")

    # Off-diagonal elements of electron repulsion operator
    axes[1].set_ylabel(r"electron repulsion $W_{IJ}$ / $E_h$")
    axes[1].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                bond_length_lda, W_exact[:,i,j],
                lw=2, alpha=0.5,
                label=rf"$W_{{{i},{j}}}$")
            axes[1].plot(
                bond_length_lda, W_approximate_lda[:,i,j],
                ls="--", color=line.get_color())
            axes[1].plot(
                # The global phase of the wavefunctions is arbitary.
                # The phase of the off-diagonal matrix element is chosen so
                # that it agrees with the (random) phase of the exact results.
                bond_length_gga, (-1) * W_approximate_gga[:,i,j],
                ls="-.", color=line.get_color())
            
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(b)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
    dotdashed_line = matplotlib.lines.Line2D([], [], ls="-.", color="black")
    fig.legend(
        [solid_line, dashed_line, dotdashed_line],
        [
            r"$W^{\text{exact}}_{IJ}$", # = \langle \Psi_I \vert \sum_{a < b} 1/r_{ab} \vert \Psi_J \rangle$",
            r"$W^{\text{LDA}}_{IJ}$", # = \text{J}[\mathbf{D}]_{IJ} - \text{K}^{LDA}[\mathbf{D}]_{IJ} + \text{C}^{LDA}[\mathbf{D}]_{IJ}  + \text{SIC}~\delta_{IJ}$",
            r"$W^{\text{GGA}}_{IJ}$", # = \text{J}[\mathbf{D}]_{IJ} - \text{K}^{GGA}[\mathbf{D}]_{IJ} + \text{C}^{LDA}[\mathbf{D}]_{IJ}  + \text{SIC}~\delta_{IJ}$"
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=3
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    #plt.savefig("electron_repulsion_energies_exact_lda_gga.svg")
    #plt.savefig("electron_repulsion_energies_exact_lda_gga.png", dpi=300)

    plt.show()
