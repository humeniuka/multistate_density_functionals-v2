#!/usr/bin/env python
# coding: utf-8
"""
plot the exact matrix elements of the electron repulsion operator

  Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

and their multi-state local-density approximation

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
    with open('electron_repulsion_energies.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Bond lengths are in Angstrom.
    bond_length = numpy.array(scan_data['bond_length'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])
    W_exact = numpy.array(scan_data['W_exact'])
    W_approximate = numpy.array(scan_data['W_approximate'])
    SIC = numpy.array(scan_data['self_interaction_correction'])
    
    # number of electronic states
    nstate = W_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of electron repulsion operator
    # (~ classical Coulomb energies of electronic states)
    axes[0].set_ylabel(r"electron repulsion $W_{II}$ / $E_h$")
    axes[0].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            bond_length, W_exact[:,i,i],
            lw=2, alpha=0.5,
            label=rf"$W_{{{i},{i}}}$")
        axes[0].plot(
            bond_length, W_approximate[:,i,i],
            ls="--", color=line.get_color())

    axes[0].legend(title="$\mathbf{(a)}$ diagonal")

    # Off-diagonal elements of electron repulsion operator
    axes[1].set_ylabel(r"electron repulsion $W_{IJ}$ / $E_h$")
    axes[1].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                bond_length, W_exact[:,i,j],
                lw=2, alpha=0.5,
                label=rf"$W_{{{i},{j}}}$")
            axes[1].plot(
                bond_length, W_approximate[:,i,j],
                ls="--", color=line.get_color())

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(b)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")

    # Add SIC in formula only if it is non-zero.
    if abs(SIC).max() > 0.0:
        SIC_string = r"+ \text{SIC}~\delta_{IJ}"
    else:
        SIC_string = r""

    fig.legend(
        [solid_line, dashed_line],
        [
            r"$W_{IJ} = \langle \Psi_I \vert \sum_{a < b} 1/r_{ab} \vert \Psi_J \rangle$ (exact)",
            r"$W_{IJ} = \text{J}[\mathbf{D}]_{IJ} - \text{K}^{LDA}[\mathbf{D}]_{IJ} + \text{C}^{LDA}[\mathbf{D}]_{IJ} %s$" % SIC_string
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    #plt.savefig("electron_repulsion_energies.svg")
    #plt.savefig("electron_repulsion_energies.png", dpi=300)

    plt.show()
