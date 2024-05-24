#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy
import pyscf.dft
import scipy.linalg

from msdft.LinearAlgebra import eigensystem_derivatives
from msdft.LinearAlgebra import matrix_function_batch
from msdft.LinearAlgebra import matrix_function_derivatives_batch
from msdft.MultistateMatrixDensity import MultistateMatrixDensity


class KineticOperatorFunctional(ABC):
    def __init__(self, mol, level=8):
        """
        The abstract base class for kinetic operator functionals.

        :param mol: The molecule defines the integration grid.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type level: int
        """
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    @abstractmethod
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        pass

    def __call__(
            self,
            msmd : MultistateMatrixDensity,
            available_memory=1<<30):
        """
        compute the matrix of the kinetic energy operator in the subspace
        of electronic states by evaluating the kinetic energy functional T[D(r)]
        on the matrix density D(r):

          Tᵢⱼ = <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψⱼ> = T[D(r)]ᵢⱼ ,

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy functional should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param available_memory: The amount of memory (in bytes) that can be
           allocated for the kinetic energy density. If more memory is needed,
           the KED is evaluated in multiple chunks. (1<<30 corresponds to 1Gb)
           Since more memory is needed for intermediate quantities, this limit
           is only a rough estimate.
        :type available_memory: int

        :return kinetic_matrix: The kinetic energy matrix Tᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype kinetic_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        # number of grid points
        ncoord = self.grids.coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # matrix element of the kinetic energy operator <i|Top|j>
        kinetic_matrix = numpy.zeros((nstate,nstate))

        # If the resulting array that holds the kinetic energy density
        # exceeds `available_memory`, the KED is evaluated on smaller chunks
        # of the grid and summed into the kinetic matrix at the end.
        needed_memory = 50 * 2 * kinetic_matrix.itemsize * nstate**2 * ncoord
        number_of_chunks = max(1, (needed_memory + available_memory) // available_memory)
        # There cannot be more chunks than grid points.
        number_of_chunks = min(ncoord, number_of_chunks)

        # Loop over chunks of grid points and associated integration weights.
        for coords, weights in zip(
                numpy.array_split(self.grids.coords, number_of_chunks),
                numpy.array_split(self.grids.weights, number_of_chunks)):

            # Evaluate the kinetic energy density on the grid.
            KED = self.kinetic_energy_density(msmd, coords)

            # The matrix of the kinetic energy operator in the subspace is obtained
            # by integration T_{i,j}(r) over space and spin
            #
            #   Tᵢⱼ = ∫ KEDᵢⱼ(r) dr
            #
            kinetic_matrix += numpy.einsum('r,sijr->ij', weights, KED)

        return kinetic_matrix


class LSDAVonWeizsaecker1eFunctional(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    This functional should give the exact kinetic energy matrix for
    1-electron systems.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = -1/2 ϕᵢ*(r) ∇²ϕⱼ(r)

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r) and ∇tr(D)(r) = tr(∇D(r))
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)
        # `grad_trace_D` has shape (2,3,Ncoord) and is the gradient of `trace_D`.
        grad_trace_D = numpy.einsum('siiar->sar', grad_D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue
            #
            # R_{i,j}(r) = D_{i,j}(r) / tr(D(r))
            #
            R = D[s,...] / numpy.expand_dims(trace_D[s,...], axis=(0,1))
            #                  ∑ₖ∇D_{i,k}·∇D_{k,j}
            # C_{i,j}(r) = 1/2 -------------------
            #                        tr(D)
            #
            C_numerator = numpy.einsum('ikar,kjar->ijr', grad_D[s,...], grad_D[s,...])
            C_denominator = numpy.expand_dims(trace_D[s,...], axis=(0,1))
            C = 0.5 * C_numerator / C_denominator

            # For each grid point we have to solve the linear equation
            #  (1 - K)·T = C
            # where T_{i,j}(r) and C_{i,j}(r) are interpreted as vectors in ℂ^{Mstate x Mstate}
            dim2 = nstate*nstate
            # identity matrix
            delta = numpy.eye(nstate)
            # K_{i,j;m,n} =  - R_{i,j} delta_{m,n}  - delta_{i,m} R_{n,j}  - R_{i,m} delta_{n,j}
            K = numpy.zeros((dim2,dim2,ncoord))
            # ij is a multiindex that runs over all combinations of (i,j) (rows of K)
            ij = 0
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # mn is a multiindex that runs over all combinations of (m,n) (columns of K)
                    mn = 0
                    for m in range(0, nstate):
                        for n in range(0, nstate):
                            K[ij,mn,:] = (
                                -R[i,j,:]*delta[m,n]) -delta[i,m]*R[n,j,:] -R[i,m,:]*delta[n,j]
                            # increase column counter
                            mn += 1
                    # increase row counter
                    ij += 1

            # The kinetic energy density
            #
            #  KEDᵢⱼ(r) = 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
            #
            # is obtained by solving (1 - K).T = C for each grid point
            Id = numpy.eye(dim2)
            for r in range(0, ncoord):
                Cvec = C[:,:,r].flatten()
                Tvec = numpy.linalg.solve(Id - K[:,:,r], Cvec)
                # Reinterpret the vector Tvec as a square matrix.
                KED[s,:,:,r] = numpy.reshape(Tvec, (nstate,nstate))

        return KED


class LDAVonWeizsaecker1eFunctional(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    This functional should give the exact kinetic energy matrix for
    1-electron systems.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = -1/2 ϕᵢ*(r) ∇²ϕⱼ(r)

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (1,Mstate,Mstate,Ncoord)
           KED[0,i,j,r] is the kinetic energy density between the
           electronic states i and j at position coords[r,:].
           There is only one spin component, because the functional
           operates on the spin-traced matrix density.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((1,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Sum over spins to get total D = Dᵅ(r) + Dᵝ(r)
        total_density = D[0,...] + D[1,...]
        # and its gradient
        grad_total_density = grad_D[0,...] + grad_D[1,...]

        # Trace over electronic states to get tr(D)(r) and ∇tr(D)(r) = tr(∇D(r))
        trace_total_density = numpy.einsum('iir->r', total_density)
        grad_trace_total_density = numpy.einsum('iiar->ar', grad_total_density)

        #
        # R_{i,j}(r) = D_{i,j}(r) / tr(D(r))
        #
        R = total_density / numpy.expand_dims(trace_total_density, axis=(0,1))
        #                  ∑ₖ∇D_{i,k}·∇D_{k,j}
        # C_{i,j}(r) = 1/2 -------------------
        #                        tr(D)
        #
        C_numerator = numpy.einsum('ikar,kjar->ijr', grad_total_density, grad_total_density)
        C_denominator = numpy.expand_dims(trace_total_density, axis=(0,1))
        C = 0.5 * C_numerator / C_denominator

        # For each grid point we have to solve the linear equation
        #  (1 - K)·T = C
        # where T_{i,j}(r) and C_{i,j}(r) are interpreted as vectors in ℂ^{Mstate x Mstate}
        dim2 = nstate*nstate
        # identity matrix
        delta = numpy.eye(nstate)
        # K_{i,j;m,n} =  - R_{i,j} delta_{m,n}  - delta_{i,m} R_{n,j}  - R_{i,m} delta_{n,j}
        K = numpy.zeros((dim2,dim2,ncoord))
        # ij is a multiindex that runs over all combinations of (i,j) (rows of K)
        ij = 0
        for i in range(0, nstate):
            for j in range(0, nstate):
                # mn is a multiindex that runs over all combinations of (m,n) (columns of K)
                mn = 0
                for m in range(0, nstate):
                    for n in range(0, nstate):
                        K[ij,mn,:] = (
                            -R[i,j,:]*delta[m,n]) -delta[i,m]*R[n,j,:] -R[i,m,:]*delta[n,j]
                        # increase column counter
                        mn += 1
                # increase row counter
                ij += 1

        # The kinetic energy density
        #
        #  KEDᵢⱼ(r) = 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
        #
        # is obtained by solving (1 - K).T = C for each grid point
        Id = numpy.eye(dim2)
        for r in range(0, ncoord):
            Cvec = C[:,:,r].flatten()
            Tvec = numpy.linalg.solve(Id - K[:,:,r], Cvec)
            # Reinterpret the vector Tvec as a square matrix.
            KED[0,:,:,r] = numpy.reshape(Tvec, (nstate,nstate))

        return KED


class LSDAVonWeizsaeckerFunctional(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    The von-Weizsäcker functional for the electronic ground state

                        (∇ρ)²
           T[ρ] = ∫ 1/8 ----
                          ρ

    is turned into a matrix-density functional by replacing the density
    with the matrix density, ρ(r) -> D(r),

           T[D]ᵢⱼ = ∫ 1/8 ∑ₖ∑ₗ ∇Dᵢₖ D⁻¹ₖₗ ∇Dₗⱼ

    The matrix-inverse of D is placed symmetrically between the gradients.

    This functional does not give the exact kinetic energy matrix for
    1-electron systems but it performs much better on many-electron systems
    then :class:`~.LSDAVonWeizsaecker1eFunctional`.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r)
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue
            # (pseudo) inverse of matrix density, D⁻¹ₖₗ(r) at each grid point
            invD = numpy.zeros_like(D[s,...])
            for r in range(0, ncoord):
                invD[:,:,r] = scipy.linalg.pinv(D[s,:,:,r], rtol=1.0e-12)
            #
            # KED_{i,j}(r) = 1/8 ∑ₖ∑ₗ ∇D_{i,k} D⁻¹_{k,l} ·∇D_{l,j}
            #
            KED[s,...] = 1.0/8.0 * numpy.einsum(
                'ikar,klr,ljar->ijr',
                grad_D[s,...], invD, grad_D[s,...])

        return KED


class LDAVonWeizsaeckerFunctional(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    The von-Weizsäcker functional for the electronic ground state

                        (∇ρ)²
           T[ρ] = ∫ 1/8 ----
                          ρ

    is turned into a matrix-density functional by replacing the density
    with the matrix density, ρ(r) -> D(r),

           T[D]ᵢⱼ = ∫ 1/8 ∑ₖ∑ₗ ∇Dᵢₖ D⁻¹ₖₗ ∇Dₗⱼ

    The matrix-inverse of D is placed symmetrically between the gradients.

    This functional does not give the exact kinetic energy matrix for
    1-electron systems.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (1,Mstate,Mstate,Ncoord)
           KED[0,i,j,r] is the kinetic energy density
           between the electronic states i and j at position coords[r,:].
           There is only a single spin component because the functional
           operates on the total density D=Dᵅ+Dᵝ
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((1,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Sum over spins to get total D = Dᵅ(r) + Dᵝ(r)
        total_density = D[0,...] + D[1,...]
        # and its gradient
        grad_total_density = grad_D[0,...] + grad_D[1,...]

        # (pseudo) inverse of matrix density, D⁻¹ₖₗ(r) at each grid point
        invD = numpy.zeros_like(total_density)
        for r in range(0, ncoord):
            invD[:,:,r] = scipy.linalg.pinv(total_density[:,:,r], rtol=1.0e-12)
        #
        # KED_{i,j}(r) = 1/8 ∑ₖ∑ₗ ∇D_{i,k} D⁻¹_{k,l} ·∇D_{l,j}
        #
        KED[0,...] = 1.0/8.0 * numpy.einsum(
            'ikar,klr,ljar->ijr',
            grad_total_density, invD, grad_total_density)

        return KED


class LSDAVonWeizsaecker1eFunctionalII(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    This functional is exact for 1e systems but does not work well for multiple electrons.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r)
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue
            # (pseudo) inverse of matrix density, D⁻¹ₖₗ(r) at each grid point
            invD = numpy.zeros_like(D[s,...])
            for r in range(0, ncoord):
                invD[:,:,r] = scipy.linalg.pinv(D[s,:,:,r], rtol=1.0e-12)
            #
            # KED^{vW}_{i,j}(r) = 1/8 ∑ₖ∑ₗ ∇D_{i,k} D⁻¹_{k,l} ·∇D_{l,j}
            #
            KED_vW = 1.0/8.0 * numpy.einsum(
                'ikar,klr,ljar->ijr',
                grad_D[s,...], invD, grad_D[s,...])

            # The local identity I(r) = D(r) D⁻¹(r)
            # Since D(r) is not invertible, I is not the identity matrix.
            I = numpy.einsum('ikr,kjr->ijr', D[s,...], invD)

            # The local number of states
            #  N(r) = trace(I(r))
            # can be smaller than the number of electronic states `nstate`.
            N = numpy.einsum('iir->r', I)

            # The left hand side in the system of linear equations
            #  T⁰ = M.T
            # is the uncorrected von-Weizsaecker kinetic energy density.
            T0 = KED_vW

            # For each grid point we have to solve the linear equation
            #  T = M.T⁰   <=>  T = M⁻¹.T⁰
            # where T⁰_{i,j}(r) and T_{i,j}(r) are interpreted as vectors in ℝ^{Nstate x Nstate}
            dim2 = nstate*nstate
            # identity matrix
            delta = numpy.eye(nstate)
            # The matrix M relates T to T⁰ at each grid point
            # and has dimensions ℝ^{Nstate^2 x Nstate^2}.
            M = numpy.zeros((dim2,dim2,ncoord))

            # ij is a multiindex that runs over all combinations of (i,j) (rows of M)
            ij = 0
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # kl is a multiindex that runs over all combinations of (k,l) (columns of M)
                    kl = 0
                    for k in range(0, nstate):
                        for l in range(0, nstate):
                            #   M_{i,j;k,l}(r) =
                            #     1/4 * [ N(r) δₖᵢδₗⱼ + Iₖⱼ(r) δᵢₗ + Iₖᵢ(r) δₗⱼ + Dᵢⱼ(r) D⁻¹ₗₖ(r) ]
                            M[ij,kl,:] = 1.0/4.0 * (
                                N[:] * delta[k,i] * delta[l,j] +
                                I[k,j,:] * delta[i,l] +
                                I[k,i,:] * delta[l,j] +
                                D[s,i,j,:] * invD[l,k,:])
                            # increase column counter
                            kl += 1
                    # increase row counter
                    ij += 1

            # The kinetic energy density
            #
            #  KEDᵢⱼ(r) = 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
            #
            # is obtained by solving M(r).T(r) = T⁰(r) for each grid point r
            Id = numpy.eye(dim2)
            for r in range(0, ncoord):
                T0vec = T0[:,:,r].flatten()
                # The matrix might be singular, so we have to solve
                # the equation in a least-square sense.
                Tvec, _, _, _ = numpy.linalg.lstsq(M[:,:,r], T0vec, rcond=None)

                # Reinterpret the vector Tvec as a square matrix.
                KED[s,:,:,r] = numpy.reshape(Tvec, (nstate,nstate))

            # WARNING: For some reason the KED has to be multiplied by (N(r)+1)/2.
            #          I don't know yet how to derive this factor.
            KED[s,...] *= (N+1.0)/2.0

        return KED


class LSDAThomasFermiFunctional(KineticOperatorFunctional):
    """
    A Thomas-Fermi-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

      Tᵢⱼ = 3/10 (6π²)²ᐟ³ ∫ (D(r)⁵ᐟ³)ᵢⱼ dr

    Note that the power of 5/3 is not taken element-wise. D(r)⁵ᐟ³ is a matrix
    power that mixes the different elements of D(r).

    The kinetic energy is calculated separately for the spin-up and spin-down
    parts of the density matrix and summed:

      Tᵢⱼ[D(up)] + Tᵢⱼ[D(down)]

    Therefore the prefactor of the Thomas-Fermi energy contains (6π²) instead
    of (3π²), which is for the kinetic energy of the total density, Tᵢⱼ[D(up)+D(down)].
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density of the free electron gas

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r).
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue

            # The kinetic energy density
            #
            #  KEDᵢⱼ(r) = 3/10 (6π²)²ᐟ³ (D(r)⁵ᐟ³)ᵢⱼ
            #
            # For a closed-shell molecule, Dtot = 2*D(up), so that
            #
            #  t = 3/10 (3π²)²ᐟ³ (2 Dtot) = 2 3/10 (6π²)²ᐟ³ D(up)
            #    = 2 tₛₚᵢₙ
            prefactor = 3.0/10.0 * pow(6.0*numpy.pi**2, 2.0/3.0)
            for r in range(0, ncoord):
                # Compute eigenvalues Λ and eigenvectors U of the symmetric
                # matrix D.
                L, U = numpy.linalg.eigh(D[s,:,:,r])
                # Numerical rounding errors might produce tiny, negative eigenvalues instead of 0.
                assert numpy.all(L > -1.0e-12), "Eigenvalues of matrix density D are expected to be positive."

                # The fractional matrix power is obtained from the eigenvalue decomposition
                # as D⁵ᐟ³(r) = U(r) Λ⁵ᐟ³(r) Uᵀ(r)
                D_matrix_power = numpy.einsum('ia,a,ja->ij', U, pow(abs(L), 5.0/3.0), U)
                # Thomas-Fermi kinetic energy density
                ked_r = prefactor * D_matrix_power
                # Check that the kinetic energy density is real.
                assert numpy.sum(abs(ked_r.imag)) < 1.0e-10

                KED[s,:,:,r] = ked_r.real

        return KED


class LDAThomasFermiFunctional(KineticOperatorFunctional):
    """
    A Thomas-Fermi-like functional that maps the matrix density D(r)=D(up)+D(down)
    to the matrix of the kinetic energy in the subspace.

      Tᵢⱼ = 3/10 (3π²)²ᐟ³ ∫ (D(r)⁵ᐟ³)ᵢⱼ dr

    Note that the power of 5/3 is not taken element-wise. D(r)⁵ᐟ³ is a matrix
    power that mixes the different elements of D(r).
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density of the free electron gas

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (1,Mstate,Mstate,Ncoord)
           KED[1,i,j,r] is the kinetic energy density between the electronic states i and j
           at position coords[r,:]. There is only one spin component because the functional
           operates on the total (spin-traced) density.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((1,nstate,nstate,ncoord))

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(coords)

        # Sum over spins to get total D = Dᵅ(r) + Dᵝ(r)
        total_density = D[0,...] + D[1,...]

        # The kinetic energy density
        #
        #  KEDᵢⱼ(r) = 3/10 (3π²)²ᐟ³ (D(r)⁵ᐟ³)ᵢⱼ
        prefactor = 3.0/10.0 * pow(3.0*numpy.pi**2, 2.0/3.0)
        for r in range(0, ncoord):
            # Compute eigenvalues Λ and eigenvectors U of the symmetric
            # matrix D.
            L, U = numpy.linalg.eigh(total_density[:,:,r])
            # Numerical rounding errors might produce tiny, negative eigenvalues instead of 0.
            assert numpy.all(L > -1.0e-12), "Eigenvalues of matrix density D are expected to be positive."

            # The fractional matrix power is obtained from the eigenvalue decomposition
            # as D⁵ᐟ³(r) = U(r) Λ⁵ᐟ³(r) Uᵀ(r)
            D_matrix_power = numpy.einsum('ia,a,ja->ij', U, pow(abs(L), 5.0/3.0), U)
            # Thomas-Fermi kinetic energy density
            ked_r = prefactor * D_matrix_power
            # Check that the kinetic energy density is real.
            assert numpy.sum(abs(ked_r.imag)) < 1.0e-10

            KED[0,:,:,r] = ked_r.real

        return KED


class EigendecompositionKineticFunctional(KineticOperatorFunctional):
    """
    This kinetic energy functional is based on an eigenvalue decomposition
    of the matrix density D(r).

    Let λₐ(r) and Uᵢₐ(r) be the eigenvalues and eigenvectors of the matrix
    density D(r) at each position,

      ∑ⱼ Dᵢⱼ(r) Uⱼₐ(r) = λₐ(r) Uᵢₐ(r),

    then the kinetic energy matrix is approximated as

      Tᵢⱼ = ∫ 1/2 ∑ₐ ∇(λₐ¹ᐟ² Uᵢₐ)·∇(λₐ¹ᐟ² Uⱼₐ)

    """
    @staticmethod
    def eigen_decomposition(
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray,
            epsilon=1.0e-12):
        """
        Compute the eigenvalues Λ(r) and eigenvectors U(r) of the
        multistate matrix density D(r) and their derivatives ∇Λ(r)
        and ΛU(r). The orthogonal matrix U(r) diagonalizes D(r) and
        depends on the grid point r.

           U(r)ᵀ.D(r).U(r) = Λ(r)

        For each spin component, the matrix density is diagonalized separately.

        :param msmd: The multistate matrix density in the electronic subspace
           which the eigenvalues, eigenvectors and their derivatives should be
           determined for.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param epsilon: Threshold for treating eigenvalues as zero.
        :type epsilon: float

        Mstate is the number of electronic states
        Ncoord is the number of grid points.

        :return:
            L, U, grad_L, grad_U
        :rtype: tuple of numpy.ndarray
           `L` has shape (2,Mstate,Ncoord), L[s,j,:] is the j-th eigenvalue of D(r) for spin s.
           `U` has shape (2,Mstate,Mstate,Ncoord), U[s,i,j,:] is the i-th component of the j-th
            eigenvector of D(r) for spin s.
           `grad_L` has shape (2,Mstate,3,Ncoord), grad_L[s,j,xyz,:] is the derivative of the
            j-th eigenvalue of D for spin s, dΛ_j(r)/dq (q=0(x), 1(y), 2(z))
           `grad_U` has shape (2,Mstate,Mstate,3,Ncoord), grad_U[s,i,j,xyz,:] is the derivative
            of the i-th component of the j-th eigenvector of D for spin s,
            dU_ij(r)/dq (q=0(x), 1(y), 2(z)).
        """
        # Evaluate D(r) and its 1st and 2nd order derivatives on the grid.
        D_derivs = msmd.evaluate_derivatives(coords, deriv=2)

        # The shape of D_derivs is (2,Mstate,Mstate,3,deriv+1,Ncoord).
        # D(x,y,z)
        D = D_derivs[:,:,:,0,0,:]
        # gradient ∇D(r) = [∂/∂x D, ∂/∂y D, ∂/∂z D]
        D_deriv1 = D_derivs[:,:,:,:,1,:]
        # second derivatives [∂²/∂x², ∂²/∂y² D, ∂²/∂z²]
        D_deriv2 = D_derivs[:,:,:,:,2,:]

        # number of spins, number of states, ,..., number of grid points
        nspin, nstate, _, _, _, ncoord = D_derivs.shape

        # Λ(r), reserve space for eigenvalues of D
        L = numpy.zeros((nspin, nstate, ncoord))
        # U(r), reserve space for eigenvectors of D(r)
        U = numpy.zeros((nspin, nstate, nstate, ncoord))

        # ∇Λ(r), reserve space for gradients of eigenvalues of D(r)
        grad_L = numpy.zeros((nspin, nstate, 3, ncoord))
        # ∇U(r), reserver space for gradients of eigenvectors of D(r)
        grad_U = numpy.zeros((nspin, nstate, nstate, 3, ncoord))

        # Diagonalize D(r) and compute gradients for each spin and grid point separately.
        for spin in range(0, nspin):
            for r in range(0, ncoord):
                L[spin,:,r], U[spin,:,:,r], grad_L[spin,:,:,r], grad_U[spin,:,:,:,r] = \
                    eigensystem_derivatives(
                        # D
                        D[spin,:,:,r],
                        # D'
                        D_deriv1[spin,:,:,:,r],
                        # D'' is needed if there are repeated eigenvalues.
                        D_deriv2[spin,:,:,:,r],
                        # Eigenvalues |λₛ-λₜ| <= `epsilon` are treated as identical.
                        epsilon=epsilon)

        return L, U, grad_L, grad_U

    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray,
            epsilon=1.0e-12):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param epsilon: Threshold for neglecting singular eigenvalues.
           Eigenvalues |λₐ| <= epsilon are treated as zero.
        :type epsilon: float

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # Diagonalize D(r) at each grid point to find its eigenvalues Λ(r)
        # and eigenvectors U(r) as well as their gradients, ∇Λ(r) and ∇U(r).
        L, U, grad_L, grad_U = EigendecompositionKineticFunctional.eigen_decomposition(msmd, coords, epsilon=epsilon)

        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # reserve space for kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # a enumerates non-zero eigenvalues
        #
        # KEDᵢⱼ(r) = ∑ₐ { 1/8 (∇λₐ·∇λₐ)/λₐ Uᵢₐ Uⱼₐ + 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
        #                 + 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ) + 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ) }

        # compute (∇λ·∇λ)
        grad_L_product = numpy.einsum('sadr,sadr->sar', grad_L, grad_L)
        # compute (∇λ¹ᐟ²·∇λ¹ᐟ²) = 1/4 (∇λ·∇λ)/λ
        grad_sqrtL_product = numpy.zeros_like(L)
        # Avoid dividing by zero for λ=0
        # Non-zero eigenvalues, for which division is not problematic.
        good = abs(L) > epsilon
        grad_sqrtL_product[good] = 1.0/4.0 * grad_L_product[good] / L[good]

        # ∑ₐ 1/2 (∇λₐ¹ᐟ²·∇λₐ¹ᐟ²) Uᵢₐ Uⱼₐ
        KED += 1.0/2.0 * numpy.einsum('sar,siar,sjar->sijr', grad_sqrtL_product, U, U)
        # ∑ₐ 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
        KED += 1.0/2.0 * numpy.einsum('sar,siadr,sjadr->sijr', L, grad_U, grad_U)
        # ∑ₐ 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ)
        KED += 1.0/4.0 * numpy.einsum('siar,sadr,sjadr->sijr', U, grad_L, grad_U)
        # ∑ₐ 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ)
        KED += 1.0/4.0 * numpy.einsum('sjar,sadr,siadr->sijr', U, grad_L, grad_U)

        return KED

class EigendecompositionKineticFunctionalvW(KineticOperatorFunctional):
    """
    This kinetic energy functional is based on an eigenvalue decomposition
    of the matrix density D(r).

    Let λₐ(r) and Uᵢₐ(r) be the eigenvalues and eigenvectors of the matrix
    density D(r) at each position,

      ∑ⱼ Dᵢⱼ(r) Uⱼₐ(r) = λₐ(r) Uᵢₐ(r),

    then the kinetic energy matrix is approximated as

      T[D]ᵢⱼ = ∑ₐ ∫ 1/8 ∑ₖ∑ₗ ∇Dᵃᵢₖ Dᵃ⁻¹ₖₗ ∇Dᵃₗⱼ

    where Dᵃᵢⱼ = λₐ(r) Uᵢₐ(r) Uⱼₐ(r) is the part of the density matrix
    belonging to eigenvalue a. Effectively density matrix is projected
    onto its eigenvectors and the von Weizsaecker functional is applied
    to each part.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray,
            epsilon=1.0e-12):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param epsilon: Threshold for neglecting singular eigenvalues.
           Eigenvalues |λₐ| <= epsilon are treated as zero.
        :type epsilon: float

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # Diagonalize D(r) at each grid point to find its eigenvalues Λ(r)
        # and eigenvectors U(r) as well as their gradients, ∇Λ(r) and ∇U(r).
        L, U, grad_L, grad_U = EigendecompositionKineticFunctional.eigen_decomposition(msmd, coords, epsilon=epsilon)

        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # reserve space for kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Eigendecomposition of density matrix
        # Dᵃᵢⱼ = λₐ(r) Uᵢₐ(r) Uⱼₐ(r)
        D_eigen = numpy.einsum('sar,siar,sjar->asijr', L, U, U)
        # its gradient
        # ∇Dᵃᵢⱼ = ∇λₐ(r) Uᵢₐ(r) Uⱼₐ(r) + λₐ(r) ∇Uᵢₐ(r) Uⱼₐ(r) + λₐ(r) Uᵢₐ(r) ∇Uⱼₐ(r)
        grad_D_eigen = (
            numpy.einsum('sadr,siar,sjar->asijdr', grad_L, U, U) +
            numpy.einsum('sar,siadr,sjar->asijdr', L, grad_U, U) +
            numpy.einsum('sar,siar,sjadr->asijdr', L, U, grad_U)
        )

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Trace over electronic states to get tr(Dᵃ)(r)
        # `trace_D` has shape (nstate,2,Ncoord,), trace_D[eigval,spin,:] = sum_i D[eigval,spin,i,i,:]
        trace_D_eigen = numpy.einsum('asiir->asr', D_eigen)

        # Loop over eigenvalues
        for a in range(0, nstate):
            # Loop over spins. The kinetic energy is computed separately for each spin
            # projection and added.
            for s in range(0, nspin):
                if numpy.all(trace_D_eigen[a,s,...] == 0.0):
                    # There are no electrons with spin projection s
                    # that could contribute to the kinetic energy.
                    continue
                # (pseudo) inverse of matrix density, Dᵃ⁻¹ₖₗ(r) at each grid point
                invD_eigen = numpy.zeros_like(D_eigen[a,s,...])
                for r in range(0, ncoord):
                    invD_eigen[:,:,r] = scipy.linalg.pinv(D_eigen[a,s,:,:,r], rtol=1.0e-12)
                #
                # KED_{i,j}(r) += 1/8 ∑ₖ∑ₗ ∇Dᵃ_{i,k} Dᵃ⁻¹_{k,l} ·∇Dᵃ_{l,j}
                #
                KED[s,...] += 1.0/8.0 * numpy.einsum(
                    'ikdr,klr,ljdr->ijr',
                    grad_D_eigen[a,s,...], invD_eigen, grad_D_eigen[a,s,...])

        return KED


class EigendecompositionKineticFunctionalII(KineticOperatorFunctional):
    """
    This kinetic energy functional is based on an eigenvalue decomposition
    of the matrix density D(r).

    Let λₐ(r) and Uᵢₐ(r) be the eigenvalues and eigenvectors of the matrix
    density D(r) at each position,

      ∑ⱼ Dᵢⱼ(r) Uⱼₐ(r) = λₐ(r) Uᵢₐ(r),

    then the kinetic energy matrix is approximated as

      Tᵢⱼ = ∫ 1/2 ∑ₐ ∇φᵢₐ·∇φⱼₐ + ∫ 1/2 ∑ᵤ∑ᵥ∑ₖ (∇φᵢᵤ φⱼᵥ - φᵢᵤ ∇φⱼᵥ) Uₖᵤ∇Uₖᵥ

    with

       φᵢₐ = λₐ¹ᐟ² Uᵢₐ

    For a one-electron system the second term cancels and the functional gives
    the exact kinetic energy.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray,
            epsilon=1.0e-12):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param epsilon: Threshold for neglecting singular eigenvalues.
           Eigenvalues |λₐ| <= epsilon are treated as zero.
        :type epsilon: float

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # Diagonalize D(r) at each grid point to find its eigenvalues Λ(r)
        # and eigenvectors U(r) as well as their gradients, ∇Λ(r) and ∇U(r).
        L, U, grad_L, grad_U = EigendecompositionKineticFunctional.eigen_decomposition(msmd, coords, epsilon=epsilon)

        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # Numerical rounding errors might produce tiny, negative eigenvalues instead of 0.
        assert numpy.all(L > -epsilon), "Eigenvalues of matrix density D are expected to be positive."
        # Square root of eigenvalues, Λ¹ᐟ²
        L_square_root = numpy.sqrt(abs(L))

        # Allocate memory for ∇Λ¹ᐟ² = 1/2 Λ⁻¹ᐟ² ∇Λ.
        grad_L_square_root = numpy.zeros_like(grad_L)
        # To avoid dividing by zero for λ=0, the gradient is only computed for
        # non-zero eigenvalues, for which division is not problematic.
        good = abs(L) > epsilon
        # Loop over components of gradient d/dx, d/dy, d/dz
        for xyz in [0,1,2]:
            # dΛ/dx (xyz=0), dΛ/dy (xyz=1) or dΛ/dz (xyz=2)
            dL = grad_L[:,:,xyz,:]
            # dΛ¹ᐟ²/dx, dΛ¹ᐟ²/dy or dΛ¹ᐟ²/dz
            dL_square_root = numpy.zeros_like(L)
            # dΛ¹ᐟ²/dx = 1/2 Λ⁻¹ᐟ² dΛ/dx etc.
            dL_square_root[good] = 1.0/2.0 * 1.0/L_square_root[good] * dL[good]
            # Copy x,y or z component into gradient vector ∇Λ¹ᐟ².
            grad_L_square_root[:,:,xyz,:] = dL_square_root

        # "Wavefunctions" φᵢₐ = λₐ¹ᐟ² Uᵢₐ
        wavefunctions = numpy.einsum('sar,siar->siar', L_square_root, U)
        # and their gradients ∇φᵢₐ = ∇λₐ¹ᐟ² Uᵢₐ + λₐ¹ᐟ² ∇Uᵢₐ
        grad_wavefunctions = (
            # ∇λₐ¹ᐟ² Uᵢₐ
            numpy.einsum('sadr,siar->siadr', grad_L_square_root, U) +
            # λₐ¹ᐟ² ∇Uᵢₐ
            numpy.einsum('sar,siadr->siadr', L_square_root, grad_U))

        # Any scale factor can be used, since the additional term
        #  scale * ....
        # disappear for one-electron systems.
        scale = 1.0

        #
        # KEDᵢⱼ(r) =
        #    1/2 ∑ₐ ∇φᵢₐ·∇φⱼₐ + 1/2 ∑ᵤ∑ᵥ∑ₖ (∇φᵢᵤ φⱼᵥ - φᵢᵤ ∇φⱼᵥ) Uₖᵤ ∇Uₖᵥ
        KED = (
            # 1/2 ∑ₐ ∇φᵢₐ·∇φⱼₐ
            0.5 * numpy.einsum('siadr,sjadr->sijr', grad_wavefunctions, grad_wavefunctions) +
            # 1/2 ∑ᵤ∑ᵥ∑ₖ ∇φᵢᵤ φⱼᵥ Uₖᵤ ∇Uₖᵥ
            scale * 0.5 * numpy.einsum('siudr,sjvr,skur,skvdr->sijr',
                grad_wavefunctions, wavefunctions, U, grad_U) -
            # - 1/2 ∑ᵤ∑ᵥ∑ₖ φᵢᵤ ∇φⱼᵥ) Uₖᵤ ∇Uₖᵥ
            scale * 0.5 * numpy.einsum('siur,sjvdr,skur,skvdr->sijr',
                wavefunctions, grad_wavefunctions, U, grad_U)
            # If the following term is added, the functional KED becomes
            # identical to KED = 1/2 ∇D¹ᐟ² ∇D¹ᐟ²
            #
            #+ 0.5 * numpy.einsum('uv,siur,sjvr,skudr,skvdr->sijr',
            #    offdiagonal, wavefunctions, wavefunctions, grad_U, grad_U)
            #
        )

        return KED


class MatrixSquareRootKineticFunctional(KineticOperatorFunctional):
    """
    This kinetic energy functional uses the gradient of the square root D¹ᐟ²(r)
    of the matrix density D. The kinetic energy matrix is approximated as

      T[D]ᵢⱼ = ∫ 1/2 ∑ₖ ∇(D¹ᐟ²)ᵢₖ ∇(D¹ᐟ²)ₖⱼ

    For a single state, the functional reduces to the von Weizsaecker functional.

    The calculation of the gradient of D¹ᐟ² is not trivial, since there is no
    chain rule of differentiation for analytic matrix functions.

    Note that, since ∇D and D do not commute, 1/2 ∇D¹ᐟ² ∇D¹ᐟ² and 1/8 ∇D D ∇D
    are not the same, unlike for the single state von Weizsaecker functional.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray,
            epsilon=1.0e-12):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param epsilon: Threshold for neglecting singular eigenvalues.
           Eigenvalues |λₐ| <= epsilon are treated as zero.
        :type epsilon: float

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        def sqrt_deriv1(density):
            """
            Compute derivative of f(ρ) = ρ¹ᐟ², which is f'(ρ) = 1/2 ρ⁻¹ᐟ².
            At grid points where ρ=0, f'(0) is arbitrarily set to 0. This
            choice should not affect the kinetic energy density since at
            the grid points where D=0, ∇D=0, too, so that ∇D¹ᐟ²=0 in any case.
            """
            # Avoid dividing by zero for ρ=0.
            # Non-zero eigenvalues, for which division is not problematic.
            good = abs(density) > epsilon
            # At ρ=0, ρ¹ᐟ² is not differentiable, at these points we set f'(0) = 0.
            deriv1 = numpy.zeros_like(density)
            # Compute d/dρ ρ¹ᐟ² = 1/2 ρ⁻¹ᐟ²  for grid point where ρ > 0.
            deriv1[good] = 0.5 * 1.0/numpy.sqrt(abs(density[good]))
            return deriv1

        # D¹ᐟ² and its gradient ∇D¹ᐟ² are computed from the eigen decomposition of D.
        D_square_root, grad_D_square_root = matrix_function_derivatives_batch(
            # f(ρ) = ρ¹ᐟ²
            # The eigenvalues of the matrix density should all be positive.
            lambda density: numpy.sqrt(abs(density)),
            # f'(ρ) = 1/2 ρ⁻¹ᐟ²
            sqrt_deriv1,
            # matrix density Dᵢⱼ
            D,
            # derivatives of matrix density ∇Dᵢⱼ
            grad_D,
            # threshold for neglecting singular eigenvalues
            epsilon=epsilon
        )

        # kinetic energy density
        # KEDᵢⱼ(r) = 1/2 ∑ₖ ∇(D¹ᐟ²)ᵢₖ ∇(D¹ᐟ²)ₖⱼ
        KED = 0.5 * numpy.einsum('sikdr,skjdr->sijr', grad_D_square_root, grad_D_square_root)

        return KED


class GGALeeLeeParr91KineticFunctional(KineticOperatorFunctional):
    # The Thomas-Fermi coefficient.
    C_F = 3.0/10.0 * pow(3.0*numpy.pi**2, 2.0/3.0)
    # The empirical value of α (taken from caption of table I in [LLP91])
    # apparently was determined from a least square fit to the kinetic
    # energies of rare gas atoms.
    alpha = 0.0044188
    # gamma for Eqn.(6) of [LLP91]. Note that there is a typo in the paper. There should be
    # a factor of x in front of gamma.
    gamma = 0.0253

    def __init__(self, mol, level=8):
        """
        Multi-state kinetic energy of of Lee, Lee and Parr [LLP91], which is based on the assumption
        that the same functional form can be used for the kinetic energy as for the exchange energy.
        This conjoint hypothesis is not true but gives reasonable kinetic energies (see Table I in
        [Thakkar1992])

        The LLP functional is given by

            T[Dᵅ(r), Dᵝ(r)] = T[Dᵅ(r)] + T[Dᵝ(r)]

        with

            T[Dᵅ(r)] = 2²ᐟ³ C_F ∫ ½ [ Dᵅ(r)⁵ᐟ³ G(X²(r)ᵅ) + G(X²(r)ᵅ) Dᵅ(r)⁵ᐟ³] dr

            (similarly for T[Dᵝ(r)])

        and the enhancement factor over the LDA kinetic energy,

            G(X²(r)) = 1 + α X²(r) / (1 + γ X(r) sinh⁻¹(X(r))).

        - D(r)⁵ᐟ³ is a fractional matrix-power of D(r), which is calculated by diagonalizing D.
        - G(X²(r)) is the enhancement factor over LDA. It is a matrix function of the square of
          the dimensionless (reduced) gradient, X²(r) = (36π)²ᐟ³ ∇R(r)·∇R(r), which depends on
          the gradient of the Wigner-Seitz radius R(r) = (4π/3 D(r))⁻¹ᐟ³.
          X(r) is the matrix square root of X²(r).
          X²(r) and R(r) are position-dependent matrices with the same dimensions
          as the matrix density D(r). G(X²) is calculated by diagonalizing X² and applying
          the scalar function G(·) to its eigenvalues.

        The kinetic-energy matrix has to be symmetric/hermitian. Since the product of two
        matrices is not symmetric (unless the two matrices commute), the product of two matrix
        functions A and B that depend on D and ∇D, A[D].B[∇D], will not be symmetric,
        because [D,∇D]≠0. The simplest way to symmetrize the expression is to replace
        A.B with 1/2 (A.B+B.A). Therefore the GGA kinetic-energy density

            ρᵅ(r)⁵ᐟ³ G(x²(r)ᵅ)

        is replaced by

            ½ [ Dᵅ(r)⁵ᐟ³ G(X²(r)ᵅ) + G(X²(r)ᵅ) Dᵅ(r)⁵ᐟ³]

        in the matrix functional.

        References
        ----------
        [LLP91] Lee, Lee, Parr (1991), Phys. Rev. A 44, 768
            "Conjoint gradient correction to the Hartree-Fock
            kinetic- and exchange-energy density functionals".
        [Thakkar1992] A. Thakkar, Phys. Rev. A 46, 6920
            "Comparison of kinetic-energy density functionals"

        :param mol: The molecule defines the integration grid.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type level: int
        """
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def enhancement_factor(self, x2):
        """ The scalar function g(x²) for the enhancement factor """
        x = numpy.sqrt(x2)
        f = 1.0 + self.alpha * (
            x2 / (1 + self.gamma * x * numpy.arcsinh(x))
        )
        return f

    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray,
            epsilon = 1.0e-12):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

                    ≈ 2²ᐟ³ C_F ½ [Dᵅ(r)⁵ᐟ³ G(X²(r)ᵅ) + G(X²(r)ᵅ) Dᵅ(r)⁵ᐟ³]ᵢⱼ

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param epsilon: Threshold for neglecting singular eigenvalues.
           Eigenvalues |λₐ| <= epsilon are treated as zero.
        :type epsilon: float

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # The fractional matrix power is obtained from the eigenvalue decomposition.
        # as D⁵ᐟ³(r) = U(r) Λ⁵ᐟ³(r) Uᵀ(r)
        D_matrix_power = matrix_function_batch(lambda L: pow(abs(L), 5.0/3.0), D)

        # The matrix version of the Wigner-Seitz radius is also calculated from the eigenvalue
        # decomposition as R(r) = U(r) (4π/3 Λ(r))⁻¹ᐟ³ Uᵀ(r).
        def wigner_seitz_radius(density):
            # Avoid dividing by zero for ρ=0.
            # Non-zero eigenvalues, for which division is not problematic.
            good = abs(density) > epsilon
            # When ρ=0, the electron radius should be r=inf. However, since the
            # kinetic-energy is 0 if there are no electrons, any value can be chosen
            # for r(ρ=0). Here we set r(ρ=0) to 0.
            radius = numpy.zeros_like(density)
            # Compute Wigner-Seitz radius for grid point where ρ > 0.
            radius[good] = pow((4.0*numpy.pi/3.0) * abs(density[good]), -1.0/3.0)
            return radius

        def wigner_seitz_radius_deriv1(density):
            # Derivative of the Wigner-Seitz radius w/r/t the density
            #   ∇rₐ(r) = (-1/3) (4π/3)⁻¹ᐟ³  (ρ(r))⁻⁴ᐟ³ ∇ρ(r)
            #          = (-1/3) (4π/3) rₐ⁴ ∇ρ(r)
            #          = rₐ'(ρ) ∇ρ(r)
            radius = wigner_seitz_radius(density)
            # rₐ'(ρ) = (-1/3) (4π/3) rₐ⁴
            radius_deriv1 = (-1.0/3.0) * (4.0*numpy.pi/3.0) * pow(radius, 4.0)
            return radius_deriv1

        # Wigner-Seitz radius matrix, Rᵢⱼ, and its gradient, ∇Rᵢⱼ.
        R, grad_R = matrix_function_derivatives_batch(
            # f(ρ)
            wigner_seitz_radius,
            # f'(ρ)
            wigner_seitz_radius_deriv1,
            # matrix density Dᵢⱼ
            D,
            # derivatives of matrix density ∇Dᵢⱼ
            grad_D,
            # threshold for neglecting singular eigenvalues
            epsilon=epsilon
        )

        # Compute X²(r) = (36π)²ᐟ³ ∇R(r)·∇R(r)
        #   X²ᵢⱼ = ∑ₐ ∇Rᵢₐ·∇Rₐⱼ
        X2 = pow(36.0 * numpy.pi, 2.0/3.0) * numpy.einsum(
            'siadr,sajdr->sijr', grad_R, grad_R)

        # The enhancement factor G(X²) is a matrix function. It is calculated
        # via the eigendecomposition of the matrix X² = V x² Vᵀ, where x² and V
        # are the eigenvalues and eigenvectors of X², respectively. Then the
        # enhancement factor over LDA is calculated as
        #   G(X²) = V g(x²) Vᵀ
        G_enhancement_factor = matrix_function_batch(
          lambda x2: self.enhancement_factor(abs(x2)), X2)

        # Symmetrized GGA kinetic-energy density,
        #   KED[D]ᵢⱼ(r) = 2²ᐟ³ C_F ½ [Dᵅ(r)⁵ᐟ³ G(X²(r)ᵅ) + G(X²(r)ᵅ) Dᵅ(r)⁵ᐟ³]ᵢⱼ
        KED = pow(2.0, 2.0/3.0) * self.C_F * 0.5 * (
            # D(r)⁵ᐟ³ G(X²(r))
            numpy.einsum(
                'siar,sajr->sijr',
                D_matrix_power, G_enhancement_factor) +
            # G(X²(r)) D(r)⁵ᐟ³
            numpy.einsum(
                'siar,sajr->sijr',
                G_enhancement_factor, D_matrix_power)
        )
        # Check that the kinetic energy density is real.
        assert numpy.max(abs(KED).imag) < 1.0e-10

        return KED
