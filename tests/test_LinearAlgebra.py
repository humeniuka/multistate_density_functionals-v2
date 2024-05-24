#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy
import numpy.testing
import scipy.linalg

from msdft.LinearAlgebra import eigensystem_derivatives
from msdft.LinearAlgebra import matrix_function
from msdft.LinearAlgebra import matrix_function_batch
from msdft.LinearAlgebra import matrix_function_derivatives
from msdft.LinearAlgebra import matrix_function_derivatives_batch
from msdft.LinearAlgebra import LinearAlgebraException


def kinetic_energy_density(L, U, grad_L, grad_U):
    """
    The kinetic energy density matrix

    KEDᵢⱼ(r) = ∑ₐ { 1/8 (∇λₐ·∇λₐ)/λₐ Uᵢₐ Uⱼₐ + 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
                      + 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ) + 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ) }

    is invariant under reordering of the eigenvalues.

    :param L: eigenvalues λₐ
    :type L: numpy.ndarray of shape (n,)

    :param U: eigenvectors Uᵢₐ
    :type U: numpy.ndarray of shape (n,n)

    :param grad_L: gradients of eigenvalues ∇λₐ
    :type grad_L: numpy.ndarray of shape (n,d)
        where d is the number of partial derivatives
        in the gradient vector.

    :param grad_U: gradients of eigenvectors ∇Uᵢₐ
    :type grad_U: numpy.ndarray of shape (n,n,d)
        where d is the number of partial derivatives
        in the gradient vector.

    :return: kinetic energy density KEDᵢⱼ
    :rtype: numpy.ndarray of shape (n,n)
    """
    # reserve space for kinetic energy density KEDᵢⱼ(r)
    KED = numpy.zeros_like(U)

    # a enumerates non-zero eigenvalues
    #
    # KEDᵢⱼ(r) = ∑ₐ { 1/8 (∇λₐ·∇λₐ)/λₐ Uᵢₐ Uⱼₐ + 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
    #                 + 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ) + 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ) }

    # compute (∇λ·∇λ)
    grad_L_product = numpy.einsum('ad,ad->a', grad_L, grad_L)
    # compute (∇λ¹ᐟ²·∇λ¹ᐟ²) = 1/4 (∇λ·∇λ)/λ
    grad_sqrtL_product = numpy.zeros_like(L)
    # Avoid dividing by zero for λ=0
    # Non-zero eigenvalues, for which division is not problematic.
    good = abs(L) > 0.0
    grad_sqrtL_product[good] = 1.0/4.0 * grad_L_product[good] / L[good]

    # ∑ₐ 1/2 (∇λₐ¹ᐟ²·∇λₐ¹ᐟ²) Uᵢₐ Uⱼₐ
    KED += 1.0/2.0 * numpy.einsum('a,ia,ja->ij', grad_sqrtL_product, U, U)
    # ∑ₐ 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
    KED += 1.0/2.0 * numpy.einsum('a,iad,jad->ij', L, grad_U, grad_U)
    # ∑ₐ 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ)
    KED += 1.0/4.0 * numpy.einsum('ia,ad,jad->ij', U, grad_L, grad_U)
    # ∑ₐ 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ)
    KED += 1.0/4.0 * numpy.einsum('ja,ad,iad->ij', U, grad_L, grad_U)

    return KED


class TestLinearAlgebra(unittest.TestCase):
    def _random_continuous_eigensystem(
            self,
            dim=3,
            repeated_eigenvalues=False,
            repeated_eigenvalue_derivatives=False,
            t0=0.234):
        """
        This fixture creates functions needed by multiple tests.

        In order to verify the code for derivatives of eigenvectors, we have to
        construct a one-parameter family of symmetric, differentiable matrices S(t),
        for which the eigenvalue derivatives and eigenvector derivatives are known.
        We start with differentiable functions for the orthogonal transformation
        U(t) = exp(f(t) X) and eigenvalues λ1(t), λ2(t), ..., and build the symmetric
        matrix as
          S(t) = U(t).diag(λ1(t), λ2(t), ...).Uᵀ(t)

        :param dim: dimension of test matrix
        :type dim: int

        :param repeated_eigenvalues:
            True: The matrix has repeated eigenvalues, but the
            eigenvalue derivatives of the repeated eigenvalues are distinct.
            False: All eigenvalues are distinct.
        :type repeated_eigenvalues: bool

        :param repeated_eigenvalue_derivatives:
            True: The matrix has repeated eigenvalues with repeated eigenvalue
            derivatives.
            False: The derivatives of repeated eigenvalues are dsitinct.
        :type repeated_eigenvalue_derivatives: bool

        :param t0: The parameter at which the derivative S(t)|_{t=t0}
            is taken.
        :type t0: float

        :return:
            symmetric_matrix(t,deriv=0) computes S(t), S'(t) and S''(t) for deriv=0,1,2
            eigenvectors(t,deriv=0) computes the eigenvalues of S, λ(t), λ'(t) and λ''(t)
            eigenvalues(t,deriv=0) computes the eigenvectors of S, U(t), U'(t) and U''(t)
        :rtype: tuple of callable
        """
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=6789)

        # Create a random antisymmetric matrix Xᵀ = -X
        X = random_number_generator.random((dim, dim))
        X = 0.5 * (X - X.T)

        # function f(t) and its derivatives
        def f(t, deriv=0):
            phase = pow(-1, deriv//2)
            if deriv % 2 == 0:
                # f(t), f''(t), .., f^{(2n)}(t)
                return phase * numpy.sin(t)
            else:
                # f'(t), f'''(t), ..., f^{(2n+1)}(t)
                return phase * numpy.cos(t)

        # One-parameter family of orthogonal matrices U(t) = exp(f(t) X)
        # and its derivatives.
        def eigenvectors(t, deriv=0):
            U = scipy.linalg.expm(f(t, deriv=0) * X)
            if deriv == 0:
                return U
            elif deriv == 1:
                return f(t, deriv=1) * numpy.dot(U, X)
            elif deriv == 2:
                UX = numpy.dot(U, X)
                UX2 = numpy.dot(UX, X)
                f1 = f(t, deriv=1)
                f2 = f(t, deriv=2)
                return f2 * UX + pow(f1,2) * UX2
            else:
                raise NotImplementedError("Higher derivatives of orthogonal matrix U(t) not implemented.")

        # A one-parameter family of eigenvalues [λ1(t), λ2(t), ..., λdim(t)]
        def eigenvalues(t, deriv=0):
            if deriv == 0:
                # [λ1(t), λ2(t), ..., λdim(t)]
                evals = numpy.zeros(dim)
                for i in range(0, dim):
                    if repeated_eigenvalues and repeated_eigenvalue_derivatives:
                        evals[i] = pow(t, i//2+1)
                    elif repeated_eigenvalues:
                        evals[i] = pow(t, i//2+1) + i*(t-t0) + i*pow(t-t0,2)
                    else:
                        evals[i] = pow(t, i+1)
                return evals
            elif deriv == 1:
                # first derivatives [λ1'(t), λ2'(t), ..., λdim'(t)]
                evals_deriv = numpy.zeros(dim)
                for i in range(0, dim):
                    if repeated_eigenvalues and repeated_eigenvalue_derivatives:
                        evals_deriv[i] = (i//2+1) * pow(t, i//2)
                    elif repeated_eigenvalues:
                        evals_deriv[i] = (i//2+1) * pow(t, i//2) + i + 2*i*(t-t0)
                    else:
                        evals_deriv[i] = (i+1) * pow(t, i)
                return evals_deriv
            elif deriv == 2:
                # second derivatives [λ1''(t), λ2''(t), ..., λdim''(t)]
                evals_deriv2 = numpy.zeros(dim)
                for i in range(0, dim):
                    if repeated_eigenvalues and repeated_eigenvalue_derivatives:
                        if i//2-1 >= 0:
                            evals_deriv2[i] = (i//2+1) * (i//2) * pow(t, i//2-1)
                    elif repeated_eigenvalues:
                        if i//2-1 >= 0:
                            evals_deriv2[i] = (i//2+1) * (i//2) * pow(t, i//2-1)
                        evals_deriv2[i] += 2*i
                    else:
                        if i-1 >= 0:
                            evals_deriv2[i] = (i+1) * i * pow(t, i-1)
                return evals_deriv2
            else:
                raise NotImplementedError("Higher derivatives are not implemented.")

        # A one-parameter family of symmetric matrices S(t) and its
        # derivatives S'(t) and S''(t).
        def symmetric_matrix(t, deriv=0):
            if deriv == 0:
                # S(t) is assembled from its eigenvalues and eigenvectors,
                #    S(t) = U(t).diag(λ1(t), λ2(t), ...).Uᵀ(t)
                evals = eigenvalues(t)
                U = eigenvectors(t)
                S = numpy.einsum('a,ia,ja->ij', evals, U, U)
                return S
            elif deriv == 1:
                # S'(t) is computed by the chain rule from its eigen decomposition.
                evals = eigenvalues(t, deriv=0)
                evals_deriv = eigenvalues(t, deriv=1)
                U = eigenvectors(t, deriv=0)
                U_deriv = eigenvectors(t, deriv=1)
                S_deriv = (
                    numpy.einsum('a,ia,ja->ij', evals_deriv, U, U) +
                    numpy.einsum('a,ia,ja->ij', evals, U_deriv, U) +
                    numpy.einsum('a,ia,ja->ij', evals, U, U_deriv)
                )
                return S_deriv
            elif deriv == 2:
                # S''(t) is computed by applying the chain rule to its eigen decomposition.
                evals_deriv0 = eigenvalues(t, deriv=0)
                evals_deriv1 = eigenvalues(t, deriv=1)
                evals_deriv2 = eigenvalues(t, deriv=2)
                U_deriv0 = eigenvectors(t, deriv=0)
                U_deriv1 = eigenvectors(t, deriv=1)
                U_deriv2 = eigenvectors(t, deriv=2)
                S_deriv2 = (
                    numpy.einsum('a,ia,ja->ij', evals_deriv2, U_deriv0, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv1, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv0, U_deriv1) +

                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv1, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv2, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv1, U_deriv1) +

                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv0, U_deriv1) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv1, U_deriv1) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv0, U_deriv2)
                )
                return S_deriv2
            else:
                raise NotImplementedError("Higher derivatives are not implemented.")

        return symmetric_matrix, eigenvectors, eigenvalues

    def check_eigensystem_derivatives(
            self,
            dim=3,
            repeated_eigenvalues=False,
            repeated_eigenvalue_derivatives=False,
            t0=0.234):
        """
        For a one-parameter famility of matrix D(t), the analytical derivatives of
        the eigenvectors and eigenvalues are compared with the numerical ones.
        Complications due to repeated eigenvalues are also checked.

        :param dim: dimension of test matrix
        :type dim: int

        :param repeated_eigenvalues:
            True: The matrix has repeated eigenvalues, but the
            eigenvalue derivatives of the repeated eigenvalues are distinct.
            False: All eigenvalues are distinct.
        :type repeated_eigenvalues: bool

        :param repeated_eigenvalue_derivatives:
            True: The matrix has repeated eigenvalues with repeated eigenvalue
            derivatives.
            False: The derivatives of repeated eigenvalues are dsitinct.
        :type repeated_eigenvalue_derivatives: bool

        :param t0: The parameter at which the derivative D(t)|_{t=t0}
            is taken.
        :type t0: float
        """
        if repeated_eigenvalue_derivatives:
            assert repeated_eigenvalues, (
                "Repeated eigenvalue derivatives are only problematic "
                "if they belong to repeated eigenvalues.")

        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=6789)

        t = t0
        # One-parameter family of symmetric matrices D(t), their eigenvalues
        # and eigenvectors as well as their 1st and 2nd derivatives.
        symmetric_matrix, eigenvectors, eigenvalues = self._random_continuous_eigensystem(
            dim, repeated_eigenvalues, repeated_eigenvalue_derivatives, t)

        # matrix function D(t) and derivatives D'(t), D''(t).
        D = symmetric_matrix(t, deriv=0)
        D_deriv1 = symmetric_matrix(t, deriv=1)
        D_deriv2 = symmetric_matrix(t, deriv=2)

        # Compare analytical derivatives with finite differences.
        dt = 0.0001
        D_plus = symmetric_matrix(t+dt, deriv=0)
        D_deriv1_plus = symmetric_matrix(t+dt, deriv=1)
        D_minus = symmetric_matrix(t-dt, deriv=0)
        D_deriv1_minus = symmetric_matrix(t-dt, deriv=1)
        # finite-difference quotients
        D_deriv1_fd = (D_plus - D_minus)/(2*dt)
        D_deriv2_fd = (D_deriv1_plus - D_deriv1_minus)/(2*dt)

        # Check that D'(t) and D''(t) are implemented correctly
        # by comparing with the finite difference quotients.
        numpy.testing.assert_almost_equal(D_deriv1_fd, D_deriv1)
        numpy.testing.assert_almost_equal(D_deriv2_fd, D_deriv2)

        # Compute eigenvalues and eigenvector derivatives from D, D' and D''.
        L, U, grad_L, grad_U = eigensystem_derivatives(
            D,
            D_deriv1[:,:,numpy.newaxis],
            D_deriv2[:,:,numpy.newaxis])

        # Reference values for eigenvalues, eigenvectors and their derivatives.
        L_ref = eigenvalues(t, deriv=0)
        U_ref = eigenvectors(t, deriv=0)
        grad_L_ref = eigenvalues(t, deriv=1)[:,numpy.newaxis]
        grad_U_ref = eigenvectors(t, deriv=1)[:,:,numpy.newaxis]

        # Compare the eigenvalues.
        numpy.testing.assert_almost_equal(numpy.sort(L_ref), numpy.sort(L))

        # The eigenvalue/eigenvector derivatives cannot be compared easily if there
        # are repeated eigenvalues, since the order of degenerate eigenvalues
        # is arbitary.
        # The kinetic energy density, on the other hand, is invariant under a reordering
        # of eigenvalues and corresponding eigenvectors.
        KED = kinetic_energy_density(L, U, grad_L, grad_U)
        KED_ref = kinetic_energy_density(L_ref, U_ref, grad_L_ref, grad_U_ref)

        numpy.testing.assert_allclose(KED_ref, KED)

    def test_eigensystem_derivatives_distinct_eigenvalues(self):
        """
        Test eigenvalue derivatives for symmetric matrices
        with distinct eigenvalues.
        """
        # Loop over points t=t0, at which the derivatives of
        # the symmetric matrix S(t) are taken.
        for t0 in [0.01, 0.2345]:
            # matrix size
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_eigensystem_derivatives(
                        dimension,
                        repeated_eigenvalues=False,
                        t0=t0)

    def test_eigensystem_derivatives_repeated_eigenvalues(self):
        """
        Test eigenvalue derivatives for symmetric matrices with
        repeated eigenvalues but distinct eigenvalue derivatives.
        """
        # Loop over points t=t0, at which the derivatives of
        # the symmetric matrix S(t) are taken.
        for t0 in [0.01, 0.2345]:
            # Loop over sizes of matrix
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_eigensystem_derivatives(
                        dimension,
                        repeated_eigenvalues=True,
                        t0=t0)

    def test_raises_exception(self):
        """
        If the repeated eigenvalues have repeated derivatives,
        higher order derivatives of D are needed to determine
        the eigenvector derivatives. This is not implemented and
        an error should be raised if this situation occurs.
        """
        with self.assertRaises(LinearAlgebraException):
            self.check_eigensystem_derivatives(
                4,
                repeated_eigenvalues=True,
                repeated_eigenvalue_derivatives=True,
                t0=0.0001)

    def test_matrix_function(self):
        """
        Check that the matrix function exp(X) is evaluated correctly for a random,
        symmetric matrix X.
        """
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run.
        random_number_generator = numpy.random.default_rng(seed=3453)

        # Create a random symmetric matrix Xᵀ = X
        dim = 4
        X = random_number_generator.random((dim, dim))
        X = 0.5 * (X + X.T)

        # Compute exp(X) using scipy.
        F_ref = scipy.linalg.expm(X)
        # Test implementation.
        F = matrix_function(numpy.exp, X)

        numpy.testing.assert_almost_equal(F_ref, F)

    def test_matrix_function_batch(self):
        """
        Check that the matrix function exp(X) is evaluated correctly for a batch
        of random, symmetric matrices.
        """
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run.
        random_number_generator = numpy.random.default_rng(seed=3453)

        # Create a batch of random symmetric matrices Xᵀ = X
        # The batch contains a matrix for each spin and coordinate.
        nspin = 2
        ncoord = 10
        # matrix dimension
        dim = 4
        X = random_number_generator.random((nspin, dim, dim, ncoord))
        # Symmetrize matrices in the batch. Axes 1 and 2 are exchanged to compute the transpose.
        X = 0.5 * (X + numpy.transpose(X, (0,2,1,3)) )

        # Compute exp(X) using scipy for each matrix in the batch.
        F_ref = numpy.zeros_like(X)
        # Loop over matrices in the batch.
        for s in range(0, nspin):
            for r in range(0, ncoord):
                F_ref[s,:,:,r] = scipy.linalg.expm(X[s,:,:,r])
        # Test implementation.
        F = matrix_function_batch(numpy.exp, X)

        numpy.testing.assert_almost_equal(F_ref, F)

    def test_matrix_function_batch_2(self):
        """
        Check that the batch implementation of a matrix function gives the same result
        as applying the function to each matrix in the batch separately.
        """
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run.
        random_number_generator = numpy.random.default_rng(seed=3453)

        # Create a batch of random symmetric matrices Xᵀ = X
        # The batch contains a matrix for each spin and coordinate.
        nspin = 2
        ncoord = 11
        # matrix dimension
        dim = 5
        X = random_number_generator.random((nspin, dim, dim, ncoord))
        # Symmetrize matrices in the batch. Axes 1 and 2 are exchanged to compute the transpose.
        X = 0.5 * (X + numpy.transpose(X, (0,2,1,3)) )

        # Scalar function f(x) defining the action on the eigenvalues.
        def func(x):
            return pow(x,2) + numpy.sin(x)

        # Compute F(X) for each matrix in the batch separately.
        F_ref = numpy.zeros_like(X)
        # Loop over matrices in batch. There is a matrix density for each spin and position.
        for s in range(0, nspin):
            for r in range(0, ncoord):
                # Apply matrix function to each matrix in the batch.
                F_ref[s,:,:,r] = matrix_function(func, X[s,:,:,r])

        # Test batch implementation.
        F = matrix_function_batch(func, X)

        numpy.testing.assert_almost_equal(F_ref, F)

    def check_matrix_function_derivatives(
            self,
            dim=3,
            repeated_eigenvalues=False,
            repeated_eigenvalue_derivatives=False,
            t0=0.234):
        """
        For a one-parameter famility of matrix X(t) and an analytical matrix function
        F(X), the analytical derivatives dF/dt = d/dt F(X(t)) are compared with the
        numerically exact ones.
        Complications due to repeated eigenvalues are also checked.

        :param dim: dimension of test matrix
        :type dim: int

        :param repeated_eigenvalues:
            True: The matrix has repeated eigenvalues, but the
            eigenvalue derivatives of the repeated eigenvalues are distinct.
            False: All eigenvalues are distinct.
        :type repeated_eigenvalues: bool

        :param repeated_eigenvalue_derivatives:
            True: The matrix has repeated eigenvalues with repeated eigenvalue
            derivatives.
            False: The derivatives of repeated eigenvalues are dsitinct.
        :type repeated_eigenvalue_derivatives: bool

        :param t0: The parameter at which the derivative X(t)|_{t=t0}
            is taken.
        :type t0: float
        """
        if repeated_eigenvalue_derivatives:
            assert repeated_eigenvalues, (
                "Repeated eigenvalue derivatives are only problematic "
                "if they belong to repeated eigenvalues.")

        t = t0
        # One-parameter family of symmetric matrices X(t), their eigenvalues
        # and eigenvectors as well as their 1st and 2nd derivatives.
        symmetric_matrix, eigenvectors, eigenvalues = self._random_continuous_eigensystem(
            dim, repeated_eigenvalues, repeated_eigenvalue_derivatives, t)

        # The matrix function F(X) is defined by a scalar function f(x)
        def func(x):
            return numpy.cos(x) + 0.1 * pow(x,3)
        # The derivative f'(x)
        def func_deriv1(x):
            return -numpy.sin(x) + 0.3 * pow(x,2)

        def matrix_function(t):
            # one-parameter family of matrices F(t) = F(X(t))
            L = eigenvalues(t, deriv=0)
            U = eigenvectors(t, deriv=0)
            # F(X(t)) = U(t).f(Λ(t)).U(t)ᵀ
            F = numpy.einsum('ia,a,ja->ij', U, func(L), U)
            return F

        # matrix function X(t) and derivative X'(t).
        X = symmetric_matrix(t, deriv=0)
        X_deriv1 = symmetric_matrix(t, deriv=1)

        # Compare analytical derivatives with finite differences.
        dt = 0.0001
        X_plus = symmetric_matrix(t+dt, deriv=0)
        X_minus = symmetric_matrix(t-dt, deriv=0)
        # finite-difference quotients
        X_deriv1_fd = (X_plus - X_minus)/(2*dt)

        # Check that X'(t) is implemented correctly
        # by comparing with the finite difference quotient.
        numpy.testing.assert_almost_equal(X_deriv1_fd, X_deriv1)

        # Compare analytical derivatives with finite differences.
        # analytical derivative (to be tested)
        F, F_deriv1 = matrix_function_derivatives(func, func_deriv1, X, numpy.expand_dims(X_deriv1,2))
        # Check that matrix function is calculated properly.
        numpy.testing.assert_almost_equal(matrix_function(t), F)
        # finite-difference quotients
        F_plus = matrix_function(t+dt)
        F_minus = matrix_function(t-dt)
        F_deriv1_fd = (F_plus - F_minus)/(2*dt)

        # Compare F'(t) with the finite difference quotiones.
        numpy.testing.assert_almost_equal(F_deriv1_fd, F_deriv1[:,:,0])

    def test_matrix_function_derivatives_distinct_eigenvalues(self):
        """
        Test derivatives of matrix function for symmetric matrices
        with distinct eigenvalues.
        """
        # Loop over points t=t0, at which the derivatives of
        # the matrix function F(X(t)) are taken.
        for t0 in [0.01, 0.2345]:
            # matrix size
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_matrix_function_derivatives(
                        dimension,
                        repeated_eigenvalues=False,
                        t0=t0)

    def test_matrix_function_derivatives_repeated_eigenvalues(self):
        """
        Test derivatives of matrix function for symmetric matrices with
        repeated eigenvalues but distinct eigenvalue derivatives.
        """
        # Loop over points t=t0, at which the derivatives of
        # the matrix function F(X(t)) are taken.
        for t0 in [0.01, 0.2345]:
            # Loop over sizes of matrix
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_matrix_function_derivatives(
                        dimension,
                        repeated_eigenvalues=True,
                        t0=t0)

    def test_matrix_function_derivatives_repeated_eigenvalue_derivatives(self):
        """
        Test derivatives of matrix function for symmetric matrices when the
        repeated eigenvalues also have repeated eigenvalue derivatives.
        """
        # Loop over points t=t0, at which the derivatives of
        # the matrix function F(X(t)) are taken.
        for t0 in [0.01, 0.2345]:
            # Loop over sizes of matrix
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_matrix_function_derivatives(
                        dimension,
                        repeated_eigenvalues=True,
                        repeated_eigenvalue_derivatives=True,
                        t0=t0)

    def test_matrix_function_derivatives_batch(self):
        """
        Test the derivative of a matrix functional for the analytic matrix function

            F(X(t)) = X(t)²

        The chain rule is not valid for matrix functions, because X and dX/dt do not
        necessarily commute, so the correct derivative is

            dF/dt = dX/dt.X + X.dX/dt.

        """
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run.
        random_number_generator = numpy.random.default_rng(seed=3453)

        # Create a batch of random symmetric matrices Xᵀ = X
        # The batch contains a matrix for each spin and coordinate.
        nspin = 2
        ncoord = 10
        # matrix dimension
        dim = 4
        # number of external parameters t
        nparam = 3
        X = random_number_generator.random((nspin, dim, dim, ncoord))
        # Symmetrize matrices in the batch. Axes 1 and 2 are exchanged to compute the transpose.
        X = 0.5 * (X + numpy.transpose(X, (0,2,1,3)) )
        # random, symmetric dX/dt
        X_deriv1 = random_number_generator.random((nspin, dim, dim, nparam, ncoord))
        X_deriv1 = 0.5 * (X_deriv1 + numpy.transpose(X_deriv1, (0,2,1,3,4)))

        # Compute F(X) = X²
        F_ref = numpy.einsum('sikr,skjr->sijr', X, X)
        # and its derivative, dF/dt = dX.X + X.dX
        F_deriv1_ref = (
            numpy.einsum('sikdr,skjr->sijdr', X_deriv1, X) +
            numpy.einsum('sikr,skjdr->sijdr', X, X_deriv1))
        # Evaluate F and F' via eigen-decomposition of X.
        F, F_deriv1 = matrix_function_derivatives_batch(
            # f(x)
            lambda x: x**2,
            # f'(x)
            lambda x: 2*x,
            X, X_deriv1)

        # Compare F(t) and F'(t) with the exact references.
        numpy.testing.assert_almost_equal(F_ref, F)
        numpy.testing.assert_almost_equal(F_deriv1_ref, F_deriv1)


if __name__ == "__main__":
    unittest.main()
