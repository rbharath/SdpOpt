# Author:  Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

from cvxopt import matrix, solvers, spmatrix, spdiag, sparse
from numpy import bmat, zeros, reshape, array, dot, eye, outer, shape
from numpy import sqrt, real, ones
from numpy.linalg import pinv, eig, matrix_rank
from scipy.linalg import block_diag, sqrtm, pinv2
import scipy
import scipy.linalg
import numpy as np
import cvxopt.misc as misc
import math
import IPython
import pdb


def construct_coeff_matrix(x_dim, Q, C, B, E):
    # x = [s vec(Z) vec(A)]
    # F = Q^{-.5}(C-B) (not(!) symmetric)
    # J = Q^{-.5} (symmetric)
    # H = E^{.5} (symmetric)
    # ------------------------------------------
    #|Z+sI-JAF.T -FA.TJ  JAH
    #|    (JAH).T         I
    #|                       D-eps_I    A
    #|                       A.T        D^{-1}
    #|                                         I  A.T
    #|                                         A   I
    #|                                                Z
    # -------------------------------------------
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    p_dim = int(1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2)
    g_dim = 7 * x_dim
    G = spmatrix([], [], [], (g_dim ** 2, p_dim), 'd')
    # Block Matrix 1
    g1_dim = 2 * x_dim
    # Add a small positive offset to avoid taking sqrt of singular matrix
    #J = real(sqrtm(pinv(Q)+epsilon*eye(x_dim)))
    J = real(sqrtm(pinv2(Q) + epsilon * eye(x_dim)))
    H = real(sqrtm(E + epsilon * eye(x_dim)))
    F = dot(J, C - B)
    # First Block Column
    # Z+sI-JAF.T -FA.TJ
    left = 0
    top = 0
    # Z
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
            G[mat_pos, vec_pos] += 1.
    # sI
    prev = 0
    for i in range(x_dim):  # row/col on diag
        vec_pos = prev  # pos in param vector
        mat_pos = left * g_dim + i * g_dim + top + i
        G[mat_pos, vec_pos] += 1.
    # - J A F.T
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = -J A F.T
            #   M_ij = -sum_m (JA)_im (F.T)_mj
            #        = -sum_m (JA)_im F_jm
            #        = -sum_m (sum_n J_in A_nm) F_jm
            #        = -sum_m sum_n J_in A_nm F_jm
            for m in range(x_dim):
                for n in range(x_dim):
                    val = -J[i, n] * F[j, m]
                    vec_pos = prev + n * x_dim + m
                    G[mat_pos, vec_pos] += val
    # - F A.T J
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = F A.T J
            #   M_ij = sum_m (FA.T)_im J_mj
            #        = sum_m (sum_n F_in (A.T)_nm) J_mj
            #        = sum_m (sum_n F_in A_mn) J_mj
            #        = sum_m sum_n F_in A_mn J_mj
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + m * x_dim + n
                    G[mat_pos, vec_pos] += -F[i, n] * J[m, j]
    # H A.T J
    left = 0
    top = x_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = H A.T J
            #   M_ij = sum_m (HA.T)_im J_mj
            #        = sum_m (sum_n H_in (A.T)_nm) J_mj
            #        = sum_m (sum_n H_in A_mn) J_mj
            #        = sum_m sum_n H_in A_mn J_mj
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + m * x_dim + n
                    G[mat_pos, vec_pos] += H[i, n] * J[m, j]
    # Second Block Column
    # J A H
    left = x_dim
    top = 0
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = J A H
            #   M_ij = sum_m (JA)_im H_mj
            #        = sum_m (JA)_im H_mj
            #        = sum_m (sum_n J_in A_nm) H_mj
            #        = sum_m sum_n J_in A_nm H_mj
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + n * x_dim + m
                    G[mat_pos, vec_pos] += J[i, n] * H[m, j]
    # Block Matrix 2
    g2_dim = 2 * x_dim
    # Third Block Column
    # A.T
    left = 0 * x_dim + g1_dim
    top = 1 * x_dim + g1_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Fourth Block Column
    # A
    left = 1 * x_dim + g1_dim
    top = 0 * x_dim + g1_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Block Matrix 3
    g3_dim = 2 * x_dim
    # Fifth Block Column
    # A
    left = 0 * x_dim + g1_dim + g2_dim
    top = 1 * x_dim + g1_dim + g2_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Sixth Block Column
    # A.T
    left = 1 * x_dim + g1_dim + g2_dim
    top = 0 * x_dim + g1_dim + g2_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Block Matrix 4
    g4_dim = 1 * x_dim
    # Seventh Block Column
    # Z
    left = 0 * x_dim + g1_dim + g2_dim + g3_dim
    top = 0 * x_dim + g1_dim + g2_dim + g3_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
            G[mat_pos, vec_pos] += 1

    Gs = [G]
    Z = matrix(zeros((p_dim, p_dim)))
    I = matrix(eye(g_dim ** 2))
    D = matrix(sparse([[Z, G], [G.T, -I]]))
    return Gs, F, J, H


def construct_const_matrix(x_dim, D):
    # --------------------------
    #| 0   0
    #| 0   I
    #|        D-eps_I    0
    #|         0        D^{-1}
    #|                         I  0
    #|                         0  I
    #|                              0
    # --------------------------
    # Construct B1
    H1 = zeros((2 * x_dim, 2 * x_dim))
    H1[x_dim:, x_dim:] = eye(x_dim)
    H1 = matrix(H1)

    # Construct B2
    eps = 1e-4
    H2 = zeros((2 * x_dim, 2 * x_dim))
    H2[:x_dim, :x_dim] = D - eps * D
    H2[x_dim:, x_dim:] = pinv(D)
    H2 = matrix(H2)

    # Construct B3
    H3 = eye(2 * x_dim)
    H3 = matrix(H3)

    # Construct B5
    H4 = zeros((x_dim, x_dim))
    H4 = matrix(H4)

    # Construct Block matrix
    H = spdiag([H1, H2, H3, H4])
    hs = [H]
    return hs


def solve_A(x_dim, B, C, E, D, Q, max_iters, show_display):
    # x = [s vec(Z) vec(A)]
    c_dim = int(1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2)
    c = zeros((c_dim, 1))
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = int(prev + i * (i + 1) / 2 + i)
        c[vec_pos] = 1.
    cm = matrix(c)

    # Scale objective down by T for numerical stability
    eigsQinv = max([abs(1. / q) for q in eig(Q)[0]])
    eigsE = max([abs(e) for e in eig(E)[0]])
    eigsCB = max([abs(cb) for cb in eig(C - B)[0]])
    S = max(eigsQinv, eigsE, eigsCB)
    Qdown = Q / S
    Edown = E / S
    Cdown = C / S
    Bdown = B / S
    # Ensure that D doesn't have negative eigenvals
    # due to numerical issues
    min_D_eig = min(eig(D)[0])
    if min_D_eig < 0:
        # assume abs(min_D_eig) << 1
        D = D + 2 * abs(min_D_eig) * eye(x_dim)
    Gs, _, _, _ = construct_coeff_matrix(x_dim, Qdown, Cdown, Bdown, Edown)
    for i in range(len(Gs)):
        Gs[i] = -Gs[i] + 1e-6
    hs = construct_const_matrix(x_dim, D)

    solvers.options['maxiters'] = max_iters
    solvers.options['show_progress'] = show_display
    sol = solvers.sdp(cm, Gs=Gs, hs=hs)
    # check norm of A:
    avec = np.array(sol['x'])
    avec = avec[int(1 + x_dim * (x_dim + 1) / 2):]
    A = np.reshape(avec, (x_dim, x_dim), order='F')
    return sol, c, Gs, hs


def test_A_generate_constraints(x_dim):
    # Define constants
    xs = zeros((2, x_dim))
    xs[0] = ones(x_dim)
    xs[1] = 2 * ones(x_dim)
    b = 0.5 * ones((x_dim, 1))
    Q = eye(x_dim)
    D = 2 * eye(x_dim)
    B = outer(xs[1], xs[0])
    E = outer(xs[0], xs[0])
    C = outer(b, xs[0])
    return B, C, E, D, Q


def test_A_solve_sdp(x_dim):
    B, C, E, D, Q = test_A_generate_constraints(x_dim)
    sol, c, G, h = solve_A(x_dim, B, C, E, D, Q)
    return sol, c, G, h, B, C, E, D, Q
