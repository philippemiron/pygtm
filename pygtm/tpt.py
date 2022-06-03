import numpy as np


class path_theory:
    def __init__(self, d, P, p):
        self.P = P
        self.p = p
        self.Pm = np.diag(1 / p) @ P.T @ np.diag(p)
        self.N = len(self.P)

        # center of the bins of the domain
        self.xB = np.mean(d.coords[d.bins, 0], 1)
        self.yB = np.mean(d.coords[d.bins, 1], 1)

    def committors(self, ind_a, ind_b):
        """
        Evaluate the forward committor which is the probability of starting from source set
        (ind_a) to reach the target set (ind_b) and the backward committor which probability
        in backward time to start at the target and reach the source

        Args:
            ind_a: array of indices of the source
            ind_b: array of indices of the target

        Returns:
            q_b: backward committor
            q_f: forward committor
        """

        # indices of the domain \ (target and source)
        ind_c = np.setdiff1d(np.arange(0, self.N), np.union1d(ind_a, ind_b))

        # retrieve subset of the transition matrix
        # from states in c to c in forward and backward
        P_c = self.P[np.ix_(ind_c, ind_c)]
        Pm_c = self.Pm[np.ix_(ind_c, ind_c)]

        # forward from c to b, backward from c to a
        P_cb = self.P[np.ix_(ind_c, ind_b)]
        Pm_ca = self.Pm[np.ix_(ind_c, ind_a)]

        # compute forward committor on ind_c
        qf_C = np.zeros(len(ind_c))
        b = np.sum(P_cb, axis=1)
        inv1 = np.linalg.inv(np.diag(np.ones(len(ind_c))) - P_c)
        qf_C = inv1.dot(b)

        # add entries of the forward committor on source a, target b and domain c
        # (i.e. q_f is 0 on a, 1 on b)
        q_f = np.zeros(self.N)
        q_f[ind_b] = 1.0
        q_f[ind_c] = qf_C

        # compute backward committor on ind_c
        qb_C = np.zeros(len(ind_c))
        a = np.sum(Pm_ca, axis=1)
        inv2 = np.linalg.inv(np.diag(np.ones(len(ind_c))) - Pm_c)
        qb_C = inv2.dot(a)

        # add entries of the backward committor on source a and target b and domain c
        # (i.e. q_b is 1 on a, 0 on b)
        q_b = np.zeros(self.N)
        q_b[ind_a] = 1.0
        q_b[ind_c] = qb_C

        return q_b, q_f

    def reactive_trajectories_current(self, q_b, q_f):
        """
        Evaluate the reactive trajectories from the forward and backward committor

        Args:
            q_b: array backward committor
            q_f: array forward committor

        Returns:
            f: matrix of the reactive current from and to all bins of the domain
            fx: array [N] x-direction of the current for each bin
            fy: array [N] y-direction of the current for each bin
        """
        # reactive current
        # f = q_i^+ P_ij q_j^+ p_i
        f = np.diag(q_b * self.p) @ self.P @ np.diag(q_f)
        np.fill_diagonal(f, 0)  # Metzner sets the diagonal to 0

        # to obtain the reactive current we have to evaluate
        # flow out - flow in
        # positive value means exiting the domain
        fp = f - f.T
        fp[fp < 0] = 0

        # we don't calculate the current towards
        # the virtual nirvana state
        if self.N > len(self.xB):
            N = self.N - 1
        else:
            N = self.N

        fx = np.ones(N)
        fy = np.ones(N)
        for i in range(0, N):
            ex = self.xB - self.xB[i]
            ey = self.yB - self.yB[i]
            e = np.hypot(ex, ey)
            e[e == 0] = 1
            ex = ex / e
            ey = ey / e
            fx[i] = np.sum(fp[i, :N] * ex)
            fy[i] = np.sum(fp[i, :N] * ey)

        return f, fx, fy

    def reactive_trajectories_properties(self, q_b, q_f, f, ind_a, ind_b):
        """
        Evaluate the density, rate and expected time of reactive trajectories

        Args:
            q_b: array backward committor
            q_f: array forward committor
            f: matrix of the reactive currents
            ind_a: array of indices of the source
            ind_b: array of indices of the target

        Returns:
            mu: array of the density of reactive trajectories
            k: rate of reactive trajectories
            t: expected time of reactive trajectories

        """
        # density of reactive trajectories
        mu = q_b * self.p * q_f
        z = np.sum(mu)

        if z == 0:
            mu = np.nan
        else:
            mu /= z

        # rate of reactive trajectories
        k = np.sum(f[ind_a, :])  # out of the source

        # which is equal to the rate into the target
        # k = np.sum(f[:, ind_b])  # in the target

        # mean duration of reactive trajectories
        if k == 0:
            t = np.nan
        else:
            t = z / k

        return mu, k, t
