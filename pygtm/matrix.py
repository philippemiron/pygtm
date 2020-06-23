from pygtm import tools
import numpy as np
import scipy as sc
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
from sklearn.preprocessing import maxabs_scale
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class matrix_space:
    def __init__(self, domain):
        self.domain = domain
        self.N = len(domain.bins)
        self.B = None
        self.P = None
        self.M = None
        self.fi = None
        self.fo = None
        self.eigL = None
        self.L = None
        self.eigR = None
        self.R = None
        self.fi = None
        self.fo = None
        self.largest_cc = None
        self.ccs = None

    def fill_transition_matrix(self, data):
        """
        Calculate the transition matrix of all points (x0, y0) to (xT, yT) on grid domain.bins
        Args:
            data: object containing initial and final points of trajectories segments
                data.x0: array longitude of segment initial points
                data.y0: array latitude of segment initial points
                data.xT: array longitude of segment T days later
                data.yT: array latitude of segment T days later

        Returns:
            B: containing particles at initial time for each bin
            P: transition matrix
            M: number of particles per bins at time t
        """
        # Function to evaluate the transition Matrix
        # For each elements id [1:N]
        # B[id] stores the index of all particles in this bin at time t0
        idel = self.domain.find_element(data.x0, data.y0)
        self.B = [[] for i in range(0, self.N)]
        for i in range(0, len(idel)):
            if idel[i] != -1:
                self.B[idel[i]].append(i)

        # exclude bins inside domain where no particle visited
        self.B = np.asarray(self.B)
        keep = self.B.astype(bool)

        print('Domain contains %g bins. (%g bins were removed)' % (sum(keep), len(self.B) - sum(keep)))
        self.B, self.domain.bins, self.domain.id_og = tools.filter_vector([self.B, self.domain.bins, self.domain.id_og],
                                                                          keep)
        self.N = len(self.domain.bins)

        # Fill-in Transition Matrix P
        # transition probabilities from (to) element i are stored in P[i,:] (P[:,i])
        self.M = np.array([len(x) for x in self.B])  # number of particles per bin at time t0
        self.P = np.zeros((self.N, self.N))
        for i in range(0, self.N):
            # get elements of all particles in bins B[i] at the final time (-1 when outside of domain)
            idel = self.domain.find_element(data.xt[self.B[i]], data.yt[self.B[i]])
            idel = idel[idel > -1]

            if idel.size:
                # calculate the weight to add in the P matrix in function
                # of the number of particles in B[i] at time t0
                weight = np.bincount(idel)

                # keep unique element which gives the column j
                # divide the weight in fct of the number of particles
                self.P[i, np.unique(idel)] += np.divide(weight[weight > 0], self.M[i])

        # remove empty lines and columns
        # we have to do it recursively because remove one line/column might create another one
        zero_line = np.where(~self.P.any(axis=1))[0]
        while len(zero_line):
            self.P = np.delete(self.P, zero_line, axis=0)
            self.P = np.delete(self.P, zero_line, axis=1)
            self.B = np.delete(self.B, zero_line)
            self.M = np.delete(self.M, zero_line)
            self.domain.bins = np.delete(self.domain.bins, zero_line, axis=0)
            self.domain.id_og = np.delete(self.domain.id_og, zero_line)
            zero_line = np.where(~self.P.any(axis=1))[0]
        self.N = len(self.P)

        # calculate variables useful for postprocessing
        self.transition_matrix_extras(data)
        return

    def transition_matrix_extras(self, data):
        """
        Miscellaneous operations perform after the calculations of the transition matrix
        Args:
            data: trajectory object

        Returns:
            None
        """
        d = self.domain
        in_domain = np.all((data.x0 >= d.lon[0], data.x0 <= d.lon[1],
                            data.y0 >= d.lat[0], data.y0 <= d.lat[1]), axis=0)
        coming_in = np.all((~in_domain,
                            data.xt >= d.lon[0], data.xt <= d.lon[1],
                            data.yt >= d.lat[0], data.yt <= d.lat[1]), axis=0)

        # where particles are coming in
        idel = d.find_element(data.xt[np.where(coming_in)], data.yt[np.where(coming_in)])
        if idel.size:
            fi = np.bincount(idel[idel > -1], minlength=self.N)
            self.fi = fi / np.sum(fi)
        else:
            self.fi = np.zeros(self.N)

        # where particles are coming out
        self.fo = 1 - np.sum(self.P, 1)

    def largest_connected_components(self):
        """
        Restrict the matrix to the strongly connected components
        """
        # a set of nodes is consider strongly connected if there
        # is a connection between each pair of nodes in the set
        _, self.ccs = connected_components(self.P, directed=True, connection='strong')
        _, components_count = np.unique(self.ccs, return_counts=True)
        self.largest_cc = np.where(self.ccs == np.argmax(components_count))[0]

        d = self.domain
        lcc = self.largest_cc
        self.N = len(lcc)
        self.P = self.P[np.ix_(lcc, lcc)]
        self.M = self.M[lcc]
        self.B = self.B[lcc]
        d.bins = d.bins[lcc, :]
        d.id_og = d.id_og[lcc]
        if np.sum(self.fi[lcc] > 0):
            self.fi = self.fi[lcc] / np.sum(self.fi[lcc])
        else:
            self.fi = np.zeros(self.N)
        self.fo = 1 - np.sum(self.P, 1)

    @staticmethod
    def eigenvectors(m, n):
        """
        Calculate n real eigenvalues and eigenvectors of the matrix mat
        Args:
            m: square matrix
            n: number of eigenvalues and eigenvectors to calculate

        Returns:
            d: top n real eigenvalues in descending order
            v: top n eigenvectors associated with eigenvalues d
        """
        if n is None:
            d, v = sla.eig(m)  # all eigenvectors
        else:
            d, v = ssla.eigs(m, n, which='LM')

        # ordered eigenvectors in descending order
        perm = d.argsort()[::-1]
        d = d[perm]
        v = v[:, perm]

        # keep only real eigenvectors
        real_i = d.imag == 0
        d = d[real_i].real
        v = v[:, real_i].real
        v = maxabs_scale(v)
        return d, v

    def left_and_right_eigenvectors(self, n=None):
        """
        Function to call eigenvectors() for left and right calculations
        Args:
            n: number of eigenvectors to calculate (default all)

        Returns:
            eigR: right eigenvalues
            eigL: left eigenvalues (equal to eigR in our case)
            R: right eigenvectors
            L: left eigenvectors
        """
        self.eigR, self.R = self.eigenvectors(self.P, n)
        self.eigL, self.L = self.eigenvectors(np.transpose(self.P), n)

    def lagrangian_geography(self, selected_vec, n_clusters):
        """
        Cluster the eigenvectors to evaluate the Lagrangian Geography
        Args:
            selected_vec: list of selected eigenvectors
            n_clusters: number of clusters

        Returns:
            model.labels_: array corresponding to the cluster associated with each bin
        """
        vectors_geo = self.R[:, selected_vec]
        model = KMeans(n_clusters=n_clusters, random_state=1).fit(vectors_geo)

        return model.labels_

    def push_forward(self, d0, exp):
        """
        Dispersion of a initial distribution
        Args:
            d0: initial distribution
            exp: proportional to the time to evolve the density (time = exp*T)

        Returns:
            d: evolved density at t0 + exp*T
        """
        # for loop is faster than using matrix_power() with big matrix
        d = np.copy(d0)
        for i in range(0, exp):
            d = d @ self.P
        return d

    def matrix_to_graph(self, mat=None):
        """
        Transform the transition matrix into a graph
        Args:
            mat: transition matrix

        Returns:
            graph: dictionary where each key is a node and the values are its connection(s)
        """
        graph = {}

        if mat is None:
            mat = self.P

        nnz = np.nonzero(mat)
        for i in range(0, len(nnz[0])):
            key = nnz[0][i]
            if key in graph:
                graph[key].append(nnz[1][i])
            else:
                graph[key] = [nnz[1][i]]
        return graph

    def residence_time(self, target):
        """
        Calculate residence time from P matrix and a subset element list
        Args:
            target: bin indices in the zone of interest
        Returns:
            array [N]: residence time
        """
        c = np.zeros(self.N)
        a = sc.sparse.eye(len(target)) - self.P[np.ix_(target, target)]
        b = np.ones(len(a))
        # naturally, outside of the target the residence time is zero
        c[target] = sla.lstsq(a, b)[0]

        return c

    def hitting_time(self, target):
        '''
        Calculate hitting time from P matrix and a subset element list
        Args:
            target: bin indices in the zone to hit
        Returns:
            array [N]: hitting time
        '''
        #  The hitting time of set A is equal to the residence time
        # of set B with the intersection of A and B the full domain
        diff_target = np.setdiff1d(np.arange(0, len(P)), target)
        return residence_time(diff_target)
