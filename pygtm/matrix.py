from pygtm import tools
import numpy as np
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

    def fill_transition_matrix(self, segments):
        """
        Calculate the transition matrix of all points (x0, y0) to (xT, yT) on grid domain.bins
        :param segments: object containing initial and final points of trajectories segments
        :      segments.x0: array longitude of segment initial points
        :      segments.y0: array latitude of segment initial points
        :      segments.xT: array longitude of segment T days later
        :      segments.yT: array latitude of segment T days later
        :return:
          B[N]: containing particles at initial time for each bin
          P[N,N]: transition matrix
          M[N]: number of particles per bins at time t
        """
        # Function to evaluate the transition Matrix
        # For each elements id [1:N]
        # B[id] stores the index of all particles in this bin at time t0
        idel = self.domain.find_element(segments.x0, segments.y0)
        self.B = [[] for i in range(0, self.N)]
        for i in range(0, len(idel)):
            if idel[i] != -1:
                self.B[idel[i]].append(i)

        # exclude bins inside domain where no particle visited
        self.B = np.asarray(self.B)
        keep = self.B.astype(bool)

        print('%g empty bins out of %g bins. (%1.2f%%)' % (
            len(self.B) - sum(keep), len(self.B), (len(self.B) - sum(keep)) / len(self.B) * 100))
        self.B, self.domain.bins, self.domain.id_og = tools.filter_vector([self.B, self.domain.bins, self.domain.id_og], keep)
        self.N = len(self.domain.bins)

        # Fill-in Transition Matrix P
        # transition probabilities from (to) element i are stored in P[i,:] (P[:,i])
        self.M = np.array([len(x) for x in self.B])  # number of particles per bin at time t0
        self.P = np.zeros((self.N, self.N))
        for i in range(0, self.N):
            # get elements of all particles in bins B[i] at the final time (-1 when outside of domain)
            idel = self.domain.find_element(segments.xT[self.B[i]], segments.yT[self.B[i]])
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
        self.transition_matrix_extras(segments)

        return

    def transition_matrix_extras(self, segments):
        d = self.domain
        in_domain = np.all((segments.x0 >= d.lon[0], segments.x0 <= d.lon[1],
                            segments.y0 >= d.lat[0], segments.y0 <= d.lat[1]), axis=0)
        coming_in = np.all((~in_domain,
                            segments.xT >= d.lon[0], segments.xT <= d.lon[1],
                            segments.yT >= d.lat[0], segments.yT <= d.lat[1]), axis=0)
        idel = d.find_element(segments.xT[np.where(coming_in)], segments.yT[np.where(coming_in)])
        fi = np.bincount(idel[idel > -1], minlength=self.N)
        self.fi = fi / np.sum(fi)  # where particles are coming in
        self.fo = 1 - np.sum(self.P, 1)  # where particles are coming out

        # calculate the connected components
        _, self.ccs = connected_components(self.P, directed=True, connection='strong')
        _, components_count = np.unique(self.ccs, return_counts=True)
        self.largest_cc = np.where(self.ccs == np.argmax(components_count))[0]

        return

    @staticmethod
    def eigenvectors(m, n):
        """Calculate n real eigenvalues and eigenvectors of the matrix mat
        m: square matrix
        n: number of eigenvalues and eigenvectors to calculate
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
        v = maxabs_scale(v[:, real_i].real)

        return d, v

    def left_and_right_eigenvectors(self, n=None):
        """
        :param n: number of eigenvectors to calculate (default all)
        :return:
        """
        self.eigR, self.R = self.eigenvectors(self.P, n)
        self.eigL, self.L = self.eigenvectors(np.transpose(self.P), n)

    def lagrangian_geography(self, selected_vec, n_clusters):
        # restrict the analysis to the largest strongly connected components
        vectors_geo = self.R[np.ix_(self.largest_cc, selected_vec)]
        model = KMeans(n_clusters=n_clusters, random_state=1).fit(vectors_geo)

        cluster_labels = np.zeros(self.N)
        cluster_labels[self.largest_cc] = model.labels_

        return cluster_labels

    def push_forward(self, d0, exp):
        # for loop is faster than using matrix_power() with big matrix
        d = np.copy(d0)
        for i in range(0, exp):
            d = d @ self.P

        return d

    def matrix_to_graph(self, mat=None):
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
