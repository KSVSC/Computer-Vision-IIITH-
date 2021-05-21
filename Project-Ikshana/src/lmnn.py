import os
import numpy as np
from collections import Counter
import math
from sklearn.metrics import pairwise_distances
import sys


class LMNN:
    def __init__(self):
        self.k = 3
        self.convergence_tol = 0.001

    def process(self, X, labels):
        self.X = X
        X_col = X.shape[1]
        unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        length = len(unique_labels)
        self.labels = np.arange(length)
        self.L = np.eye(X_col)
        count = np.bincount(self.label_inds)
        required_k = count.min()

    def pairwise_L2(self, A, B):
        ans = pow((A-B),2).sum(axis=1)
        return ans


    def metric(self):
        L_val = self.L
        ans = L_val.T.dot(L_val)
        return ans

    def transform(self, X=None):
        L_val = self.L
        if X is not None:
            X = X
        else:
            X = self.X
        ans = L_val.dot(X.T).T
        return ans

    def select_targets(self):
        k = self.k
        X_shape = self.X.shape[0]
        lab = self.labels
        labi = self.label_inds
        Xi = self.X

        target_neighbors = np.empty((X_shape, k), dtype=int)

        for label in lab:
            inds, = np.nonzero(labi == label)
            Xi = self.X[inds]
            dd = pairwise_distances(Xi)
            np.fill_diagonal(dd, np.inf)
            nn = np.argsort(dd)[...,:k]
            target_neighbors[inds] = inds[nn]
        return target_neighbors

    def find_impostors(self, furthest_neighbors):

        impostors = []
        Lx = self.transform()
        margin_radii = self.pairwise_L2(Lx, Lx[furthest_neighbors]) + 1
        labs = self.labels[:-1]
        labi = self.label_inds

        for label in labs:
            in_inds, = np.nonzero(labi == label)
            out_inds, = np.nonzero(labi > label)

            dist = pairwise_distances(Lx[out_inds], Lx[in_inds])
            val = margin_radii[out_inds][:,None]

            i1,j1 = np.nonzero(dist < val)
            val = margin_radii[in_inds]

            i2,j2 = np.nonzero(dist < val)

            i = np.hstack((i1,i2))
            j = np.hstack((j1,j2))
            ct = np.vstack((i,j))
            ind = ct.T

            if ind.size <= 0:
                ind = 0
            else:
                ind = np.array(list(set(map(tuple,ind))))

            ct = np.atleast_2d(ind)
            i,j = ct.T
            ans = np.vstack((in_inds[j], out_inds[i]))
            impostors.append(ans)

        final = np.hstack(impostors)
        return final

    def sum_outer_products(self, data, a_inds, b_inds, weights=None):
        Xab = data[a_inds] - data[b_inds]
        if weights is not None:
            ans = np.dot(Xab.T, Xab * weights[:,None])
        else:
            ans = np.dot(Xab.T, Xab)
        return ans

    def count_edges(self, act1, act2, impostors, targets):
        chk = 1
        imp = impostors[0,act1]
        zipped = zip(imp, targets[imp])
        c = Counter(zipped)
        imp = impostors[1,act2]
        zipped = zip(imp, targets[imp])
        c.update(zipped)

        if c:
            a = []
            for i in c:
                a.append(i)
            if chk:
                active_pairs = np.array(a)
        else:
            if chk:
                active_pairs = np.empty((0,2), dtype=int)
        if chk:
            v = []
            for i in c.values():
                v.append(i)
            f = np.array(v)
            return active_pairs, f


    def fit(self, X, Y):

        self.X = X
        kval = self.k
        learn_rate = 1e-7
        self.labels = Y

        self.process(X, Y)
        target_neighbors = self.select_targets()
        impostors = self.find_impostors(target_neighbors[:,-1])
        a = np.repeat(np.arange(self.X.shape[0]), kval)
        dfG = self.sum_outer_products(self.X, target_neighbors.flatten(),a)
        df = np.zeros_like(dfG)
        a1 = [None]*kval
        a2 = [None]*kval
        for nn_idx in range(kval):
            a1[nn_idx] = np.array([])
            a2[nn_idx] = np.array([])

        # initialize gradient and L
        G = dfG * 0.5 + df * (1-0.5)
        L = self.L
        objective = np.inf

        for it in range(1000):
            df_old = df.copy()
            a1_old = [a.copy() for a in a1]
            a2_old = [a.copy() for a in a2]
            objective_old = objective
            temp = self.X.T
            Lx = L.dot(temp).T
            g0 = self.pairwise_L2(*Lx[impostors])
            arr = Lx[np.array(target_neighbors).astype('int')]
            Ni = np.sum((Lx[:,None,:] - arr)**2, axis=2)
            Ni += 1
            g1, g2 = Ni[impostors]


            total_active = 0
            for nn_idx in reversed(range(self.k)):
                act1 = g0 < g1[:,nn_idx]
                act2 = g0 < g2[:,nn_idx]
                total_active =  total_active + act1.sum() + act2.sum()

                if it <= 1:
                    plus1 = act1
                    plus2 = act2
                    minus1 = np.zeros(0, dtype=int)
                    minus2 = np.zeros(0, dtype=int)
                
                else:
                    plus1 = act1 & ~a1[nn_idx]
                    minus1 = a1[nn_idx] & ~act1
                    plus2 = act2 & ~a2[nn_idx]
                    minus2 = a2[nn_idx] & ~act2
                    
                targets = target_neighbors[:,nn_idx]
                PLUS, pweight = self.count_edges(plus1, plus2, impostors, targets)

                df = df + self.sum_outer_products(self.X, PLUS[:,0], PLUS[:,1], pweight)
                MINUS, mweight = self.count_edges(minus1, minus2, impostors, targets)
                df = df - self.sum_outer_products(self.X, MINUS[:,0], MINUS[:,1], mweight)

                in_imp, out_imp = impostors
                df = df + self.sum_outer_products(self.X, in_imp[minus1], out_imp[minus1])
                df = df + self.sum_outer_products(self.X, in_imp[minus2], out_imp[minus2])

                df = df - self.sum_outer_products(self.X, in_imp[plus1], out_imp[plus1])
                df = df - self.sum_outer_products(self.X, in_imp[plus2], out_imp[plus2])

                a1[nn_idx] = act1
                a2[nn_idx] = act2

            vals = df * (0.5)
            G = dfG * 0.5 + vals
            objective = total_active/2
            objective = objective + G.flatten().dot(L.T.dot(L).flatten())
            delta_obj = objective - objective_old
            
            if delta_obj <= 0:
                L -= learn_rate * 2 * L.dot(G)
                learn_rate = learn_rate * 1.01
            
            else:
                learn_rate = learn_rate / 2.0
                df = df_old
                a1 = a1_old
                a2 = a2_old
                objective = objective_old
                
            if it > 50 and abs(delta_obj) < self.convergence_tol:
                break

        self.L = L
        return self

