# NumPy implementation of a first-order HMM.
# Adapted from CS155 2020 Set 6 Skeleton Code (Andrew Kang)

import random
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, L, D):
        '''
        Initializes an HMM.
        Parameters:
            L:          Number of states.
            D:          Number of observations.
        '''

        self.L = L
        self.D = D

        self.A = np.zeros((L, L))
        self.O = np.zeros((L, D))
        self.A0 = np.log(np.ones(L)/L)

    def forward(self, x, normalize = False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            alphas:     Matrix of alphas with shape (M, L).
        '''
        M = len(x)      # Length of sequence.
        L = self.L

        alphas = np.zeros((M, L))
        alphas[0,:] = self.A0 + self.O[:,x[0]]
        for i in range(1, M):
            alphas[i,:] = self.O[:,x[i]] + logsumexp(
                    alphas[i-1,:].reshape(L, 1) + self.A, axis=0)

        return alphas


    def backward(self, x):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            betas:      Vector of betas with shape (M, L).
        '''

        M = len(x)      # Length of sequence.
        L = self.L

        betas = np.zeros((M, L))
        for i in range(M-2, -1, -1):
            betas[i,:] = logsumexp(
                    self.O[:,x[i+1]] + betas[i+1,:] + self.A, axis=1)

        return betas

    def unsupervised_learning(self, X, N_iters,
            A_init=None, O_init=None, progress=True):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.

            progress:   Whether to show a progress bar for iterations.

            A_init:
            O_init:     Starting A (O). If None, A (O) will be initialized
                        randomly.
        '''
        N = len(X)
        L = self.L
        D = self.D

        # Initialize A and O.
        self.A = A_init
        if self.A is None:
            self.A = np.random.random((L, L))
            self.A /= np.sum(self.A, axis=1).reshape((L, 1))
            self.A = np.log(self.A)

        self.O = O_init
        if self.O is None:
            self.O = np.random.random((L, D))
            self.O /= np.sum(self.O, axis=1).reshape((L, 1))
            self.O = np.log(self.O)

        # We have to deal with the fact that sequences in X can have different
        # lengths. We do this by setting M to the maximum length, and then
        # (1) Padding alphas and betas with -Inf at the end. These will be used
        #     to either (a) become 0 when used in logsumexp, or (b) become -Inf
        #     (and therefore 0) when summed with other trash values.
        # (2) Padding each sequence in X with an arbitrary value at the end;
        #     this works because the only time these values are used are in
        #     summation with betas[padded] values, which are -Inf.
        Mj = [len(x) for x in X]
        M = max(Mj)
        # Routine for padding alphas and betas.
        pad = lambda a: np.pad(a, ((0,M-len(a)), (0,0)), constant_values=-np.Inf)
        # Padded version of X.
        Xp = np.array([
            np.pad(x, (0, M-len(x)), constant_values=0)
            for x in X])

        # Construct a log(one-hot) encoding of X. This is used to compute O.
        onehot = -np.ones((N, M, D))*np.Inf
        onehot[np.repeat(np.arange(N), M),
                np.tile(np.arange(M), N),
                Xp.flatten()] = 0
        onehot = onehot.reshape((N, M, 1, D))

        # Train.
        it = range(N_iters)
        if progress:
            it = tqdm(it)

        for _ in it:
            # Indexing: [seq, len, state] = [j,i,z].
            alphas = np.array([pad(self.forward(x)) for x in X])
            betas = np.array([pad(self.backward(x)) for x in X])

            # Compute marginals of the form P(y_j^i = z | x_j).
            # Indexing: [j,i,z].
            absum = alphas + betas
            m1 = absum - logsumexp(absum, axis=2).reshape(N, M, 1)
            m1[np.isnan(m1)] = -np.Inf # define 0/0=0

            # Compute marginals of the form P(y_j^i = a, y_j^{i+1} = b | x).
            # Indexing: [j,i,a,b]. Note that i runs from 0 to M-2 (inclusive).
            m2nums = (alphas[:,:-1,:].reshape((N, M-1, L, 1))
                    + betas[:,1:,:].reshape((N, M-1, 1, L))
                    + self.A.reshape((1, 1, L, L))
                    + self.O[:,Xp[:,1:]].transpose(1,2,0).reshape(N, M-1, 1, L))
            m2 = m2nums - logsumexp(m2nums, axis=(2,3)).reshape(N, M-1, 1, 1)
            m2[np.isnan(m2)] = -np.Inf

            self.O = (logsumexp(onehot + m1.reshape((N, M, L, 1)), axis=(0,1))
                    - logsumexp(m1, axis=(0,1)).reshape(L, 1))

            # For computing the denominator of A_{ba}, we don't include
            # i=len(seq)-1 for any sequence.
            for j, i in enumerate(Mj):
                m1[j,i-1,:] = -np.Inf

            self.A = (logsumexp(m2, axis=(0,1))
                    - logsumexp(m1, axis=(0,1)).reshape(L, 1))


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        L = self.L
        D = self.D
        A_start = np.exp(self.A0)
        A = np.exp(self.A)
        O = np.exp(self.O)
        
        states.append(np.random.choice(np.arange(L), 1, p=A_start)[0])
        emission.append(np.random.choice(np.arange(D), 1, p=O[states[len(states) - 1]])[0])
        
        for i in range(1, M):
            summ = sum(A[states[len(states) - 1]])
            for index in range(len(A[states[len(states) - 1]])):
                if summ != 0:
                    A[states[len(states) - 1]][index] /= summ
            next_state = np.random.choice(np.arange(L), 1, p=A[states[len(states) - 1]])[0]
            states.append(next_state)
            next_emission = np.random.choice(np.arange(D), 1, p=O[states[len(states) - 1]])[0]
            emission.append(next_emission)

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = np.sum(np.exp(alphas[-1]))
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = np.sum(np.exp(np.array(
            [betas[0,j] + self.A0[j] + self.O[j][x[0]]
                for j in range(self.L)])))

        return prob

def unsupervised_HMM(X, n_states, N_iters, **kwa):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.

        **kwa:      Passed on to HMM.unsupervised_learning().
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(L, D)
    HMM.unsupervised_learning(X, N_iters, **kwa)

    return HMM
