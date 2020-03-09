########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M)]
        seqs = [['' for _ in range(self.L)] for _ in range(M)]
        
        
        for state in range(self.L):
            probability = self.A_start[state] * self.O[state][x[0]]
            probs[0][state] = probability
            seqs[0][state] = str(state)
            
        for i in range(1, M):
            for state in range(self.L):
                max_prob = 0
                for index in range(self.L):
                    comp = probs[i - 1][index] * self.A[index][state] * self.O[state][x[i]]
                    if comp >= max_prob:
                        max_prob = comp
                        next_seq = seqs[i - 1][index]
                probs[i][state] = max_prob
                seqs[i][state] = next_seq + str(state)

        max_seq = ''
        index_max = probs[M - 1].index(max(probs[M - 1]))
        max_seq = seqs[M - 1][index_max]
        return max_seq


    def forward(self, x, normalize = False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M)]

        for state in range(self.L):
            probability = self.A_start[state] * self.O[state][x[0]]
            alphas[0][state] = probability

     
        for i in range(1, M):
            summ = 0
            for state in range(self.L):
                probability = self.O[state][x[i]]
                summation = 0
                for item in range(self.L):
                    summation += alphas[i - 1][item] * self.A[item][state]
                probability *= summation
                summ += summation
                alphas[i][state] = probability
            if normalize == True:
                for index in range(self.L):
                    alphas[i][index] /= summ

            
        return alphas


    def backward(self, x, normalize):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M)]

        for state in range(self.L):
            betas[M - 1][state] = 1
        if normalize == True:
            summ = self.L
            for index in range(len(betas[M - 1])):
                betas[M - 1][index] /= summ
        
        for i in range(M - 2, -1, -1):
            summ = 0
            for state in range(self.L):
                summation = 0
                for item in range(self.L):
                    summation += betas[i + 1][item] * self.A[state][item] * self.O[item][x[i + 1]]
                betas[i][state] = summation
                summ += summation
            if normalize == True:
                for index in range(self.L):
                    betas[i][index] /= summ
          
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        
        for a in range(0, self.L):
            for b in range(0, self.L):
                numerator = 0
                denominator = 0
                
                for j in range(len(Y)):
                    for i in range(len(Y[j]) - 1):
                        if Y[j][i] == b:
                            denominator += 1
                            if Y[j][i + 1] == a:
                                numerator += 1
        
                if numerator == 0:
                    self.A[b][a] = 0
                else:
                    self.A[b][a] = numerator/denominator
        
        

        # Calculate each element of O using the M-step formulas.

        for w in range(0, self.D):
            for z in range(0, self.L):
                numerator = 0
                denominator = 0
                for j in range(len(X)):
                    for i in range(len(X[j])):
                        if Y[j][i] == z:
                            denominator += 1
                            if X[j][i] == w:
                                numerator += 1

                if numerator == 0:
                    self.O[z][w] = 0
                else:
                    self.O[z][w] = numerator/denominator



    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        
        for iteration in range(N_iters):
            A_num = [[0 for i in range(self.L)] for j in range(self.L)]
            A_den = [[0 for i in range(self.L)] for j in range(self.L)]
            O_num = [[0 for i in range(self.D)] for j in range(self.L)]
            O_den = [[0 for i in range(self.D)] for j in range(self.L)]
            
            for xitem in X:
                alphas = self.forward(xitem, normalize = True)
                betas = self.backward(xitem, normalize = True)
                
                # A_num and A_den calculation
                for i in range(len(xitem)):
                    
                    if i != (len(xitem) - 1):
                        A_num_den = []
                        A_den_den = []
                        indices = []
                        for a_p in range(self.L):
                            alpha_v = alphas[i][a_p]
                            A_den_den.append(alpha_v * betas[i][a_p])
                            for b_p in range(self.L):
                                A_num_den.append(alpha_v * self.A[a_p][b_p] * self.O[b_p][xitem[i + 1]] * betas[i + 1][b_p])
                                indices.append([a_p, b_p])
                        
                        denom_summ = sum(A_num_den)
                        A_denom_summ = sum(A_den_den)
                        
                        for index in range(len(indices)):
                            a = indices[index][0]
                            b = indices[index][1]
                            if denom_summ != 0:
                                A_num[a][b] += A_num_den[index]/denom_summ
                            if A_denom_summ != 0:
                                A_den[a][b] += A_den_den[a]/A_denom_summ
               
            
                    if i == (len(xitem) - 1):
                        A_den_den = []
                        for a_p in range(self.L):
                            alpha_v = alphas[i][a_p]
                            A_den_den.append(alpha_v * betas[i][a_p])
                            
                    # O_num and O_den calculation   
                    for w in range(self.D):
                        
                        O_denom_summ = sum(A_den_den)
                        for index in range(self.L):

                            divided = A_den_den[index]/O_denom_summ
                            
                            if A_den_den != 0:
                                O_den[index][w] += divided
                                if xitem[i] == w:
                                    O_num[index][w] += divided

                                    
            for f in range(len(A_num)):
                for s in range(len(A_num[f])):
                    if A_num[f][s] == 0:
                        self.A[f][s] = 0
                    else:
                        self.A[f][s] = A_num[f][s] / A_den[f][s]
                    
            for f in range(len(O_num)):
                for s in range(len(O_num[f])):
                    if O_num[f][s] == 0:
                        self.O[f][s] = 0
                    else:
                        self.O[f][s] = O_num[f][s] / O_den[f][s]        


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
        
        states.append(np.random.choice(np.arange(self.L), 1, p=self.A_start)[0])
        emission.append(np.random.choice(np.arange(self.D), 1, p=self.O[states[len(states) - 1]])[0])
        
        for i in range(1, M):
            summ = sum(self.A[states[len(states) - 1]])
            for index in range(len(self.A[states[len(states) - 1]])):
                if summ != 0:
                    self.A[states[len(states) - 1]][index] /= summ
            next_state = np.random.choice(np.arange(self.L), 1, p=self.A[states[len(states) - 1]])[0]
            states.append(next_state)
            next_emission = np.random.choice(np.arange(self.D), 1, p=self.O[states[len(states) - 1]])[0]
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
        prob = sum(alphas[-1])
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
        prob = sum([betas[0][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
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
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)

    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
