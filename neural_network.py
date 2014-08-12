# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:32:11 2014

Last Modified 29/07/2014 - DPD

@author: DPD
"""

import math
import numpy as np
import scipy.optimize as op

class NeuralNetwork:
    
    init_theta = {}
    theta = {}
    gradient_diff = []
    y_thresh = 0.5

    def __init__(self, X, y=None, multi_class=False, options=None):
        
        _options = {'lamb': 0.0,
                    'num_layers': 3,
                    'hidden_layer_size': 10,
                    'maxiter': 100,
                    'training_split': 1.,
                    'display': True,
                    'debug': False,
                    'gradient_check': False,
                    'randomize': True}
        
        if options is None: options = _options
        else:
            for key in options.iterkeys():
                if key in _options.keys():
                    _options[key] = options[key]

        training_split = int(float(_options['training_split']) * X.shape[0])
        
        # determine if there's a test set
        self.is_test_set = not training_split == X.shape[0]

        # init useful variables 
        self.X = X[:training_split, :]
        self.X_test = X[training_split:, :]
        if y is not None:
            self.num_labels = len(np.unique(y))
            self.y = y[:training_split]
            self.y_test = y[training_split:]
        self.m = int(self.X.shape[0])
        self.n = int(self.X.shape[1])
        self.input_layer_size = int(self.X.shape[1])
        self.iter = 0
        
        # init options
        self.lamb = float(_options['lamb'])
        self.num_layers = int(_options['num_layers'])
        self.hidden_layer_size = int(_options['hidden_layer_size'])
        self.maxiter = int(_options['maxiter'])
        self.display = bool(_options['display']) 
        self.debug = bool(_options['debug'])
        self.gradient_check = bool(_options['gradient_check'])
        self.randomize = bool(_options['randomize'])
        
        if multi_class:
            self.is_multiclass = True
            if y is not None:
                self.y_matrix = self.create_matrix(self.y)
                self.y_matrix_test = self.create_matrix(self.y_test)
                self.output_layer_size = np.unique(self.y).shape[0]
        else:
            self.is_multiclass = False
            if y is not None:
                self.y_matrix = self.y
                self.y_matrix_test = self.y_test
            self.output_layer_size = 1
        
        # create instance thetas
        if self.randomize: self.init_rand_thetas() 
        
    def train(self):
        
        print 'Training Neural Network...'
        
        # slim params for minimize function
        nn_theta = self.slim(self.init_theta)
        
        # run min function, passing init_theta in correct (vector) format
        return op.minimize(lambda t: self.cost_grad_mfunc(t),
                             nn_theta,
                             method='CG',
                             jac=True,
                             callback=self.callback,
                             options={'maxiter' :self.maxiter, 'disp': True})


    def cost_grad_mfunc(self, nn_theta):
        """ 
        Acts as controller for optimizer function. It processes feed forward
        and back prop. Returns the J and theta gradient steps in correct format
        
        Params:
        nn_theta: theta values returned by optimizer function in vector (n x 1) format.
        """
        
        # rebuild nx1 theta vector into correct matrices and update instance theta values
        self.theta = self.rebuild(nn_theta)
        
        a, z = self.feed_forward()
        
        theta_grad = self.back_prop(a, z)
        
        # slim params for minimize function
        theta_grad_vec = self.slim(theta_grad)
        
        J = self.calculate_cost(a, self.y_matrix)
        
        return J, theta_grad_vec
    
    def init_gradient_check(self, theta):
        """
        Computes the gradient using "finite differences"
        and gives us a numerical estimate of the gradient.
        
        Returns a vector.
        
        Params:
        theta: vector
        """
        
        grad_approx = np.zeros((len(theta), ))
        
        epsilon = 10**-4
        
        for i in range(len(theta)):
            
            theta_plus = theta.copy()
            theta_plus[i]  = theta_plus[i] + epsilon

            a_plus, z = self.feed_forward(None, theta_plus)
            J_plus = self.calculate_cost(a_plus, self.y_matrix)

            theta_minus    = theta.copy()
            theta_minus[i] = theta_minus[i] - epsilon

            a_minus, z = self.feed_forward(None, theta_minus)
            J_minus = self.calculate_cost(a_minus, self.y_matrix)
            
            grad_approx[i] = (J_plus - J_minus) / (2. * epsilon)
            
        return grad_approx
        
    def run_gradient_check(self, theta_grad_1, theta_grad_2): 
        """
        Computes difference between numerically calculated gradients and
        back prop gradients and print results to console.
        
        Appends total difference to 'gradient_diff' list.
        
        Params:
        theta_grad_1: Numerical Gradients (in vector format)
        theta_grad_2: Back Prop Gradients (in vector format)
        
        Returns None
        """
        if self.display:
            print 'Numerical Grads'.ljust(25) + \
                  'Back Prop Grads'.ljust(25) + \
                  'Difference'.ljust(25)
                  
            for idx in range(len(theta_grad_1)):
                diff = theta_grad_1[idx] - theta_grad_2[idx]
                print '%+010.10f'.ljust(20) %theta_grad_1[idx], \
                      '%+010.10f'.ljust(20) %theta_grad_2[idx], \
                      '%+010.10f'.ljust(20) %diff
            
            # Evaluate the norm of the difference between two solutions.
            diff = np.linalg.norm(theta_grad_1 - theta_grad_2) / \
                   np.linalg.norm(theta_grad_1 + theta_grad_2)
    
            print '--'
            print 'Total Difference: %+010.10f'.ljust(25) % (diff), \
                  '('+str(diff)+')'
              
        self.gradient_diff.append(diff)

    def callback(self, xk):
        
        self.iter += 1
        self.xk = xk
        
        # calculate new outputs on TRAINING SET for cost function / evaluation
        a, z = self.feed_forward()      
        J = self.calculate_cost(a, self.y_matrix)
        accuracy = self.predict(a, self.y)
        
        # calculate new outputs on TEST SET for cost function / evaluation
        if self.is_test_set:
            a_test, z_test = self.feed_forward(self.X_test)   
            J_test = self.calculate_cost(a_test, self.y_matrix_test, with_reg=False)
            accuracy_test = self.predict(a_test, self.y_test)
        
        # Gradient checking
        if self.gradient_check:
        
            # numerical gradient using 'gradient checking'
            grad_approx_1 = self.init_gradient_check(xk)
            
            # gradient using 'back prop'
            grad_approx_2 = self.back_prop(a, z)
            grad_approx_2 = self.slim(grad_approx_2)
            
            self.run_gradient_check(grad_approx_1, grad_approx_2)    
        
        if self.is_multiclass:
            
            print '- Iteration: %d / %d' % (self.iter, self.maxiter)
            print '/ Cost: %f' % J
            print '\ Accuracy: %04.2f %%' % accuracy
            
            if self.is_test_set:
                print '/ Cost of Test: %f' % J_test
                print '\ Accuracy of Test: %f' % accuracy_test
            print '------------------------------------' 
                  
        else:
            
            # calculate F score on training set
            f = self.calc_f_score(self.y_predict, self.y)
            
            # calculate F score on test set (if present)
            if self.is_test_set:
                f_test = self.calc_f_score(accuracy_test, self.y_test)
                
            print '- Iteration: %d / %d' % (self.iter, self.maxiter)
            print '- Cost: ' + str(J)
            print '- Cost of TEST: ' + str(J_test)
            print '/ Precision: ' + str(f['precision'])
            print '\ Precision_TEST: ' + str(f_test['precision'])
            print '/ Recall: ' + str(f['recall'])
            print '\ Recall_TEST: ' + str(f_test['recall'])
            print '/ F Score: ' + str(f['f_score'])
            print '\ F Score_TEST: ' + str(f_test['f_score'])
            print '------------------------------------'  
    
    def predict(self, a, y=None):
        """
        If is_multiclass: extracts the index of the maximum output value
        from the Network and returns the percentage of correct values.
        
        Else: returns y=1 based on the output values i from the Network
        over y_thresh.
        
        Params:
        a: dict of output values for each layer.
        y: y values to check output values against.
        """
        
        if y == None: y = self.y
        
        if self.is_multiclass:
                
            # returns the index of the maximum prop output
            y_match = y[y == a[self.num_layers].argmax(axis=0)]
            
        else:
            
            y_predict = a[self.num_layers]
            y_predict[y_predict >= self.y_thresh] = 1
            y_predict[y_predict < self.y_thresh] = 0
            
            y_match = y[y == y_predict]
            
        return (len(y_match) / float(len(y))) * 100.
        
    def calc_f_score(self, y_test, y):
        """
        Takes prediced y values and actual y vals
        
        Returns dict of 'precision', 'recall' and 'f_score'
        """
        try:
            scores = y_test * 2 - y
            result = {}
            result['precision'] = len(scores[scores==1]) / float(( len(scores[scores==1]) + len(scores[scores==2]) )) 
            result['recall'] = len(scores[scores==1]) / float(( len(scores[scores==1]) + len(scores[scores==-1]) ))
            result['f_score'] = 2 * ( (result['precision'] * result['recall']) / (result['precision'] + result['recall']) )
        except ZeroDivisionError:
            result['precision'], result['recall'], result['f_score'] =  0, 0, 0
        return result
                    
    def init_rand_thetas(self):
        """
        Initialize random weights for neural networt based on
        number of hidden layers in network
        """
        
        # create first and last weights
        self.init_theta[1] = self.init_rand_theta(self.input_layer_size,
                                                  self.hidden_layer_size)
        self.init_theta[self.num_layers-1] = self.init_rand_theta(self.hidden_layer_size,
                                                                  self.output_layer_size)

        # create hidden layer weights
        for i in range(2, self.num_layers-1):
            self.init_theta[i] = self.init_rand_theta(self.hidden_layer_size,
                                                      self.hidden_layer_size)                                        


    def init_rand_theta(self, l_in, l_out, epsilon_init=None):
        """
        Initialize random weights for neural networt,
        l_in + 1 for initializing theta for handling 'bias' feature.
    
        Params:
        l_in -- number of input features (excluding bias)
        l_out -- number of outputs
        
        Returns matrix (ndarray) of size l_out x l_in + 1
        """
        if epsilon_init is None:
            epsilon_init = self.calc_epsilon(l_in, l_out)
        t = np.zeros((l_out, l_in + 1 ))       
        t = np.random.random((l_out, l_in + 1 )) * (2 * epsilon_init) - epsilon_init    
        return t
        
    def slim(self, matrix):
        """
        Unroll parameters
        """
        vector = tuple([t.flatten() for t in matrix.itervalues()])
        return np.concatenate(vector)
        
    def rebuild(self, vector):
        """
        Rebuilds matrices from vector.
        Returns dict of matrices.
        
        Params:
        vector: nx1 vector
        """
        matrix = {}
        
        for i in range(1, self.num_layers):
            
            input_size  = self.init_theta[i].shape[1]
            output_size = self.init_theta[i].shape[0]
            
            matrix[i] = vector[: output_size * input_size]
            matrix[i] = matrix[i].reshape((output_size, input_size))
            
            #reduce size of vector
            vector = vector[output_size * input_size :]
            
        return matrix

    def feed_forward(self, X=None, theta=None):
        """
        Computes output values for a Neural Network. If no params then
        computes self Neural Network output values.
        
        Params:
        X: ndarray
        theta: dict of theta ndarrays
        """
        
        test_set = True
        
        if X is None:
            X = self.X
            test_set = False
        
        if theta is None:
            theta = self.theta
        else:
            try:
                if theta[1].shape[1] != self.theta[1].shape[1]:
                    theta = self.rebuild(theta)
            except IndexError:
                theta = self.rebuild(theta)
                
        # init a and z holders 
        a = {}
        z = {}
  
        # inits a1. Each column is a training set.
        a[1] = X.T
        
        for i in range(1, self.num_layers):           
            
            # add bias term
            a[i] = self.add_ones_row(a[i])
            
            if self.debug:
                print 'calculate z_'+str(i+1)+' term (theta_'+str(i)+' * a_'+str(i)+')'
                print 'a_'+str(i) + str(a[i].shape)
                print 'theta_'+str(i) + str(theta[i].shape)
                print '--'
                
            # calculate z term
            z[i+1] = theta[i].dot(a[i])
            
            # apply sigmoid function
            a[i+1] = self.sigmoid(z[i+1])
        
        if not test_set: self.a = a           
        
        # if one axis of resulting array is 0, set shape to (0,0) for future
        # matrix multiplication compatibility
        if a[self.num_layers].shape[0] == 0 or a[self.num_layers].shape[1] == 0:
            print a[self.num_layers].shape
            a[self.num_layers].shape = (0,0)      
        
        return a, z

    def back_prop(self, a, z):
        """
        Computes gradient steps via 'back propogation' algorithm.
        
        Params:
        a: dict of output values for each layer l in network
        z: dict of 'unactivated' values for each layer l in network
        
        Returns:
        theta_grad: dict of gradient steps for theta.
        """
        
        delta = {}
        theta_grad = {}
        
        # calculate final 'error cost'
        delta[self.num_layers] = a[self.num_layers] - self.y_matrix.T
        
        # back prop error cost for each hidden layer
        for i in range(self.num_layers-1, 1, -1):
            
            delta[i] = self.theta[i][:,1:].T.dot(delta[i+1]) * self.sigmoid_gradient(z[i])
            
        # calutate gradient steps and add regularization for each theta
        for i in self.theta:
            
            # gradient steps
            theta_grad[i] = (1.0/self.m) * (delta[i+1].dot(a[i].T))
            
            # regularization (excluding bias term)
            theta_grad[i][:,1:] = theta_grad[i][:,1:] + ( (float(self.lamb)/self.m) * self.theta[i][:,1:] )
            
        return theta_grad
        
    def calculate_cost(self, a, y_matrix, with_reg=True):
        
        # calculate cost
        J =   ( -y_matrix.T * np.log(a[self.num_layers]) ) \
            - ( (1 - y_matrix.T) * np.log(1 - a[self.num_layers]) )
       
        # np.sum sums the columns (total cost for each training set)
        J = (sum(np.sum(J, axis=0))) / self.m
        
        if with_reg:
            # add regularization
            reg = self.regularize_cost()
        else:
            reg = 0.      
        
        return J + reg
             
    def regularize_cost(self):
        
        theta_sqrd = {}
        reg = 0.
        
        for i in self.theta:
            theta_sqrd[i] = self.theta[i]**2
            reg += np.sum(theta_sqrd[i][:, 1:])
            
        return (float(self.lamb) / (2.*self.m) ) * reg
              
        
    def add_ones_column(self, a):
        """
        Appends a column of 'ones' (bias term) to the 'left' of matrix
        
        Arguments:
        a -- ndarray
        """
        return np.concatenate(( np.ones((a.shape[0],1), dtype=int) , a), 1)
    
    def add_ones_row(self, a):
        """
        Appends a row of 'ones' (bias term) to the 'top' of matrix
        
        Arguments:
        a -- ndarray
        """
        return np.concatenate(( np.ones((1,a.shape[1]), dtype=int) , a), 0)
    
    def map_y(self, y):
        """
        Takes Series and returns mapped y Series
        """
        y.sort()
        unique_vals = y.unique()
        
        for i,v in list(enumerate(unique_vals)):
            y[y == v] = i
        
        y.index = range(len(y)) 
        
        return y
    
    def create_matrix(self, y):
        if self.num_labels is None:
            num_labels = len(np.unique(y))
        else:
            num_labels = self.num_labels
        y_matrix = np.zeros((len(y), num_labels))
        for i in range(len(y)):        
            y_matrix[i, y[i]] = 1        
        return y_matrix
        
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.power(math.e,-z))
     
    def sigmoid_gradient(self, z):
        return (1.0 / (1.0 + np.power(math.e,z))) * (1.0 - (1.0 / (1.0 + np.power(math.e,z))))
        
    def set_theta(self, theta):
        self.theta = theta
    
    def calc_epsilon(self, l_in, l_out):
        return math.sqrt(6) / math.sqrt(l_in + l_out)



                                 
                                     
                                     