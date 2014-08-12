# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 17:30:10 2014

@author: DPD
"""


class TestNeuralNetwork():
    
    import numpy as np
    import scipy.io
 
    # set correct cost value as per Coursera Maching Learning Course (ex4)
    J_correct     = 0.287629
    J_correct_reg = 0.383770
    
    # load sample data   
    theta = scipy.io.loadmat('sample_data/ex4weights.mat')
    data = scipy.io.loadmat('sample_data/ex4data1.mat')
    data_array = np.concatenate((data['X'], data['y']), axis=1)

    X = y = data_array[:,:-1]
    y = data_array[:,-1]

    # minus each output by 1 as index starts at 0 (MATLAB starts at 1)
    for idx in np.unique(y):
        y[y==idx] = idx-1

    # set NN theta
    sample_theta = {}
    sample_theta[1] = theta['Theta1']
    sample_theta[2] = theta['Theta2']
    

       
                                         
    def test_cost_func_without_reg(self):
        
        from neural_network import NeuralNetwork
        
        # create NN instance
        nn = NeuralNetwork(X=self.X,
                           y=self.y,
                           multi_class=True,
                           options={'lamb': 0.0,
                                    'num_layers':3,
                                    'hidden_layer_size': 25,
                                    'maxiter': 2,
                                    'debug':False,
                                    'gradient_check': False,})
                                                              
        # set theta to sample data for testing 
        nn.set_theta(self.sample_theta)  
        
        # calculate outputs                                            
        a, z = nn.feed_forward()
        J = nn.calculate_cost(a, nn.y_matrix, with_reg=False)
        assert float('{0:.6g}'.format(J)) == self.J_correct
        
    def test_cost_func_with_reg(self):
        a, z = self.nn.feed_forward()
        J = self.nn.calculate_cost(a, self.nn.y_matrix, with_reg=True)
        print self.nn.regularize_cost()
        assert float('{0:.6g}'.format(J)) == self.J_correct_reg
        
    def test_two(self):
        x = "hello"
        assert hasattr(x, 'check')
        
    def test_gradient_check(self):
        
        from neural_network import NeuralNetwork
        import numpy as np
        
        np.random.shuffle(self.data_array)
        SUBSET = 100
        
        X = self.data_array[:SUBSET,:-1]
        y = self.data_array[:SUBSET,-1]
        
        # minus each output by 1 as index starts at 0 (MATLAB starts at 1)
        for idx in np.unique(y):
            y[y==idx] = idx-1
        
        
        # create NN instance
        nn = NeuralNetwork(X=X,
                           y=y,
                           multi_class=True,
                           options={'lamb': 0.0,
                                    'num_layers':3,
                                    'hidden_layer_size': 5,
                                    'maxiter': 5,
                                    'debug':False,
                                    'gradient_check': True,})
        
        
        nn.train()
        
        assert np.average(nn.gradient_diff) < 1e-8
        
    def test_train(self):
        
        from neural_network import NeuralNetwork
        import numpy as np
        
        np.random.shuffle(self.data_array)
        X = self.data_array[:,:-1]
        y = self.data_array[:,-1]
        
        # create NN instance
        self.nn = NeuralNetwork(X=X,
                                y=y,
                                multi_class=True,
                                options={'lamb': 1.0,
                                         'num_layers':3,
                                         'hidden_layer_size': 25,
                                         'training_split': .6,
                                         'maxiter': 400,
                                         'debug':False,
                                         'gradient_check': False,})
            
            
        self.nn.train()
        
    
test = TestNeuralNetwork()

#test.test_cost_func_with_reg()

test.test_train()
