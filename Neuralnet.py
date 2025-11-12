import numpy as np

class NeuralNet:
    def __init__(self, layers, epochs, lr, momentum, function, perc_validation):
        self.L = len(layers)    #Number of layers
        self.n = layers.copy()  #An array with the number of units in each layer

        self.h = []             #An array of arrays for the fields (h)
        for lay in range(self.L):
            self.h.append(np.zeros(layers[lay]))
        self.xi = []            #An array of arrays for the activations(Xi)
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        self.w = []             #An array of matrices for the weights
        self.w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.zeros((layers[lay], layers[lay - 1])))

        self.theta = []         #An array of arrays for thresholds
        for lay in range(self.L):
            self.theta.append(np.zeros(layers[lay]))
        
        self.delta = []         #An array of arrays for the propagation errors
        self.d_w = []           #An array of matrices for the changes on weights
        self.d_theta = []       #An array of arrays for the changes on thresholds
        self.d_w_prev = []      #An array of matrices for the previoius changes of the 
                                #weights used for the momentum term
        self.d_theta_prev = []  #An array of arrays for the previous changes of the 
                                #thresholds used for the momentum term
        self.fact = function      #The name of the activation function that will be 
                                #used (sigmoid, relu, linear, tanh)
        
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.perc_validation = perc_validation
                                
        
    
    def initialize_w_theta(self):
        for L in range(1, self.L):
            self.w[L] = np.random.randn(self.n[L], self.n[L-1]) * 0.5
        for L in range(1, self.L):
            self.theta[L] = np.random.randn(self.n[L]) * 0.5
    
    def feed_forward(self, X):
        self.xi[0] = X
        for L in range(1, self.L):
            num_neurons = self.n[L]
            for i in range (0, num_neurons):
                self.xi[L][i] = self.activation_function(i, L)
                
        return self.xi[L]
    
    def activation_function(self, i, L):
        self.h[L][i] = 0
        for j in range(0, self.n[L-1]):
            self.h[L][i] += self.w[L][i, j] * self.xi[L-1][j]
            
        self.h[L][i] -= self.theta[L][i]
            
        return (1 / (1 + np.exp(-self.h[L][i])))
        
        
                   
    def fit(self,X,Y):
        self.initialize_w_theta()
        
        for epoch in range(self.epochs):
            for pat in range(X.shape[0]):
                #Choose a random pattern (xu zu) of training set
                output = self.feed_forward(X[pat])
                print(output)
                #Back-propagate the error for this patter
                #Update the weights and threseholds
            #Feed-forward all training patterns
            #Feed-forward all validation paterns
        #Feed-forward all test partterns
        #Descale and evaluate
        
    def predict(self, X):
        return np.array([self.feed_forward(x) for x in X])
    def loss_epochs():
        return 0
