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
        for lay in range(self.L):
            self.delta.append(np.zeros(layers[lay]))
            
        self.d_w = []           #An array of matrices for the changes on weights
        self.d_w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
        
        self.d_theta = []       #An array of arrays for the changes on thresholds
        for lay in range(self.L):
            self.d_theta.append(np.zeros(layers[lay]))
            
        self.d_w_prev = []      #An array of matrices for the previoius changes of the 
                                #weights used for the momentum term
        self.d_w_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))
            
        self.d_theta_prev = []  #An array of arrays for the previous changes of the 
                                #thresholds used for the momentum term
        for lay in range(self.L):
            self.d_theta_prev.append(np.zeros(layers[lay]))                     
        
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
    
    def error_back_propagation(self, out_x, y):
        
        L = self.L
        self.delta[L - 1] = (out_x - y) * self.xi[L - 1] * (1 - self.xi[L - 1])
        print(self.delta[L - 1])
        
        for l in range(self.L - 2, 0, -1):
            for j in range(0, self.n[l]):
                sum_delta = 0
                for i in range(self.n[l+1]):
                    sum_delta += self.delta[l+1][i] * self.w[l+1][i,j]
                self.delta[l][j] = self.xi[l][j] * (1 - self.xi[l][j]) * sum_delta
                 
    
    def update_values(self):
        
        for l in range(1, self.L):
            for i in range(0, self.n[l]):
                
                self.d_theta[l][i]= (self.lr * self.delta[l][i]) + (self.momentum * self.d_theta_prev[l][i])
                self.theta[l][i] = self.theta[l][i] + self.d_theta[l][i]
                
                print(f"Layer {l}, Neuron {i}:")
                print(f"  delta: {self.delta[l][i]:.6f}")
                print(f"  d_theta: {self.d_theta[l][i]:.6f}")
                print(f"  theta after: {self.theta[l][i]:.6f}")
            
                for j in range(0, self.n[l - 1]):
                    
                    old_w = self.w[l][i,j]
                    self.d_w[l][i,j] = -self.lr * self.delta[l][i] * self.xi[l-1][j] + self.momentum * self.d_w_prev[l][i,j]
                    self.w[l][i,j] += self.d_w[l][i,j]
                    
                    print(f"    Weight w[{l}][{i},{j}]: {old_w:.6f} -> {self.w[l][i,j]:.6f} (change: {self.d_w[l][i,j]:.6f})")
                
        self.d_w_prev = [dw.copy() for dw in self.d_w]
        self.d_theta_prev = [dt.copy() for dt in self.d_theta]
        print("="*50)

        
                   
    def fit(self,X,y):
        self.initialize_w_theta()
        
        for epoch in range(self.epochs):
            
            print(f"EPOCH {epoch}")
            for pat in range(X.shape[0]):
                
                print("Pattern")
                #Choose a random pattern (xu zu) of training set
                output = self.feed_forward(X[pat])
                print(f"Output {output}")
                
                #Back-propagate the error for this pattern
                self.error_back_propagation(output, y[pat])
                
                #Update the weights and threseholds
                self.update_values()
                
            #Feed-forward all training patterns
            #Feed-forward all validation paterns
        #Feed-forward all test partterns
        #Descale and evaluate
        
    def predict(self, X):
        return np.array([self.feed_forward(x) for x in X])
    def loss_epochs():
        return 0
