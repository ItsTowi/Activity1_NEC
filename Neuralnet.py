import numpy as np
import warnings

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
        
        self.train_errors = []
        self.val_errors = []
                                
    def _activation(self, x):
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.fact == 'tanh':
            return np.tanh(x)
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'linear':
            return x
        else:
            raise ValueError(f"Function {self.fact} not supported")

    def _activation_derivative(self, x):
        if self.fact == 'sigmoid':
            s = self._activation(x)
            return s * (1 - s)
        elif self.fact == 'tanh':
            s = self._activation(x)
            return 1 - s**2
        elif self.fact == 'relu':
            return (x > 0).astype(float)
        elif self.fact == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Función {self.fact} no soportada")

    def initialize_w_theta(self):
        for L in range(1, self.L):
            n_in = self.n[L - 1]  
            n_out = self.n[L]    
            
            # Fórmula de Xavier/Glorot para inicialización uniforme
            limit = np.sqrt(6.0 / (n_in + n_out))
            
            # Inicialización de Pesos (w)
            self.w[L] = np.random.uniform(low=-limit, high=limit, size=(n_out, n_in))
            
            # Inicialización de Umbrales (theta) a cero (práctica común y segura)
            self.theta[L] = np.zeros(n_out)
    
    def feed_forward(self, X):
        self.xi[0] = X
        for L in range(1, self.L):
            num_neurons = self.n[L]
            for i in range (0, num_neurons):
                self.xi[L][i] = self.activation_function(i, L)
                
        return self.xi[self.L - 1]  #Xi[L] (1 neuron)
    
    def activation_function(self, i, L):
        self.h[L][i] = 0
        for j in range(0, self.n[L-1]):
            self.h[L][i] += self.w[L][i, j] * self.xi[L-1][j]
            
        self.h[L][i] -= self.theta[L][i]
         
        return self._activation(self.h[L][i])
        #return (1 / (1 + np.exp(-self.h[L][i])))
    
    def error_back_propagation(self, out_x, y):
        
        L = self.L
        self.delta[L - 1] = (out_x - y) * self._activation_derivative(self.h[L - 1]) #Last layer L
        #print(self.delta[L - 1])
        
        #From layer l-1 to 2.
        for l in range(self.L - 2, 0, -1):
            #From 0 to l-1
            for j in range(self.n[l]):
                sum_delta = 0
                #From 0 to l
                for i in range(self.n[l+1]):
                    sum_delta += self.delta[l+1][i] * self.w[l+1][i,j]
                self.delta[l][j] = self._activation_derivative(self.h[l][j]) * sum_delta
                 
    def update_values(self):
        for l in range(1, self.L):
            for i in range(0, self.n[l]):
                self.d_theta[l][i]= (self.lr * self.delta[l][i]) + (self.momentum * self.d_theta_prev[l][i])
                self.theta[l][i] += self.d_theta[l][i]
                for j in range(0, self.n[l - 1]):
                    self.d_w[l][i,j] = -self.lr * self.delta[l][i] * self.xi[l-1][j] + self.momentum * self.d_w_prev[l][i,j]
                    self.w[l][i,j] += self.d_w[l][i,j]

        self.d_w_prev = [dw.copy() for dw in self.d_w]
        self.d_theta_prev = [dt.copy() for dt in self.d_theta]

    def shuffle_data(self, X, y):
        indices = np.random.permutation(X.shape[0])
        return X[indices], y[indices]

    def data_division(self, X, y):
        n_total = X.shape[0]
        n_test = int(0.20 * n_total)

        X_test = X[-n_test:]
        y_test = y[-n_test:]

        X_train_val = X[:-n_test]
        y_train_val = y[:-n_test]
        
        n_train_val = X_train_val.shape[0]
        n_val = int(self.perc_validation * n_train_val)

        X_val = X_train_val[:n_val]
        y_val = y_train_val[:n_val]

        X_train = X_train_val[n_val:]
        y_train = y_train_val[n_val:]
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test

        print("Shuffle y splits realizados:")
        print(f"  Train: {X_train.shape[0]} patrones")
        print(f"  Val:   {X_val.shape[0]} patrones")
        print(f"  Test:  {X_test.shape[0]} patrones")
    
    def calculate_quadratic_error(self, X_set, y_set):
        total_error = 0.0
        num_patterns = X_set.shape[0]

        for mu in range(num_patterns):
            x_mu = X_set[mu]
            z_mu = y_set[mu]

            # 1. Feed-forward: Obtener la predicción o(x^mu)
            o_mu = self.feed_forward(x_mu)

            squared_difference = (o_mu - z_mu)**2
            pattern_error_sum = np.sum(squared_difference) 
            
            total_error += pattern_error_sum

        return 0.5 * total_error
    
    def fit(self,X,y):
        
        self.initialize_w_theta()
        X, y = self.shuffle_data(X, y)
        self.data_division(X, y)
        num_train = self.X_train.shape[0]
        val_err = np.nan
        
        for epoch in range(self.epochs):
            for _ in range(num_train):
                idx = np.random.randint(0, num_train)   # patrón aleatorio
                #print(f"Pattern idx: {idx}")
                
                #print("Pattern")
                #Choose a random pattern (xu zu) of training set
                output = self.feed_forward(self.X_train[idx])
                #print(f"Output {output}")
                
                #Back-propagate the error for this pattern
                self.error_back_propagation(output, self.y_train[idx])
                
                #Update the weights and threseholds
                self.update_values()
            
            train_err = self.calculate_quadratic_error(self.X_train, self.y_train)
            self.train_errors.append(train_err)
            
            if self.perc_validation > 0:
                val_err = self.calculate_quadratic_error(self.X_val, self.y_val)
                self.val_errors.append(val_err)
            
            print(f"Epoch {epoch}: Train Error={train_err:.4f}, Val Error={val_err:.4f}")
        
    def predict(self, X, y_scaler=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)

        all_scaled_predictions = []
        
        # 1. Quitar el DEBUG de producción si la red ya funciona
        # print("\n--- DEBUGGING DE PREDICT (Por Patrón) ---")

        # Iterar sobre cada patrón de prueba
        for idx, x in enumerate(X):
            
            # 1. Obtener predicción ESCALADA (s)
            scaled_prediction = self.feed_forward(x) 
            
            # --- CORRECCIÓN CLAVE ---
            # Acceder al primer (y único) elemento [0] para asegurar que se añade un escalar,
            # no un array de un solo elemento.
            all_scaled_predictions.append(scaled_prediction[0])
            
            # Opcional: imprimir el debug si es necesario
            # print(f"\nPatrón {idx}:")
            # print(f"  Input 'x' (primeros 5 features): {x[:5]}")
            # print(f"  PREDICCIÓN ESCALADA (s): {scaled_prediction[0]}")
            
        # Convertir la lista de escalares en un array (N, 1) para el scaler
        scaled_predictions_array = np.array(all_scaled_predictions).reshape(-1, 1)
        
        # 2. Desescalado (Línea 15 del Listing 1)
        if y_scaler is not None:
            
            # Opcional: El warning de estancamiento es útil para la Parte 3
            if np.allclose(scaled_predictions_array, scaled_predictions_array[0]):
                warnings.warn("ALERTA: Todas las predicciones ESCALADAS son idénticas.")
            
            # Desescalamos a la escala original de log_price
            final_predictions = y_scaler.inverse_transform(scaled_predictions_array)
            
            # Opcional: imprimir el debug final
            # print("\n--- DEBUGGING DESESCALADO ---")
            # print(f"MEDIA de predicciones desescaladas: {np.mean(final_predictions):.6f}")
            # print(f"DESVIACIÓN ESTÁNDAR de predicciones desescaladas: {np.std(final_predictions):.6f}")
            
            return final_predictions
        
        # Si no se proporciona un scaler, devuelve la predicción escalada
        return scaled_predictions_array
    
    def loss_epochs(self):
        return np.array(self.train_errors), np.array(self.val_errors)
