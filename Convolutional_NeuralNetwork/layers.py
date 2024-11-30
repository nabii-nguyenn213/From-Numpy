import numpy as np

class Layers:
    
    def Convo(num_filter, kernel_size = (3, 3), activation = 'relu', stride = 1, padding = 'valid', input_shape = -1):
        return Convolutional(num_filter=num_filter, kernel_size=kernel_size, activation=activation, stride=stride, padding=padding, input_shape=input_shape)
    
    def MaxPool(kernel_size, stride = 2, padding = 'valid'):
        return MaxPooling(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def AvgPool(kernel_size, stride = 2, padding = 'valid'):
        return AveragePooling(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def Flatten():
        return Flatten_Layer()
    
    def Dense(dim, activation = 'linear', train_bias = True, xavier_uniform = True):
        return Dense_Layer(dim=dim, activation=activation, train_bias=train_bias, xavier_uniform=xavier_uniform)
    
# ! Convolutional class
class Convolutional:
    
    def __init__(self, num_filter, kernel_size = (3, 3), activation = 'relu', stride = 1, padding = 'valid', input_shape = -1):
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = self.generate_kernel()
    
    def generate_kernel(self):
        print("genearate filters")
        if self.input_shape != -1:
            return np.random.rand(self.num_filter, self.kernel_size[0], self.kernel_size[1], self.input_shape[-1])
    
    def activate(self, input_img):
        if self.activation == 'relu':
            return np.maximum(0, input_img)
        elif self.activation == 'tanh':
            return np.tanh(input_img)
    
    def forward(self, input_img, input_shape = -1, predict = False):
           
        batch_size, input_height, input_width, input_channel = input_img.shape
        
        if self.padding == 'same':
            pad_height = ((input_height - 1) * self.stride + self.kernel_size[0] - input_height) // 2
            pad_width = ((input_width - 1) * self.stride + self.kernel_size[1] - input_width) // 2
            input_img = np.pad(input_img, pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
            
            output_height = (input_height + 2 * pad_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width + 2 * pad_width - self.kernel_size[1]) // self.stride + 1
        else:
            output_height = (input_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width - self.kernel_size[1]) // self.stride + 1
        
        output = np.zeros((batch_size, output_height, output_width, self.num_filter))
        
        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for k in range(self.num_filter):
                        region = input_img[b,
                                           i * self.stride : i * self.stride + self.kernel_size[0],
                                           j * self.stride : j * self.stride + self.kernel_size[1], 
                                           :]
                        output[b, i, j, k] = np.sum(region * self.weights[k])

                
        output = self.activate(output)
        
        if predict == False:
            self.forward_cache['weights'] = self.weights
            self.forward_cache['output'] = output
        
        return output
    
    def backprop(self, previous_layer_cache = None, next_layer_cache = None, weights_next_layer = None, output_layer = None):
        input_img = previous_layer_cache['output']
        last_gradient = next_layer_cache['dZ']
        B_out, H_out, W_out, C_out = last_gradient.shape
        H_k, W_k, C_in = self.weights.shape[1:]
        
        dL_dZ = np.zeros_like(input_img)
        dL_dW = np.zeros_like(self.weights)
        
        flipped_weights = np.flip(self.weights, axis=(1, 2))

        for b_out in range(B_out):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + H_k
                        w_start = w * self.stride
                        w_end = w_start + W_k

                        dL_dZ[b_out, h_start:h_end, w_start:w_end, :] = dL_dZ[b_out, h_start:h_end, w_start:w_end, :] + last_gradient[b_out, h, w, c_out] * flipped_weights[c_out]
                        dL_dW[c_out] += last_gradient[b_out, h, w, c_out] * input_img[b_out, h_start:h_end, w_start:w_end, :]
                    
        dL_dZ = np.clip(dL_dZ, -1, 1)
        self.backward_cache['dZ'] = dL_dZ
        self.backward_cache['dW'] = dL_dW

    
# ! Max Pooling class 
class MaxPooling:
    
    def __init__(self, kernel_size, stride = 2, padding = 'valid'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = None # ! Pooling layer does not have weights
    
    def forward(self, input_img, input_shape = -1):
            
        input_height, input_width, input_channel = input_img.shape
        if self.padding == 'same':
            pad_height = ((input_height - 1) * self.stride + self.kernel_size[0] - input_height) // 2
            pad_width = ((input_width - 1) * self.stride + self.kernel_size[1] - input_width) // 2
            input_img = np.pad(input_img, pad_width=((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
            
            output_height = (input_height + 2 * pad_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width + 2 * pad_width - self.kernel_size[1]) // self.stride + 1
        else:
            output_height = (input_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width - self.kernel_size[1]) // self.stride + 1
        
        output = np.zeros((output_height, output_width, input_channel))
        max_indices = np.zeros((output_height, output_width, input_channel, 2), dtype=int)
        
        for d in range(input_channel):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_img[i * self.stride:i * self.stride + self.kernel_size[0], j * self.stride:j * self.stride + self.kernel_size[1], d] 
                    output[i, j, d] = np.max(region)
                    
                    max_indices[i, j, d] = np.unravel_index(np.argmax(region), region.shape)
        
        
        self.forward_cache['max_indices'] = max_indices
        self.forward_cache['output'] = output
        
        return output
        
    def backprop(self, previous_layer_cache = None, next_layer_cache = None, weights_next_layer = None, output_layer = None):
        
        input_ = previous_layer_cache['output']
        mask = self.forward_cache['max_indices']
        doutput = next_layer_cache['dZ']
        dinput = np.zeros_like(input_)


        for h in range(0, input_.shape[0] - self.kernel_size[0] + 1, self.stride):
            for w in range(0, input_.shape[1] - self.kernel_size[1] + 1, self.stride):
                for c in range(input_.shape[2]):
                    
                    max_indices = mask[h // self.stride, w // self.stride, c]
                    dA_value = doutput[h // self.stride, w // self.stride, c]
                    row, col = max_indices
                    dinput[h + row, w + col, c] += dA_value
        self.backward_cache['dA'] = dinput
        # self.backward_cache['dW'] = doutput

# ! Avg Pooling class
class AveragePooling(MaxPooling):
    
    def __init__(self, kernel_size, stride = 2, padding = 'valid'):
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, input_img, input_shape = -1, predict = False):
        batch_size, input_height, input_width, input_channel = input_img.shape
        
        if self.padding == 'same':
            pad_height = ((input_height - 1) * self.stride + self.kernel_size[0] - input_height) // 2
            pad_width = ((input_width - 1) * self.stride + self.kernel_size[1] - input_width) // 2
            input_img = np.pad(input_img, pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
            
            output_height = (input_height + 2 * pad_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width + 2 * pad_width - self.kernel_size[1]) // self.stride + 1
        else:
            output_height = (input_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width - self.kernel_size[1]) // self.stride + 1
        
        output = np.zeros((batch_size, output_height, output_width, input_channel))
        for b in range(batch_size):
            for d in range(input_channel):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input_img[b, i * self.stride:i * self.stride + self.kernel_size[0], j * self.stride:j * self.stride + self.kernel_size[1], d] 
                        output[b, i, j, d] = np.mean(region)
        
        if predict == False:
            self.forward_cache['output'] = output
        
        return output
    
    def backprop(self, previous_layer_cache=None, next_layer_cache=None, weights_next_layer=None, output_layer=None):
        input_img = previous_layer_cache['output']
        batch_size, input_height, input_width, input_channel = input_img.shape
        g_batch, g_height, g_width, g_channel = next_layer_cache['dZ'].shape
        
        dZ = np.zeros_like(input_img)
        for b in range(batch_size):
            for c in range(input_channel):
                for i in range(g_height):
                    for j in range(g_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size[1]

                        gradient = next_layer_cache['dZ'][b, i, j, c] / (self.kernel_size[0] * self.kernel_size[1])
                        
                        dZ[b, h_start:h_end, w_start:w_end, c] += gradient

        self.backward_cache['dZ'] = dZ
        
# ! Flatten class
class Flatten_Layer:

    def __init__(self):
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = None # ! Flatten layer does not have weights
    
    def forward(self, input_img, input_shape = -1, predict = False):
        self.forward_cache['input_shape'] = input_img.shape
        batch_size = input_img.shape[0]
        if batch_size == 1:
            output = input_img.flatten()
            output = output.reshape(output.shape[0], 1)
        else:
            squeezed = np.squeeze(input_img)
            output = squeezed.reshape(batch_size, squeezed.shape[1])
        
        if predict == False:
            self.forward_cache['output'] = output
            
        return output
    
    def backprop(self, previous_layer_cache = None, next_layer_cache = None, weights_next_layer = None, output_layer = None):
        '''
        reshape gradient to match to previous layer
        '''
        dA = np.matmul(weights_next_layer, next_layer_cache['dZ'])   # (120, 84) (84, 1) -> (120, 1)
        dZ = dA.reshape(self.forward_cache['input_shape'])           # Reshape -> (1, 1, 120)
        self.backward_cache['dA'] = dA
        self.backward_cache['dZ'] = dZ

# ! Dense class
class Dense_Layer:
    
    def __init__(self, dim, activation = 'linear', train_bias = True, xavier_uniform = True):
        self.dim = dim
        self.activation = activation
        self.train_bias = train_bias
        self.xavier_uniform = xavier_uniform
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = self.initialize_weights(self.xavier_uniform)  
        if self.train_bias:
            self.bias = np.zeros((self.dim[1], 1))
        
    
    def initialize_weights(self, uniform = True):
        print("generate weights")
        if uniform:
            limit = np.sqrt(6 / (self.dim[0] + self.dim[1]))
            w = np.random.uniform(-limit, limit, (self.dim[0], self.dim[1]))
        else:
            std = np.sqrt(2 / (self.dim[0] + self.dim[1]))
            w = np.random.normal(0, std, (self.dim[0], self.dim[1]))
        return w
    
    def activate(self, x):
        return Dense_Layer.activation_function(x, self.activation)
    
    def derivative(self, x):
        return Dense_Layer.derivative_activation_function(x, self.activation)
    
    def activation_function(x, function):
        if function == "relu":
            return np.maximum(0, x)
        elif function == "sigmoid":
            return 1/(1 + np.exp(-x))
        elif function == "tanh":
            return np.tanh(x)
        elif function == 'softmax':
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:
            return x
        
    def derivative_activation_function(x, function):
        if function == "relu":
            return np.where(x < 0, 0, 1)
        elif function == "sigmoid":
            return Dense_Layer.activation_function(x, function) * (1 - Dense_Layer.activation_function(x, function))
        elif function == "tanh":
            return 1 - np.tanh(x) ** 2
        elif function == 'softmax':
            softmax_output = Dense_Layer.activation_function(x, function)        
            s = softmax_output.reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            return jacobian
        else:
            return np.ones(x.shape)
    
    def forward(self, input_img, input_shape = -1, predict = False):
        check_input = np.squeeze(input_img)
        if check_input.ndim == 1:
            linear_combination = np.dot(self.weights.T, input_img) + self.bias
        else:
            bias = np.squeeze(self.bias)
            linear_combination = np.dot(input_img, self.weights) + bias
        
        output = self.activate(linear_combination)
        
        if predict == False:
            self.forward_cache['linear'] = linear_combination
            self.forward_cache['output'] = output
        
        return output
    
    def backprop(self, previous_layer_cache = None, next_layer_cache = None, weights_next_layer = None, output_layer = None):      
        # ? output_layprer == None -> hidden layer
        if output_layer is None:
            # * hidden
            dA = np.matmul(weights_next_layer, next_layer_cache['db'])
            self.backward_cache['dA'] = dA
            dz = np.multiply(dA, self.derivative(self.forward_cache['linear']))
        else:
            # * output
            dA = output_layer
            self.backward_cache['dA'] = dA
            dz = np.matmul(self.derivative(self.forward_cache['linear']), dA)
        dW = np.matmul(previous_layer_cache['output'], dz.T) 
        db = dz
        self.backward_cache['dZ'] = dz
        self.backward_cache['dW'] = dW
        self.backward_cache['db'] = db