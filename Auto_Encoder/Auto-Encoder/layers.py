import numpy as np
import math

def stable_sigmoid(x):
    # Use different branches to avoid overflow
    x = x.astype(np.longdouble)
    out = np.empty_like(x)

    # x >= 0
    pos_mask = x >= 0
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # x < 0: reformulate to avoid overflow
    neg_mask = ~pos_mask
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)

    return out

class Layers:
    
    def Convo(num_filter, kernel_size = (3, 3), activation = 'relu', stride = 1, padding = 'valid', input_shape = -1):
        return Convolutional(num_filter=num_filter, kernel_size=kernel_size, activation=activation, stride=stride, padding=padding, input_shape=input_shape)
    
    def ConvoTranspose(num_filter, kernel_size=(3, 3), activation='relu', stride=1, padding='valid', input_shape=-1):
        return ConvTranspose_Layer(num_filter=num_filter, kernel_size=kernel_size, activation=activation, stride=stride, padding=padding, input_shape=input_shape)
    
    def Flatten():
        return Flatten_Layer()
    
    def UnFlatten(target_shape):
        return UnFlatten_Layer(target_shape=target_shape)
    
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
        # print("genearate filters")
        if self.input_shape != -1:
            return np.random.rand(self.num_filter, self.kernel_size[0], self.kernel_size[1], self.input_shape[-1])
    
    def activate(self, input_img):
        if self.activation == 'relu':
            return np.maximum(0, input_img)
        elif self.activation == 'tanh':
            return np.tanh(input_img)
    
    def forward(self, input_img, input_shape=-1, predict=False):
        batch_size, input_height, input_width, input_channel = input_img.shape

        if self.padding == 'same':
            pad_along_height = max((math.ceil(input_height / self.stride) - 1) * self.stride + self.kernel_size[0] - input_height, 0)
            pad_along_width  = max((math.ceil(input_width / self.stride) - 1) * self.stride + self.kernel_size[1] - input_width, 0)

            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left

            input_img = np.pad(
                input_img,
                pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )

            output_height = math.ceil(input_height / self.stride)
            output_width = math.ceil(input_width / self.stride)

        else: 
            output_height = (input_height - self.kernel_size[0]) // self.stride + 1
            output_width = (input_width - self.kernel_size[1]) // self.stride + 1

        output = np.zeros((batch_size, output_height, output_width, self.num_filter))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for k in range(self.num_filter):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        region = input_img[b, h_start:h_end, w_start:w_end, :]
                        output[b, i, j, k] = np.sum(region * self.weights[k])

        output = self.activate(output)

        if not predict:
            self.forward_cache['weights'] = self.weights
            self.forward_cache['output'] = output

        return output
    
    def backprop(self, previous_layer_cache, next_layer_cache, *args, **kwargs):
        X = previous_layer_cache['output']             # (batch, H_in, W_in, C_in)
        dA = next_layer_cache['dZ']                    # (batch, H_out, W_out, C_out)
        A = self.forward_cache['output']            
        if self.activation == 'relu':
            dZ = dA * (A > 0).astype(float)
        else:  # tanh
            dZ = dA * (1 - A**2)

        batch, H_in, W_in, C_in = X.shape
        kH, kW = self.kernel_size
        stride = self.stride
        if self.padding == 'same':
            pad_h = max((math.ceil(H_in/stride)-1)*stride + kH - H_in, 0)
            pad_w = max((math.ceil(W_in/stride)-1)*stride + kW - W_in, 0)
            pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2; pad_right  = pad_w - pad_left
            X_pad = np.pad(X,
                           ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)),
                           mode='constant', constant_values=0)
        else:
            X_pad = X
            pad_top=pad_left=0

        _, H_out, W_out, C_out = dZ.shape
        dW = np.zeros_like(self.weights, dtype=float)
        dX_pad = np.zeros_like(X_pad, dtype=float)

        for b in range(batch):
            for i in range(H_out):
                for j in range(W_out):
                    for f in range(self.num_filter):
                        h_start = i * stride
                        h_end   = h_start + kH
                        w_start = j * stride
                        w_end   = w_start + kW
                        # region and gradient scalar
                        region = X_pad[b, h_start:h_end, w_start:w_end, :]
                        grad_out = dZ[b, i, j, f]
                        dW[f] += region * grad_out
                        dX_pad[b, h_start:h_end, w_start:w_end, :] += self.weights[f] * grad_out

        if self.padding == 'same':
            dX = dX_pad[:, pad_top:pad_top+H_in, pad_left:pad_left+W_in, :]
        else:
            dX = dX_pad

        self.backward_cache['dW'] = dW
        self.backward_cache['dZ'] = dX
    
# ! Convolutional Transpose Layer
class ConvTranspose_Layer:
    def __init__(self, num_filter, kernel_size=(3, 3), activation='relu', stride=1, padding='valid', input_shape=-1):
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
        # weights shape: (out_channels, kh, kw, in_channels)
        if self.input_shape != -1:
            in_ch = self.input_shape[-1]
            kh, kw = self.kernel_size
            return np.random.randn(self.num_filter, kh, kw, in_ch) * 0.01

    def activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            x = x.astype(np.longdouble)
            # return 1/(1+np.exp(-x))
            return stable_sigmoid(x)
        else:
            return x
        
    def activation_derivative(self, raw, activated):
        if self.activation == 'relu':
            return (raw > 0).astype(raw.dtype)
        elif self.activation == 'tanh':
            return 1 - np.square(activated)
        elif self.activation == 'sigmoid':
            return activated * (1 - activated)
        elif self.activation in ['linear', None]:
            return np.ones_like(raw)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
    def forward(self, input_img, input_shape=-1, predict=False):
        """
        input_img: (N, H_in, W_in, C_in)
        returns: (N, H_out, W_out, C_out) where C_out = self.num_filter
        """
        N, H_in, W_in, C_in = input_img.shape
        kh, kw = self.kernel_size
        s = self.stride

        H_raw = (H_in - 1) * s + kh
        W_raw = (W_in - 1) * s + kw

        out = np.zeros((N, H_raw, W_raw, self.num_filter))

        for n in range(N):
            for i in range(H_in):
                for j in range(W_in):
                    for c in range(C_in):
                        h0 = i * s
                        w0 = j * s
                        # for each output filter k
                        for k in range(self.num_filter):
                            out[n,
                                h0:h0+kh,
                                w0:w0+kw,
                                k] += input_img[n, i, j, c] * self.weights[k, :, :, c]

        if self.padding == 'same':
            pad_h = kh - s
            pad_w = kw - s

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            out = out[:, top:H_raw - bottom, left:W_raw - right, :]

        out_act = self.activate(out)

        if not predict:
            self.forward_cache['input'] = input_img
            self.forward_cache['weights'] = self.weights
            self.forward_cache['raw_output'] = out
            self.forward_cache['output'] = out_act

        return out_act

    def backprop(self, previous_layer_cache = None, next_layer_cache=None, weights_next_layer=None, output_layer=None):
        X = self.forward_cache['input']  # (N, H_in, W_in, C_in)
        raw_out = self.forward_cache['raw_output']
        act_out = self.forward_cache['output']

        if next_layer_cache is not None:
            dOut = next_layer_cache['dZ']
        elif output_layer is not None:
            dOut = output_layer
        else:
            raise ValueError("Need gradients from next layer or output_layer.")

        dZ = dOut * self.activation_derivative(raw_out, act_out)

        N, H_in, W_in, C_in = X.shape
        kh, kw = self.kernel_size
        s = self.stride

        if self.padding == 'same':
            H_raw = (H_in - 1) * s + kh
            W_raw = (W_in - 1) * s + kw
            pad_h = kh - s
            pad_w = kw - s
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            dZ_padded = np.zeros((N, H_raw, W_raw, self.num_filter))
            dZ_padded[:, top:H_raw-bottom, left:W_raw-right, :] = dZ
            dZ = dZ_padded

        dX = np.zeros_like(X)
        dW = np.zeros_like(self.weights)

        for n in range(N):
            for i in range(H_in):
                for j in range(W_in):
                    for c in range(C_in):
                        h0 = i * s
                        w0 = j * s
                        for k in range(self.num_filter):
                            grad_slice = dZ[n, h0:h0+kh, w0:w0+kw, k]
                            dW[k, :, :, c] += X[n, i, j, c] * grad_slice
                            dX[n, i, j, c] += np.sum(self.weights[k, :, :, c] * grad_slice)

        dX = np.clip(dX, -1, 1)

        self.backward_cache['dZ'] = dX
        self.backward_cache['dW'] = dW
    
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
            batch_size, height, width, channel = input_img.shape
            output = input_img.reshape(batch_size, height*width*channel)
        
        if predict == False:
            self.forward_cache['output'] = output
            
        return output
    
    def backprop(self, previous_layer_cache = None, next_layer_cache = None, weights_next_layer = None, output_layer = None):
        dA = np.matmul(weights_next_layer, next_layer_cache['dZ'])   
        dZ = dA.reshape(self.forward_cache['input_shape'])          
        self.backward_cache['dA'] = dA
        self.backward_cache['dZ'] = dZ

# ! UnFlatten class
class UnFlatten_Layer:
    
    def __init__(self, target_shape):
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = None # ! Flatten layer does not have weights
        self.target_shape=target_shape
        
    def forward(self, input_img, input_shape=-1, predict=False):
        self.forward_cache['input_shape'] = input_img.shape
        try:
            output = input_img.reshape(self.target_shape)
            if predict == False:
                self.forward_cache['output'] = output
            return output
        except ValueError:
            print(f"Cannot reshape {input_img.shape} to {self.target_shape}")
            return
    
    def backprop(self, previous_layer_cache=None, next_layer_cache=None, weights_next_layer=None, output_layer=None):
        if 'dZ' in next_layer_cache:
            dZ = next_layer_cache['dZ']
            dA = dZ.reshape(previous_layer_cache['output'].shape) 
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
        # print("generate weights")
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
            x = x.astype(np.longdouble)
            return stable_sigmoid(x)
            # return 1/(1 + np.exp(-x))
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
            if hasattr(self, 'bias'):
                linear_combination = np.dot(self.weights.T, input_img) + self.bias
            else:
                linear_combination = np.dot(self.weights.T, input_img)
        else:
            bias = np.squeeze(self.bias)
            linear_combination = np.dot(input_img, self.weights) + bias
        
        output = self.activate(linear_combination)
        
        if predict == False:
            self.forward_cache['linear'] = linear_combination
            self.forward_cache['output'] = output
        
        return output
    
    def backprop(self, previous_layer_cache = None, next_layer_cache = None, weights_next_layer = None, output_layer = None):      
        # ? output_layer == None -> hidden layer
        if output_layer is None:
            # * hidden
            if 'db' in next_layer_cache:
                dA = np.matmul(weights_next_layer, next_layer_cache['db'])
                self.backward_cache['dA'] = dA
                dz = np.multiply(dA, self.derivative(self.forward_cache['linear']))
            else:
                dz = np.multiply(next_layer_cache['dA'], self.derivative(x=self.forward_cache['linear']))
                dA = np.matmul(self.weights, dz)
                self.backward_cache['dA'] = dA
                
        else:
            # * output
            dA = output_layer # 10, 1
            self.backward_cache['dA'] = dA
            dz = np.matmul(self.derivative(self.forward_cache['linear']), dA)
            
        dW = np.matmul(previous_layer_cache['output'], dz.T)
        db = dz
        self.backward_cache['dZ'] = dz
        self.backward_cache['dW'] = dW
        if hasattr(self, 'bias'):
            self.backward_cache['db'] = db
