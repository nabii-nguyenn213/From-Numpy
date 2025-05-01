import numpy as np
import math

def stable_sigmoid(x):
    x = x.astype(np.longdouble)
    out = np.empty_like(x)

    pos_mask = x >= 0
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    neg_mask = ~pos_mask
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    return out

class Layers:
    
    def Convo(num_filter, kernel_size = (3, 3), activation = 'relu', stride = 1, padding = 'valid', input_shape = -1, initialize = None, train_bias=False):
        return Convolutional(num_filter=num_filter, kernel_size=kernel_size, activation=activation, stride=stride, padding=padding, input_shape=input_shape, initialize=initialize, train_bias=train_bias)
    
    def ConvoTranspose(num_filter, kernel_size=(3, 3), activation='relu', stride=1, padding='valid', input_shape=-1, initialize = None, train_bias=False):
        return ConvTranspose_Layer(num_filter=num_filter, kernel_size=kernel_size, activation=activation, stride=stride, padding=padding, input_shape=input_shape, initialize=initialize, train_bias=train_bias)

    def MaxPooling(pool_size=(2, 2), stride=2):
        return MaxPooling_Layer(pool_size=pool_size, stride=stride)

    def Upsample(size=(2, 2)):
        return Upsample_Layer(size=size)
    
    def Flatten():
        return Flatten_Layer()
    
    def UnFlatten(target_shape):
        return UnFlatten_Layer(target_shape=target_shape)
    
    def Dense(dim, activation = 'linear', train_bias = True, initialize = None):
        return Dense_Layer(dim=dim, activation=activation, train_bias=train_bias, initialize=initialize)

    def BatchNorm(epsilon=1e-5, momentum=0.9, num_channels=-1, train_bias=False):
        return BatchNormalization(epsilon=epsilon, momentum=momentum, num_channels=num_channels, train_bias=train_bias)
    
# ! Convolutional class
class Convolutional:
    
    def __init__(self, num_filter, kernel_size = (3, 3), activation = 'relu', stride = 1, padding = 'valid', input_shape = -1, initialize = None, train_bias=False):
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = self.generate_kernel(initialize=initialize)
        self.bias = None
        if train_bias == True: 
            self.bias = np.zeros((num_filter, 1))
    
    def generate_kernel(self, initialize):
        # print("genearate filters")
        if self.input_shape != -1:
            if initialize == None:
                return np.random.rand(self.num_filter, self.kernel_size[0], self.kernel_size[1], self.input_shape[-1])
            elif initialize == 'he':
                k_h, k_w = self.kernel_size
                in_channels = self.input_shape[-1]  # usually: (H, W, C)
                fan_in = k_h * k_w * in_channels
                std = np.sqrt(2.0 / fan_in)
                # Shape: (num_filters, kernel_height, kernel_width, in_channels)
                return np.random.normal(0, std, size=(self.num_filter, k_h, k_w, in_channels))
            elif initialize == 'xavier':
                k_h, k_w = self.kernel_size
                in_channels = self.input_shape[-1]
                out_channels = self.num_filter
                fan_in = k_h * k_w * in_channels
                fan_out = k_h * k_w * out_channels
                limit = np.sqrt(6 / (fan_in + fan_out))
                return np.random.uniform(-limit, limit, size=(out_channels, k_h, k_w, in_channels))
    
    def activate(self, input_img):
        if self.activation == 'relu':
            return np.maximum(0, input_img)
        elif self.activation == 'leakyrelu':
            alpha = 0.01
            return np.where(input_img>0, input_img, alpha*input_img)
        elif self.activation == 'tanh':
            return np.tanh(input_img)
        elif self.activation == 'sigmoid':
            input_img = input_img.astype(np.longdouble)
            return stable_sigmoid(x=input_img)
        else:
            return input_img

    def activation_derivative(self, x):
        if self.activation == "relu":
            return np.where(x < 0, 0, 1)
        elif self.activation == 'leakyrelu':
            alpha = 0.01
            return np.where(x>0, 1, alpha)
        elif self.activation == "sigmoid":
            activated=self.activate(x)
            return activated * (1 - activated)
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        else:
            return np.ones(x.shape)
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
                        if self.bias is not None:
                            output[b, i, j, k] = np.sum(region * self.weights[k]) + self.bias[k, 0]
                        else : 
                            output[b, i, j, k] = np.sum(region * self.weights[k])

        output = self.activate(output)

        if not predict:
            self.forward_cache['weights'] = self.weights
            self.forward_cache['output'] = output

        return output
    
    def backprop(self, previous_layer_cache=None, next_layer_cache=None, weights_next_layer=None, output_layer=None):
        X = previous_layer_cache['output']             # (batch, H_in, W_in, C_in)
        dA = next_layer_cache['dZ'] if next_layer_cache else output_layer                   # (batch, H_out, W_out, C_out)
        A = self.forward_cache['output']            
        dZ = dA * self.activation_derivative(A)

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
        db = np.zeros_like(self.bias)

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
                        db[f] += grad_out
                        dX_pad[b, h_start:h_end, w_start:w_end, :] += self.weights[f] * grad_out

        if self.padding == 'same':
            dX = dX_pad[:, pad_top:pad_top+H_in, pad_left:pad_left+W_in, :]
        else:
            dX = dX_pad

        self.backward_cache['dW'] = dW
        self.backward_cache['dZ'] = dX
        self.backward_cache['db'] = db
    
# ! Convolutional Transpose Layer
class ConvTranspose_Layer:
    def __init__(self, num_filter, kernel_size=(3, 3), activation='relu', stride=1, padding='valid', input_shape=-1, initialize = None, train_bias=False):
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = self.generate_kernel(initialize=initialize)
        self.bias = None
        if train_bias: 
            self.bias = np.zeros((num_filter, 1))

    def generate_kernel(self, initialize):
        # weights shape: (out_channels, kh, kw, in_channels)
        if self.input_shape != -1:
            if initialize == None:
                in_ch = self.input_shape[-1]
                kh, kw = self.kernel_size
                return np.random.randn(self.num_filter, kh, kw, in_ch) * 0.01
            elif initialize == 'he':
                k_h, k_w = self.kernel_size
                in_channels = self.input_shape[-1]  # Usually NHWC format
                fan_in = k_h * k_w * in_channels
                std = np.sqrt(2.0 / fan_in)
                return np.random.normal(0, std, size=(self.num_filter, k_h, k_w, in_channels))
            elif initialize == 'xavier':
                k_h, k_w = self.kernel_size
                in_channels = self.input_shape[-1]
                out_channels = self.num_filter  # number of filters = output channels
                fan_in = k_h * k_w * in_channels
                fan_out = k_h * k_w * out_channels
                limit = np.sqrt(6 / (fan_in + fan_out))
                return np.random.uniform(-limit, limit, size=(out_channels, k_h, k_w, in_channels))

    def activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'leakyrelu':
            alpha = 0.01 
            return np.where(x>0, x, x*alpha)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            x = x.astype(np.longdouble)
            return stable_sigmoid(x)
        else:
            return x
        
    def activation_derivative(self, raw, activated):
        if self.activation == 'relu':
            return (raw > 0).astype(raw.dtype)
        elif self.activation == 'leakyrelu':
            alpha=0.01
            return np.where(raw>0, 1, alpha)
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
                            if self.bias is not None:
                                out[n,
                                    h0:h0+kh,
                                    w0:w0+kw,
                                    k] += input_img[n, i, j, c] * self.weights[k, :, :, c] + self.bias[k, 0]
                            else : 
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

        dOut = next_layer_cache['dZ'] if next_layer_cache is not None else output_layer
        # print("dOut shape :", dOut.shape)
        # print("derivative shape :", self.activation_derivative(raw_out, act_out).shape)
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
        db = np.zeros_like(self.bias)

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

        # dX = np.clip(dX, -1, 1)
        db = np.sum(dZ, axis = (0, 1, 2))

        self.backward_cache['dZ'] = dX
        self.backward_cache['dW'] = dW
        self.backward_cache['db'] = db


# ! Max-Pooling layer
class MaxPooling_Layer:

    def __init__(self, pool_size=(2, 2), stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.weights = None
        self.bias = None
        self.forward_cache = {}
        self.backward_cache = {}

    def forward(self, input_img, input_shape=-1, predict=False):
        # input_img: (batch, H_in, W_in, C)
        batch, H_in, W_in, C = input_img.shape
        ph, pw = self.pool_size
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        H_out = (H_in - ph) // sh + 1
        W_out = (W_in - pw) // sw + 1
        output = np.zeros((batch, H_out, W_out, C))
        mask = np.zeros_like(input_img, dtype=bool)

        for b in range(batch):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * sh
                    w0 = j * sw
                    region = input_img[b, h0:h0+ph, w0:w0+pw, :]
                    # get max values and mask
                    max_vals = np.max(region, axis=(0, 1))
                    output[b, i, j, :] = max_vals
                    # create mask for backprop
                    for c in range(C):
                        # boolean mask of where region equals max
                        region_mask = (region[:, :, c] == max_vals[c])
                        mask[b, h0:h0+ph, w0:w0+pw, c] |= region_mask

        if not predict:
            self.forward_cache['mask'] = mask
            self.forward_cache['input_shape'] = input_img.shape
            self.forward_cache['dZ'] = None
            self.forward_cache['output'] = output

        return output

    def backprop(self, previous_layer_cache=None, next_layer_cache=None,
                 weights_next_layer=None, output_layer=None):
        # collect upstream gradient
        dA = next_layer_cache.get('dZ')
        mask = self.forward_cache['mask']
        dX = np.zeros(self.forward_cache['input_shape'], dtype=float)
        # distribute gradients to max locations
        batch, H_out, W_out, C = dA.shape
        ph, pw = self.pool_size
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        for b in range(batch):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * sh
                    w0 = j * sw
                    for c in range(C):
                        # only positions that were max get gradient
                        region_mask = mask[b, h0:h0+ph, w0:w0+pw, c]
                        dX[b, h0:h0+ph, w0:w0+pw, c] += region_mask * dA[b, i, j, c]

        self.backward_cache['dZ'] = dX

# ! Upsample layer
class Upsample_Layer:

    def __init__(self, size=(2, 2)):
        self.size = size
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = None
        self.bias = None

    def forward(self, input_img, input_shape=-1, predict=False):
        batch, H, W, C = input_img.shape
        sh, sw = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        output = np.repeat(np.repeat(input_img, sh, axis=1), sw, axis=2)

        if not predict:
            self.forward_cache['input_shape'] = input_img.shape
            self.forward_cache['output'] = output

        return output

    def backprop(self, previous_layer_cache=None, next_layer_cache=None,
                 weights_next_layer=None, output_layer=None):
        dA = next_layer_cache.get('dZ')
        batch, H, W, C = self.forward_cache['input_shape']
        sh, sw = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        dX = np.zeros((batch, H, W, C), dtype=float)

        for i in range(H):
            for j in range(W):
                block = dA[:, i*sh:(i*sh+sh), j*sw:(j*sw+sw), :]
                dX[:, i, j, :] = np.sum(block, axis=(1, 2))

        self.backward_cache['dZ'] = dX

# ! Flatten class
class Flatten_Layer:

    def __init__(self):
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = None # ! Flatten layer does not have weights
        self.bias = None
    
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
        self.bias = None
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
    
    def __init__(self, dim, activation = 'linear', train_bias = True, initialize = None):
        self.dim = dim
        self.activation = activation
        self.train_bias = train_bias
        self.initialize = initialize
        self.forward_cache = {}
        self.backward_cache = {}
        self.weights = self.initialize_weights()  
        self.bias = None
        if self.train_bias:
            self.bias = np.zeros((self.dim[1],1))
    
    def initialize_weights(self):
        if self.initialize == 'he':
            std = np.sqrt(2 / self.dim[0])  
            w = np.random.normal(0, std, (self.dim[0], self.dim[1]))
        elif self.initialize == 'xavier' : 
            std = np.sqrt(2 / (self.dim[0] + self.dim[1]))
            w = np.random.normal(0, std, (self.dim[0], self.dim[1]))
        elif self.initialize == 'xavier_uniform':
            limit = np.sqrt(6 / (self.dim[0] + self.dim[1]))
            w = np.random.uniform(-limit, limit, (self.dim[0], self.dim[1]))
        elif self.initialize == None : 
            w = np.random.rand((self.dim[0], self.dim[1]))
        else : 
            raise ValueError(f"Support he, xavier (uniform) , found {self.initialize}")
        return w
    
    def activate(self, x):
        return Dense_Layer.activation_function(x, self.activation)
    
    def derivative(self, x):
        return Dense_Layer.derivative_activation_function(x, self.activation)
    
    def activation_function(x, function):
        if function == "relu":
            return np.maximum(0, x)
        elif function == 'leakyrelu':
            alpha = 0.01
            return np.where(x > 0, x, alpha * x)
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
        elif function == 'leakyrelu':
            alpha= 0.01
            return np.where(x > 0, 1, alpha)
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
        if self.bias is not None:
            linear_combination = np.dot(self.weights.T, input_img) + self.bias
        else:
            linear_combination = np.dot(self.weights.T, input_img)
        
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
        if self.bias is not None:
            self.backward_cache['db'] = db

# ! BatchNormalization Layer
class BatchNormalization:

    def __init__(self, epsilon=1e-5, momentum=.9, num_channels=-1, train_bias=False):
        self.epsilon = epsilon
        self.momentum = momentum
        self.num_channels = num_channels
        
        self.weights = np.ones((1, 1, 1, num_channels)) # gamma 
        self.bias = None
        if train_bias:
            self.bias = np.zeros((1, 1, 1, num_channels)) # beta
        
        self.running_var = np.ones((1, 1, 1, num_channels))
        self.running_mean = np.zeros((1, 1, 1, num_channels))

        self.forward_cache = {}
        self.backward_cache = {}

    def forward(self, input_img, input_shape, predict=False):
        if not predict:
            mean = np.mean(input_img, axis=(0,1,2), keepdims=True)
            var  = np.var(input_img, axis=(0,1,2), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var  = self.running_var
        x_centered = input_img - mean
        inv_std = 1.0 / np.sqrt(var + self.epsilon)
        x_norm = x_centered * inv_std
        if self.bias is not None:
            out = self.weights * x_norm + self.bias
        else : 
            out = self.weights * x_norm + self.bias
        self.forward_cache['x_norm'] = x_norm
        self.forward_cache['inv_std'] = inv_std
        self.forward_cache['x_centered'] = x_centered
        self.forward_cache['mean'] = mean
        self.forward_cache['var']  = var
        self.forward_cache['input'] = input_img
        self.forward_cache['output'] = out
        return out

    def backprop(self, previous_layer_cache=None, next_layer_cache=None, weights_next_layer=None, output_layer=None):
        x_norm      = self.forward_cache['x_norm']      # (N, H, W, C)
        inv_std     = self.forward_cache['inv_std']     # (1,1,1,C)
        x_centered  = self.forward_cache['x_centered']  # (N,H,W,C)
        mean        = self.forward_cache['mean']
        var         = self.forward_cache['var']
        N, H, W, C  = self.forward_cache['input'].shape
        d_out = next_layer_cache['dZ'] if next_layer_cache is not None else output_layer
        d_gamma = np.sum(d_out * x_norm, axis=(0,1,2), keepdims=True)
        d_beta  = np.sum(d_out, axis=(0,1,2), keepdims=True)
        d_x_norm = d_out * self.weights
        d_var = np.sum(d_x_norm * x_centered * -0.5 * np.power(var + self.epsilon, -1.5), axis=(0,1,2), keepdims=True)
        d_mean = np.sum(d_x_norm * -inv_std, axis=(0,1,2), keepdims=True) + d_var * np.sum(-2.0 * x_centered, axis=(0,1,2), keepdims=True) / (N*H*W)
        d_input = d_x_norm * inv_std + d_var * 2.0 * x_centered / (N*H*W) + d_mean / (N*H*W)
        self.backward_cache['dW'] = d_gamma
        self.backward_cache['db'] = d_beta
        self.backward_cache['dZ'] = d_input
        return d_input
