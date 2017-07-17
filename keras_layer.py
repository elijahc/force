from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras.layers.recurrent import Recurrent
from keras import activations
from keras import initializers
import numpy as np

class Force(Recurrent):

    def __init__(self, units,
                 readouts=1,
                 activation=None,
                 recurrent_activation='tanh',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',

                 **kwargs):

        super(Force, self).__init__(**kwargs)
        self.units = units
        self.readouts = readouts
        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.state_spec = InputSpec(shape=(None, self.units))

    def build(self, input_shape):
        # MUST define self.input_spec and self.state_spec with
        # complete input shapes

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size,None,self.input_dim))
        self.states = [None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(
                shape=(self.input_dim, self.units),
                name='kernel',
                initializer=self.kernel_initializer)
        self.readout_kernel = self.add_weight(
                shape=(self.units, self.readouts),
                name='readout_kernel',
                initializer=self.kernel_initializer)
        self.recurrent_kernel = self.add_weight(
                shape=(self.units + self.readouts,self.units),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units + self.readouts,),
                                        name='bias',
                                        initializer=self.bias_initializer,)
        else:
            self.bias = None

        self.recurrent_kernel_z = self.recurrent_kernel[-self.readouts:,:]

        self.recurrent_kernel_r = self.recurrent_kernel[:self.units,:]

        if self.use_bias:
            self.bias_z = self.bias[-self.readouts:]
            self.bias_h = self.bias[:self.units]
        else:
            self.bias_z = self.bias_r = self.bias_h = None

        self.built = True

    def step(self, inputs, states, dt=0.1):
        h1 = states[0] # Previous State
        r1 = self.recurrent_activation(h1) # Previous firing rates
        z1 = K.dot(r1,self.readout_kernel)
        if self.activation is not None:
            z1 = self.activation(z1)
        x_h = K.dot(inputs, self.kernel)
        z_h = K.dot(z1, self.recurrent_kernel_z)
        r_h = K.dot(r1,self.recurrent_kernel_r)
        if self.use_bias:
            x_h = K.bias_add(x_h, self.bias_h)
            z_h = K.bias_add(z_h, self.bias_z)

        h = (1.0-dt)*h1 + r_h*dt + z_h*dt # Recurrent
        h += x_h*dt # Inputs
        r = self.recurrent_activation(h)
        z = K.dot(r,self.readout_kernel)
        if self.use_bias:
            z = K.bias_add(z,self.bias_z)
        if self.activation is not None:
            z = self.activation(z)

        return z, [h]


    def compute_output_shape(self, input_shape):
        if isinstance(input_shape,list):
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.units)
        else:
            output_shape = (input_shape[0],self.readouts)

        if self.return_state:
            state_shape = [(input_shape[0],self.units) for _ in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape
