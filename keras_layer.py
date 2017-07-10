from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras.layers.recurrent import Recurrent
from keras.activations import activations
from kieras.initializers import initializers
import numpy as np

class Force(Recurrent):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',

                 **kwargs):

        super(Force, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.state_spec = InputSpec(shape=(None, self.units))

    def build(self, input_shape):
        # MUST define self.input_spec and self.state_spec with
        # complete input shapes

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size,None,self.input_dim))

        self.kernel = self.add_weight(
                shape=(self.input_dim, self.units),
                name='kernel',
                initializer=self.kernel_initializer)

        self.recurrent_kernel = self.add_weight(
                shape=(self.units,self.units),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer)

        super(Force,self).build(input_shape)

    def step(self, inputs, states, dt=0.1):
        h = K.dot(inputs, self.kernel)
        prev_output = (1.0-dt)*states[0]
        output = h + K.dot(prev_output,self.recurrent_kernel)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
