from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Force(Layer):

    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(Force,self).__init__(**kwargs)

    def build(self,input_shape):
        self.kernel = self.add_weight(
                name='wout',
                shape=(input_shape[1],self.output_dim),
                initializer='uniform',
                trainable=True)
