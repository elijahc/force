from force import NeuronBath as Bath
import scipy as sp
import scipy.sparse

def test_initialization():
    b = Bath(N=1000)
    assert b.N == 1000
