import numpy as np
import numpy.matlib
import scipy as sp
import scipy.sparse
from tqdm import tqdm

class NeuronBath:
    DT = 0.1
    def __init__(self,N=1000, p=0.1, g=1.5, alpha=1.0,dt=0.1):
        self.N = N
        self.p = p
        self.g = g
        self.alpha = alpha
        self.DT = dt

        self.nRec2Out = N

        scale = 1.0/np.sqrt(self.p*self.N)
        self.M = sp.sparse.random(N,N,p)*g*scale.toarray()
        self.P = (1.0/self.alpha)*np.eye(N)
        self.wf = 2.0*(np.random.uniform(0,1,N)-0.5)
        self.wo = np.zeros(N)
        self.dw = np.zeros(N)

        self.x = 0.5*np.random.randn(N)
        self.r = np.tanh(self.x)
        self.z = 0.5*np.random.randn()

    def step(self, dt=None):
        if dt is None:
            dt = self.DT
        self.x = (1.0-dt)*self.x + np.matmul(self.M,(self.r*dt)) + self.wf*(self.z*dt)
        self.r = np.tanh(self.x)
        self.z = np.matmul(self.wo.T,self.r)

    def learn(self):
        self.k = np.matmul(self.P,self.r)
        self.rPr = np.matmul(self.r.T,self.k)
        self.c = 1.0 / (1.0 + self.rPr)
        self.P = self.P - np.matmul(self.k,(self.k.T*self.c))

    def update_wo(self,error):
        self.dw = -error*self.k*self.c
        self.wo = self.wo + self.dw


def gen_ft(simtime):
    amp = 1.3
    freq = 1/60
    sin_wave = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
    return sin_wave/1.5


if __name__=='__main__':
    N = 1000
    nsecs = 1440
    dt = 0.1
    learn_every = 2

    simtime = np.arange(nsecs/dt) * dt
    simtime2 = simtime + nsecs

    ft = gen_ft(simtime)
    ft2 = gen_ft(simtime2)

    e = 0
    zt = []
    zpt = []

    bath = NeuronBath(N)
    import pdb; pdb.set_trace()

    for idx,t in tqdm(enumerate(simtime)):
        bath.step()

        if idx % learn_every == 0:
            # Update inverse correlation matrix
            bath.learn()
            e = bath.z - ft[idx]
            bath.update_wo()
        zt.extend([bath.z])
    zt = np.array(zt)
    error_avg = abs(zt-ft).sum()/len(simtime)

    for idx,t in tqdm(enumerate(simtime2)):
        bath.step()
        zpt.extend([bath.z])
    zpt = np.array(zpt)
    test_error_avg = abs(zpt-ft2).sum()/len(simtime2)


