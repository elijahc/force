import numpy as np
import scipy as sp
import scipy.stats as stats
import numpy.matlib
import scipy.sparse
from tqdm import tqdm_notebook

class NeuralNetwork:
    DT = 0.1
    def __init__(self,N=1000, pz=1, pg=0.1, g=1.5, alpha=1,dt=0.1,num_fits=1):
        self.N = N
        self.pg = pg
        self.pz = pz
        self.g = g
        self.alpha = alpha
        self.DT = dt
        self.num_fits = num_fits

        scale = 1.0/np.sqrt(self.pg*self.N)
        M_rvs = stats.norm(loc=0,scale=scale).rvs
        self.M = sp.sparse.random(N,N,pg,data_rvs=M_rvs)*g
        self.M = self.M.toarray()
        self.P = (1.0/self.alpha)*np.identity(N)
        self.wf = np.random.uniform(-1,1,(N,num_fits))
        #self.wo = np.expand_dims(stats.norm(loc=0,scale=(1.0/np.sqrt(N))).rvs(N),num_fits)
        self.wo = np.zeros((N,num_fits))
        self.dw = np.zeros((N,num_fits))

        self.x = np.expand_dims(0.5*np.random.randn(N),1)
        self.r = np.tanh(self.x)
        self.z = np.expand_dims(0.5*np.random.randn(num_fits),1)

    def step(self, dt=None,feedback=True):
        if dt is None:
            dt = self.DT
        self.x = (1.0-dt)*self.x + np.matmul(self.M,(self.r*dt))
        if feedback:
            self.x = self.x + np.matmul(self.wf,(self.z*dt))
        self.r = np.tanh(self.x)
        self.z = np.dot(self.wo.T,self.r)

    def learn(self):
        self.k = np.dot(self.P,self.r)
        self.rPr = np.dot(self.r.T,self.k)
        self.c = (1.0 / (1.0 + self.rPr))
        self.P = self.P - np.dot(self.k,(self.k.T*self.c))

    def update_wo(self,error):
        self.dw = -error*self.k*self.c
        self.wo = self.wo + self.dw

class Simulation:
    def __init__(self,network,dt=0.1,nsecs=1440):
        self.network = network
        self.dt = dt
        self.ft = None
        self.nsecs = nsecs
        self.timeline = np.arange(self.nsecs/dt) * dt
        self.amp = 1.3
        self.freq = 1/60

    def gen_timeline(self,to=0):
        self.timeline = np.arange(self.nsecs/self.dt)*self.dt + to

    def gen_ft(self,num=1,amp=1.3,freq=1/60,to=0):
        self.amp = amp
        self.freq = freq
        self.gen_timeline(to)
        sin_wave = np.empty((len(self.timeline),num))
        for i in np.arange(num):
            sin_wave[:,i] = (self.amp/1.0)*np.sin(1.0*np.pi*self.freq*self.timeline) + (self.amp/2.0)*np.sin(2.0*np.pi*self.freq*self.timeline) + (self.amp/6.0)*np.sin(3.0*np.pi*self.freq*self.timeline) + (self.amp/3.0)*np.sin(4.0*np.pi*self.freq*self.timeline)

        self.ft = sin_wave/1.5
        return self.ft

    def run(self,msg,learn_every=2,train=True,fb=True,t_begin=0,t_end=None):
        if self.ft is None:
            self.gen_ft(self.network.num_fits,self.amp,self.freq,t_begin)
        if t_end is None:
            t_end = len(self.timeline)
        self.zt = []
        #self.wot = []
        idxs = np.arange(t_begin,t_end)
        for idx in tqdm_notebook(idxs,desc=msg,total=len(idxs)):
            self.network.step(feedback=fb)
            if idx % learn_every == 1 and train:
                # Update weights
                self.network.learn()
                e = self.network.z.T - self.ft[idx]
                self.network.update_wo(e)
            self.zt.append(self.network.z)
            #self.wot.append(np.sqrt(np.matmul(self.network.wo.T,self.network.wo)))
        self.zt = np.squeeze(np.array(self.zt))
        #self.wot = np.array(self.wot)

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

    error_avg = abs(zt-ft).sum()/len(simtime)

    for idx,t in enumerate(simtime2):
        bath.step()
        zpt.extend([bath.z])
    zpt = np.array(zpt)
    test_error_avg = abs(zpt-ft2).sum()/len(simtime2)


