import numpy as np
import scipy as sp
import scipy.stats as stats
import numpy.matlib
import scipy.sparse
import json
from tqdm import tqdm_notebook

class NeuralNetwork:
    DT = 0.1
    def from_dict(self, state_dict):
        for k,v in state_dict.iteritems():
            if isinstance(v,list):
                v = np.array(v)
            setattr(self,k,v)

    def __init__(self,N=1000, pz=1, pg=0.1, g=1.5, alpha=1,dt=0.1,num_fits=1,num_inputs=0,state=None):
        if state is not None:
            self.from_dict(state)
        else:
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
            self.woc = np.zeros((N,1))
            self.wfc = np.random.uniform(-1,1,(N,1))

            self.wu = np.random.uniform(-1,1,(N,num_inputs))

            self.x = np.expand_dims(0.5*np.random.randn(N),1)
            self.r = np.tanh(self.x)
            self.z = np.expand_dims(0.5*np.random.randn(num_fits),1)
            self.z_ctl = np.expand_dims(0.5*np.random.randn(1),1)

    def step(self, dt=None,ctl=False):
        if dt is None:
            dt = self.DT
        self.x = (1.0-dt)*self.x + np.matmul(self.M,(self.r*dt)) + np.matmul(self.wf,(self.z*dt))
        if ctl:
            self.x = self.x + np.matmul(self.wfc,(self.z_ctl*dt))
        self.r = np.tanh(self.x)
        self.z = np.dot(self.wo.T,self.r)
        self.z_ctl = np.matmul(self.woc.T,self.r)

    def learn(self):
        self.k = np.dot(self.P,self.r)
        self.rPr = np.dot(self.r.T,self.k)
        self.c = (1.0 / (1.0 + self.rPr))
        self.P = self.P - np.dot(self.k,(self.k.T*self.c))

    def update_wo(self,error):
        self.dw = -error*self.k*self.c
        self.wo = self.wo + self.dw

    def update_woc(self):
        e = np.squeeze(self.z_ctl) - 1
        dwoc = -e*self.k*self.c
        self.woc = self.woc + dwoc

    def to_json(self):
        out = {}
        for k,v in vars(self).iteritems():
            if isinstance(v,np.ndarray):
                out[k]=v.tolist()
            else:
                out[k]=v
        return json.dumps(out,sort_keys=True, indent=4)

class Simulation:
    def pretrain(self,ft,init_steps=100):
        print('pretraining')
        pretrain_idx = 0
        self.ft = ft
        while pretrain_idx < init_steps:
            self.network.step(ctl=True)
            if pretrain_idx % 2 == 1:
                self.network.learn()
                e = self.network.z.T - self.ft[:,0]
                self.network.update_wo(e)
                self.network.update_woc()
            pretrain_idx += 1
            self.pretrained = True

    def __init__(self,network,ft=None,dt=0.1):
        self.network = network
        self.dt = dt
        self.ft = ft
        self.idx = 0
        self.pretrained = False

    def iter(self,steps,ft,init_steps=2,learn_every=2,train=True,ctl=False):
        self.idx, self.steps = 0,steps
        self.ft = ft
        #if self.pretrained == False:
            #self.pretrain(ft=self.ft)
        # initialize network
        #jwhile self.idx < init_steps:
            #self.network.step(ctl=True)
            #yield self.idx
            #self.idx += 1

        while self.idx < self.steps:
            self.network.step(ctl=ctl)
            if self.idx % learn_every == 1 and train:
                self.network.learn()
                e = (self.network.z.T - self.ft[:,self.idx])/self.network.num_fits
                self.network.update_wo(e)
                if ctl:
                    self.network.update_woc()
            yield self.idx
            self.idx += 1

    def gen_ft(self,steps=10000,amp=1.3,freq=1.0/60,to=0):
        self.timeline = np.arange(steps) * self.dt
        self.amp = amp
        self.freq = freq
        sin_wave = np.empty((len(self.timeline),self.network.num_fits))
        for i in np.arange(self.network.num_fits):
            sin_wave[:,i] = (self.amp/1.0)*np.sin(1.0*np.pi*self.freq*self.timeline) + (self.amp/2.0)*np.sin(2.0*np.pi*self.freq*self.timeline) + (self.amp/6.0)*np.sin(3.0*np.pi*self.freq*self.timeline) + (self.amp/3.0)*np.sin(4.0*np.pi*self.freq*self.timeline)

        self.ft = sin_wave.swapaxes(0,1)/1.5
        return self.ft

    def run(self,msg,learn_every=2,train=True,init=True,t_begin=0,t_end=None):
        if self.ft is None:
            self.gen_ft(self.amp,self.freq,t_begin)
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


