import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

#Plotting tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout,Bar

from force import NeuralNetwork
from force import Simulation

boc = BrainObservatoryCache(manifest_file='/home/elijahc/dev/tools/allen-assistant/boc/manifest.json')

exps = boc.get_ophys_experiments(
            imaging_depths=[175],
            targeted_structures=['VISp'],
            cre_lines=['Cux2-CreERT2'],
            stimuli=['natural_scenes'])

exp_id = exps[2]['id']
data_set = boc.get_ophys_experiment_data(exp_id)
cids = data_set.get_cell_specimen_ids()
num_cells=50
print('%d Cells in Experiment %d'%(len(cids),exp_id))
idxs = data_set.get_spontaneous_activity_stimulus_table()
#idxs = data_set.get_stimulus_table('natural_scenes')

#t,ftr = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=cids[0:100])
t,ftr = data_set.get_dff_traces(cell_specimen_ids=cids[0:num_cells])

t = t[idxs.start.item():idxs.end.item()]
ftr_crop = ftr[:,idxs.start.item():idxs.end.item()]+1

delta_ts = t - np.roll(t,1)
samp_rate = stats.mode(delta_ts)[0]
dt = round(np.around(samp_rate/0.102,5),5)
net = NeuralNetwork(N=100,dt=dt,num_fits=num_cells)
sim = Simulation(network=net,dt=dt)
zt = []
rPrt = []
dwt = []
#pretrain_steps= 600
train_steps = 7000
test_steps = 1000
#ft0 = np.repeat(ftr_crop[:,:1],pretrain_steps,axis=1)
#ft_cat = np.concatenate([ft0,ftr_crop],axis=1)

for i in tqdm(
            sim.iter(steps=train_steps,ft=ftr_crop,train=True),
            total=train_steps,desc='Training'):
    zt.append(sim.network.z)
    if i > 1:
        rPrt.append(sim.network.rPr)
        dwt.append(np.power(sim.network.dw,2).sum())
