import datetime, time, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
import pandas as pd
from numba import jit
from IPython.display import clear_output
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('png', 'pdf')
plt.rcParams['figure.figsize'] = [8, 8]


def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)




class Culture():
    '''
    a_cnt ... počet agentů (v případě malého světa libovolný, 
                            v případě mřížky musí být X^2)
    c_net ... struktůra sítě (0..malý svět, 1..mřížka)
    f_cnt ... počet features
    t_cnt ... počet traits (jen jedno číslo)
    maxiter ... maximální počet iterací
    '''
    
    def __init__(self, 
                 a_cnt=20, 
                 c_net=0, 
                 f_cnt=5, 
                 t_cnt=5, 
                 maxiter=10000,
                 random_con=True):
        self.f_cnt = f_cnt
        self.t_cnt = t_cnt
        self.maxiter = maxiter
        self.usediter = maxiter
        self.a_cnt = a_cnt
        self.c_net = c_net
        self.random_con = random_con
        
        #self.sim_cnt = 0
        
        self.init_agents()
        self.init_network()
        self.init_global_similarity()
        self.init_convergence()
        
        self.simulated = False
        self.regions = 0
        self.non_zero_edges = 0

    def init_agents(self):
        '''
        Vytvoří seznam agentů a naplní ho náhodnými features/traits
        '''
        self.agents = np.random.randint(self.t_cnt, size=(self.a_cnt, self.f_cnt))
    
    def init_network(self):
        if self.c_net == 0:
            A_diags = np.ones([5,self.a_cnt])
            A_diags[2,:] = np.zeros(self.a_cnt)
            self.m_adj = sparse.spdiags(A_diags, [-2, -1, 0, 1, 2],\
                                        self.a_cnt, self.a_cnt, format = 'lil')
            
            self.m_adj[-2:, 0] = 1
            self.m_adj[-1, 1] = 1
            self.m_adj[0, -2:] = 1
            self.m_adj[1, -1] = 1
            
            ####### náhodná propojení:
            if self.random_con:
                rnd_con = int(self.a_cnt * 1.2)

                m_adj_dense = sparse.lil_matrix.toarray(self.m_adj)
                ai_idxs, aj_idxs = np.nonzero(m_adj_dense == 0)

                while rnd_con > 0:
                    rnd_idx = np.random.choice(len(ai_idxs))
                    ai = ai_idxs[rnd_idx]
                    aj = aj_idxs[rnd_idx]
                    self.m_adj[ai,aj] = self.m_adj[aj,ai] = 1
                    rnd_con -= 1
      
    
        elif self.c_net == 1:
            self.grid_size = int(np.round(np.sqrt(self.a_cnt)))
            A_diags = np.ones([5,self.a_cnt])
            A_diags[3,:] = np.array([1,]+(self.grid_size * ( (self.grid_size-1)*[1,]\
                                                            + [0,] ))[:-1])
            A_diags[1,:] = np.array((self.grid_size * ( (self.grid_size-1)*[1,]\
                                                       + [0,] )))
            A_diags[2,:] = np.zeros(self.a_cnt)
            self.m_adj = sparse.spdiags(A_diags, [-self.grid_size, -1, 0, 1, self.grid_size], self.a_cnt, self.a_cnt, format = 'lil')
           
    def calc_similarity(self, ai, aj):
        same_features = 0
        for i in range(self.f_cnt):
            if self.agents[ai, i] == self.agents[aj, i]:
                same_features += 1    

        return same_features/self.f_cnt        
    
    def init_global_similarity(self):
        self.m_gs = sparse.lil_matrix.copy(self.m_adj)
        for i in range(self.a_cnt):
            for j in range(i+1,self.a_cnt):
                if self.m_adj[i,j] == 1:
                    self.m_gs[i,j] = self.m_gs[j,i] = self.calc_similarity(i, j) 
        
    def init_convergence(self):
        self.m_conv = sparse.lil_matrix.copy(self.m_gs)
        self.m_conv[self.m_conv == 1] = 0
    
    def is_interact(self, ai, aj):
        las = self.m_conv[ai,aj] #local agent similarity

        if las == 0: 
            return False
        
        if np.random.uniform() <= las:
            return True
        else:
            return False
    
    def update_nbrs(self, ai, it_cnt):
        nbrs = sparse.lil_matrix.nonzero(self.m_adj[ai])[1]

        for n in nbrs:
            f_sim = self.calc_similarity(ai, n)
            self.m_gs[ai,n] = self.m_gs[n,ai] = f_sim

            if f_sim == 1:
                self.m_conv[ai,n] = self.m_conv[n,ai] = 0  
            else:
                self.m_conv[ai,n] = self.m_conv[n,ai] = f_sim
                
        if sparse.lil_matrix.count_nonzero(self.m_conv) == 0:
            return True
        return False
        
    def do_interact(self, ai, aj, it_cnt):
        '''
        Vrací zda model dokonvergoval nebo ne
        '''
        if self.is_interact(ai, aj):
            
            # rozdílné featury musí existovat, jinak by is_interact vrátilo false
            f_idx = np.random.choice(np.flatnonzero(self.agents[ai]-self.agents[aj]))
            self.agents[ai, f_idx] = self.agents[aj, f_idx]
            return self.update_nbrs(ai, it_cnt)
            
        return False
    
    def simulate(self, save_progress=0, legend=''):
        if self.simulated:
            print('Already simulated. Run analyze() or plot_XYZ() instead.')
        else:
            
                    
            for i in range(self.maxiter):
                if save_progress:
                    if i % save_progress == 0: 
                        self.plot_gsnet(to_file=True, file_name=f'sim{self.c_net}_{i}.png')
                    
                ai_idxs, aj_idxs = sparse.lil_matrix.nonzero(self.m_conv)
                ####### ošetřit případ, kdy len(ai_idxs) = 0. Může se to asi stát jn při inicializaci
                if len(ai_idxs) > 0:
                    rnd_idx = np.random.choice(len(ai_idxs))
                else: 
                    self.usediter = i+1
                    break

                ai = ai_idxs[rnd_idx]
                aj = aj_idxs[rnd_idx]

                if self.do_interact(ai, aj, i):
                    self.usediter = i+1
                    break
                    
            #self.sim_cnt += 1
            m_gs_array = sparse.lil_matrix.toarray(self.m_gs)
            G_comp = nx.from_numpy_array(m_gs_array)
            self.regions = nx.number_connected_components(G_comp)
            self.non_zero_edges = sparse.lil_matrix.count_nonzero(self.m_conv)
            if (save_progress) and (self.usediter < self.maxiter):
                self.plot_gsnet(to_file=True, file_name=f'sim{self.c_net}_final.png')
            self.simulated = True
            #if progress:
            #    update_progress(1)
            
            
    def analyze(self):
        
        # vypsat - počet agentů, počet kroků iterace, počet komponent
        print(f'Počet agentů: {self.a_cnt} (f:{self.f_cnt}, t:{self.t_cnt})')
        print(f'Počet všech propojení: {sparse.lil_matrix.count_nonzero(self.m_adj)}')
        print(f'Počet propojení která mohou interagovat (0 < similarity < 1): {sparse.lil_matrix.count_nonzero(self.m_conv)}')
        print(f'Počet komponent/kultur: {self.regions}\n')

        #print(f'Počet proběhlých simulací: {self.sim_cnt}')
        print(f'Maximální počet iterací: {self.maxiter}')
        if self.simulated:
            if self.usediter == self.maxiter:
                print(f'Model nekonvergoval v maximálním počtu iterací')
            else:
                print(f'Model konvergoval v {self.usediter} iteracích')            
        print('\n')
        
    
    def plot_m_gs(self):
        plt.imshow(sparse.lil_matrix.toarray(self.m_gs));
        plt.colorbar()
        plt.show()
    
    def plot_net(self):
        G = nx.from_numpy_array(sparse.lil_matrix.toarray(self.m_adj))
        plt.axis('off')
        plt.title('Graf sítě agentů')
        nx.draw_networkx(G)
        plt.show()
    
    
    def plot_gsnet(self, to_file=False, file_name=''):
        G = nx.from_numpy_array(sparse.lil_matrix.toarray(self.m_adj))
        W = nx.from_numpy_array(sparse.lil_matrix.toarray(self.m_gs), parallel_edges=False, create_using=G)
        edges,weights = zip(*nx.get_edge_attributes(W,'weight').items())
        edges = edges + ((0,0),)
        weights = weights + (0,)
        pos = nx.spring_layout(W)
        nx.draw(W, pos, node_color='b', edgelist=edges, edge_color=weights,\
                width=2.0, edge_cmap=plt.cm.Blues)
        if to_file:
            p = plt.savefig(file_name);
            plt.clf()
            plt.close()
        else:
            plt.title('Graf podobnosti agentů')
            plt.show()
        

        
        
class Simulation():
    '''
    a_cnt ... počet agentů, musí být list/pole s alespoň jednou hodnotou default=[25,]
    c_net ... struktůra sítě (0..malý svět, 1..mřížka)
    f_cnt ... počet features, musí být list/pole, default=[2,6]
    t_cnt ... počet traits, musí být list/pole, default=[2,6]
    sim_cnt ... počet opakování simulace pro jedno nastavení, default=5 
    maxiter ... maximální počet iterací na jednu simulaci, default=10000
    file ... soubor, do kterého se uloží výsldek default='culture_simulation.csv'
    '''
    def __init__(self,
                 a_cnt=[25,], 
                 c_net=0, 
                 f_cnt=[2,6], 
                 t_cnt=[2,6], 
                 sim_cnt=5, 
                 maxiter=10000, 
                 file='culture_simulation.csv'):
        self.a_cnt = a_cnt
        self.c_net = c_net 
        self.f_cnt = f_cnt
        self.t_cnt = t_cnt 
        self.sim_cnt = sim_cnt 
        self.maxiter = maxiter 
        self.file = file
        
        self.simulated = False
        
    def run_simulations(self, progress=True):
        '''
        Runs set simulation(s).
        
        progress ... má se v průběhu simulace zobrazovat progress bar?
        '''
        with open(self.file, mode='w') as csv_file:
            culture_simulation_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            culture_simulation_writer.writerow(['agents','features','traits','regions','usediter','non_zero_edges'])

            runs_cnt = 0
            runs_total = len(self.a_cnt)*len(self.f_cnt)*len(self.t_cnt)*self.sim_cnt
            if progress:
                update_progress(0)

            for a in self.a_cnt:
                for f in self.f_cnt:
                    for t in self.t_cnt:
                        for i in range(self.sim_cnt):
                            c1 = Culture(a, 0, f, t, self.maxiter)
                            c1.simulate()
                            #with open(self.file, mode='a') as csv_file:
                            culture_simulation_writer = csv.writer(csv_file, delimiter=',',\
                                                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            culture_simulation_writer.writerow([a,f,t,c1.regions,c1.usediter,c1.non_zero_edges])
                            del c1 
                            runs_cnt += 1
                            if progress:
                                update_progress(runs_cnt/runs_total)

        self.simulated = True                    
        if progress:
            update_progress(1)
            
    def regions_table(self, a_cnt=25):
        '''
        Vygeneruje tabulku se závislostí průměrného počtu regionů na počtu features a traits
        
        a_cnt ... pro jaký počet agentů se má tabulka vygenerovat?
        '''
        sim_df = pd.read_csv(self.file)
        f = sim_df[sim_df['agents']==a_cnt]
        f = f[['features','traits','regions']]
        f = f.groupby(['features','traits']).mean()
        f = f.unstack('traits')
        #f = f.style.set_properties(**{'width': '150px'})
        display(f)
            
            
    def usediter_table(self, a_cnt=25):
        '''
        Vygeneruje tabulku se závislostí počtu využitých iterací na počtu features a traits
        
        a_cnt ... pro jaký počet agentů se má tabulka vygenerovat?
        '''
        sim_df = pd.read_csv(self.file)
        f = sim_df[sim_df['agents']==a_cnt]
        f = f[['features','traits','usediter']]
        f = f.groupby(['features','traits']).mean()
        f = f.unstack('traits')
        #f = f.style.set_properties(**{'width': '150px'})
        display(f)
        
    def agents_regs_plot(self, f_cnt=6, t_cnt=10):
        '''
        Vykreslí graf závislosti průměrného počtu regionů na počtu agentů
        '''
        sim_df = pd.read_csv(self.file)
        f = sim_df[(sim_df['traits']==t_cnt) & (sim_df['features']==f_cnt)]
        f.groupby('agents')['regions'].mean().plot(figsize=(10,5))
        
    def traits_regs_plot(self, a_cnt=25, f_cnt=6):
        '''
        Vykreslí graf závislosti počtu regionů na počtu traits
        '''
        sim_df = pd.read_csv(self.file)
        f = sim_df[(sim_df['agents']==a_cnt) & (sim_df['features']==f_cnt)]
        f.plot(kind='scatter', x='traits', y='regions', logx=True, figsize=(10,5))
    
            
            
            
            
            
    
     