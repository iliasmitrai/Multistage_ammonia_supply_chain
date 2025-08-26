from pyomo.environ import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

from pyomo.environ import *
from get_data import get_data

# import data
d_years,d_counties,d_sites,d_producers, d_dist_centers, d_demand,d_tau_dc,d_lambda_p, d_tau_pd,d_tau_rc, d_sigma, d_gamma, d_zeta,  d_omega, d_xi, d_Xi = get_data()
# define model
m = ConcreteModel()

m.Nt = Param(default = len(d_years)) # number of time periods
N_nodes = 61

print('------------------------------------')
print(f'-------- Number of years:{m.Nt()}----')
print(f'---------Number of nodes:{2**m.Nt()-1}----')

# scenario tree graph and nodes
G  = nx.DiGraph()
n_ = [i for i in range(1,N_nodes+1)]

G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 4)
G.add_edge(2, 5)
G.add_edge(3, 6)
G.add_edge(4, 7)
G.add_edge(5, 8)
G.add_edge(5, 9)
G.add_edge(6, 10)
G.add_edge(6, 11)
G.add_edge(7, 12)
G.add_edge(7, 13)
G.add_edge(7, 12)
G.add_edge(7, 13)
G.add_edge(8, 14)
G.add_edge(9, 15)
G.add_edge(10, 16)
G.add_edge(11, 17)
G.add_edge(12, 18)
G.add_edge(13, 19)
G.add_edge(14, 20)
G.add_edge(15, 21)
G.add_edge(16, 22)
G.add_edge(17, 23)
G.add_edge(18, 24)
G.add_edge(19, 25)
G.add_edge(20, 26)
G.add_edge(20, 27)
G.add_edge(21, 28)
G.add_edge(21, 29)
G.add_edge(22, 30)
G.add_edge(22, 31)
G.add_edge(23, 32)
G.add_edge(23, 33)
G.add_edge(24, 34)
G.add_edge(24, 35)
G.add_edge(25, 36)
G.add_edge(25, 37)

G.add_edge(26, 38)
G.add_edge(38, 50)

G.add_edge(27, 39)
G.add_edge(39, 51)

G.add_edge(28, 40)
G.add_edge(40, 52)

G.add_edge(29, 41)
G.add_edge(41, 53)

G.add_edge(30, 42)
G.add_edge(42, 54)

G.add_edge(31, 43)
G.add_edge(43, 55)

G.add_edge(32, 44)
G.add_edge(44, 56)

G.add_edge(33, 45)
G.add_edge(45, 57)

G.add_edge(34, 46)
G.add_edge(46, 58)

G.add_edge(35, 47)
G.add_edge(47, 59)

G.add_edge(36, 48)
G.add_edge(48, 60)

G.add_edge(37, 49)
G.add_edge(49, 61)




m.C = Set(initialize = d_counties) # counties
m.P = Set(initialize = d_producers) # conventional producers
m.N = Set(initialize = [i for i in range(1,G.number_of_nodes()+1)]) # nodes in the scenario tree
m.R = Set(initialize = d_sites) # Candidate site
m.D = Set(initialize = d_dist_centers) # Distribution center


n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]

nodes_per_t = { t: [] for t in range(1,len(d_years)+1)}

nodes_per_t[1] = [1]
nodes_per_t[2] = np.arange(2,5)
nodes_per_t[3] = np.arange(5,8)
nodes_per_t[4] = np.arange(8,14)
nodes_per_t[5] = np.arange(14,20)
nodes_per_t[6] = np.arange(20,26)
nodes_per_t[7] = np.arange(26,38)
nodes_per_t[8] = np.arange(38,50)
nodes_per_t[9] = np.arange(50,62)


node_time = { n: 0 for n in m.N}

for n in m.N:
    for t_ in nodes_per_t.keys():
        if n in nodes_per_t[t_]:
            node_time[n] = t_

p_ = {}
a_ = {}
#t=1

import sys

a_[1] = int(sys.argv[1])
# t=2
a_[2] = a_[1]+50
a_[3] = a_[1]
a_[4] = a_[1]-50
#t=3
a_[5] = a_[2]+50
a_[6] = a_[1]
a_[7] = a_[4]-50
#t=4
a_[8] = a_[5]+50
a_[9] = a_[5]-50
a_[10] = a_[6]+50
a_[11] = a_[6]-50
a_[12] = a_[7]+50
a_[13] = a_[7]-50
#t=5
a_[14]=a_[8]+50
a_[15]=a_[9]-50
a_[16]=a_[10]+50
a_[17]=a_[11]-50
a_[18]=a_[12]+50
a_[19]=a_[13]-50
#t=6
a_[20]=a_[14]+50
a_[21]=a_[15]-50
a_[22]=a_[16]+50
a_[23]=a_[17]-50
a_[24]=a_[18]+50
a_[25]=a_[19]-50
#t=7
a_[26]=a_[20]+50
a_[27]=a_[20]-50

a_[28]=a_[21]+50
a_[29]=a_[21]-50

a_[30]=a_[22]+50
a_[31]=a_[22]-50

a_[32]=a_[23]+50
a_[33]=a_[23]-50

a_[34]=a_[24]+50
a_[35]=a_[24]-50

a_[36]=a_[25]+50
a_[37]=a_[25]-50
#t=8
a_[38]=a_[26]+50
a_[50]=a_[38]+50

a_[39]=a_[27]-50
a_[51]=a_[39]-50

a_[40] = a_[28]+50
a_[52] = a_[40] + 50

a_[41] = a_[29] - 50
a_[53] = a_[41] -50

a_[42]=a_[30]+50
a_[54]=a_[42]+50

a_[43]=a_[31]-50
a_[55]=a_[43]-50

a_[44]=a_[32]+50
a_[56]=a_[44]+50

a_[45]=max(100,a_[33]-50)
a_[57]=max(100,a_[45]-50)

a_[46] = max(100,a_[34]+50)
a_[58] = max(100,a_[46]+50)

a_[47] = max(100,a_[35]-50)
a_[59] = max(100,a_[47]-50)

a_[48] = max(100,a_[36]+50)
a_[60] = max(100,a_[48] + 50)

a_[49] = max(100,a_[37]-50)
a_[61] = max(100,a_[49] - 50)

p_ = {}
p_[1]=1
p_[2]=1/3
p_[3]=1/3
p_[4]=1/3
for n in m.N:
    if n>=5:
        t = node_time[n]
        if t!=4 and t!=7:
            p_[n] = p_[nx.shortest_path(G,source = 1, target = n)[-2]]
        if t==4 or t==7:
            p_[n] = p_[nx.shortest_path(G,source = 1, target = n)[-2]]*0.5


as_d = {}
for p in m.P:
    for n in m.N:
        as_d[p,n] = a_[n]*1e-3
m.alpha = Param(m.P,m.N, default = as_d)

map_ny_to_k = {1: 'k24', 2: 'k25',3:'k26', 4: 'k27', 5:'k28', 6:'k29', 7: 'k30', 8: 'k31', 9:'k32'}
d_demand_n = {(c,n): 0 for c in m.C for n in m.N}
for c in m.C:
    for n in m.N:
        d_demand_n[c,n] = d_demand[c, map_ny_to_k[node_time[n]] ]
# parameters
m.delta = Param(m.C, m.N, default = d_demand_n)
d_omega_n = { (r,n): d_omega[r,map_ny_to_k[node_time[n]] ] for r in m.R for n in m.N}
m.omega = Param(m.R, m.N, default = d_omega_n)
omega_ = {r:250 for r in m.R}
m.Omega = Param(m.R, default = omega_)
d_xi_n = { (r,n): d_xi[r,map_ny_to_k[node_time[n]] ] for r in m.R for n in m.N}
d_Xi_n = { n: d_Xi[map_ny_to_k[node_time[n]]] for n in m.N}

m.xi = Param(m.R,m.N, default = d_xi_n)
m.Xi = Param(m.N, default = d_Xi_n)
m.Lambda = Param(m.P, default =  d_lambda_p)


phi_k_t = {}
loc_ = 0
for k in range(1,len(d_years)+1):
       phi_k_t[k] = 1*(1-8.5/100)**loc_
       loc_+=1
phi_k_ = {n:phi_k_t[node_time[n]] for n in m.N}

m.phi = Param(m.N, default = phi_k_)
m.theta = Param(default = 10.23)

d_sigma_n = {(r,n): d_sigma[r, map_ny_to_k[node_time[n]]] for r in m.R for n in m.N}
m.sigma = Param(m.R,m.N,default=d_sigma_n)

d_gamma_n = {(r,n): d_gamma[r, map_ny_to_k[node_time[n]]] for r in m.R for n in m.N}
m.gamma = Param(m.R,m.N, default = d_gamma_n)
# # m.rp = Param(m.R,m.T, default= )
d_zeta_n = {(r,n): d_zeta[r, map_ny_to_k[node_time[n]]] for r in m.R for n in m.N}
m.zeta = Param(m.R,m.N, default = d_zeta_n)


import pandas as pd
tau_rc = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='renewableTransportationCost')

d_tau_rc = {}
for i in range(1,28):
    for j in range(1,83):
        county_ = tau_rc.iloc[j,0]
        cand_site_ = tau_rc.iloc[0,i]
        for r in m.R:
            if r==cand_site_:
                for c in m.C:
                    if c==county_:
                        d_tau_rc[ r,c] = tau_rc.iloc[j,i]


# transportation costs
m.tau_rc = Param(m.R,m.C, default = d_tau_rc)
m.tau_pd = Param(m.P,m.D,default = d_tau_pd)
m.tau_dc = Param(m.D,m.C, default = d_tau_dc )


m.p = Param(m.N, default = p_)
# # # variables
m.z = Var(m.R, m.N, within=Binary, initialize=1)
m.x = Var(m.R, m.N, bounds = (0,1e4), initialize=1e2)
m.y_pd = Var(m.P, m.D,m.N, bounds = (0,1e4))
m.y_dc = Var(m.D,m.C, m.N, bounds = (0,1e4))
m.y_rc = Var(m.R, m.C,m.N, bounds = (0,1e4))

# # constraint
m.sp_struct = Constraint(m.R,m.N, rule = lambda m,r,n: m.x[r,n]<=1e5*m.z[r,n])
m.lb_capacity = Constraint(m.R,m.N, rule = lambda m,r,n: m.x[r,n] >=50*m.z[r,n])


# demand satisfaction
# m.e = Var(m.C,m.N, bounds = (0,1e5))
m.dem_con = Constraint(m.C, m.N, rule = lambda m,c,n: sum( m.y_rc[r,c,n] for r in m.R) + sum( m.y_dc[d,c,n] for d in m.D) >= m.delta[c,n])

def am_prod_(m,r,n):
    if n>3:
        return sum(m.y_rc[r,c,n] for c in m.C) <=  sum(m.x[r,n_p] for n_p in nx.shortest_path(G, source=1, target=n)[:-2])
    else:
        return sum(m.y_rc[r,c,n] for c in m.C) <= 0
m.am_prod = Constraint(m.R, m.N, rule = am_prod_)


def prod_am_ub_(m,r,n):

    path_from_root_to_n = nx.shortest_path(G, source=1, target=n)
    if len(path_from_root_to_n)<=3:
        return Constraint.Skip
    else:
        return sum(m.x[r,n_p]*m.omega[r,n_p] for n_p in nx.shortest_path(G, source=1, target=n)[:-2])<=m.Omega[r]
m.prod_am_ub = Constraint(m.R, m.N, rule = prod_am_ub_)

m.elec_used = Constraint(m.N, rule = lambda m,n: sum(m.x[r,n]*m.xi[r,n] for r in m.R)<= m.Xi[n])
m.max_amount_bought = Constraint(m.P,m.N, rule = lambda m,p,n: sum(m.y_pd[p,d,n] for d in m.D)<= m.Lambda[p])
m.cum_amm_amount = Constraint(m.D,m.N, rule = lambda m,d,n: sum(m.y_pd[p,d,n] for p in m.P)>= sum(m.y_dc[d,c,n] for c in m.C))


# all ammonia is from renewables
def _final_t(m,n):
    n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    if n in n_leaf:
        return sum(m.y_pd[p,d,n] for p in m.P for d in m.D)==0
    else:
        return Constraint.Skip
if sys.argv[2]=='renewable':
    m.final_t_cons = Constraint(m.N, rule = _final_t)    


# if int(sys.argv[2])==2:

m.obj = Objective(expr = sum( m.phi[n]* m.p[n]* (sum( m.x[r,n_p]*m.sigma[r,n_p]+m.z[r,n_p]*m.gamma[r,n_p] for n_p in nx.shortest_path(G, source=1, target=n) for r in m.R)/m.theta \
                                                 + sum(m.x[r,n_p]*m.zeta[r,n_p] for n_p in nx.shortest_path(G, source=1, target=n) for r in m.R)\
                                                 + sum(sum(m.y_rc[r,c,n]*m.tau_rc[r,c] for c in m.C) for r in m.R)\
                                                 + sum(m.y_pd[p,d,n]*m.alpha[p,n] for d in m.D for p in m.P)\
                                                 + sum(m.y_pd[p,d,n]*m.tau_pd[p,d] for d in m.D for p in m.P)\
                                                 + sum(m.y_dc[d,c,n]*m.tau_dc[d,c] for d in m.D for c in m.C)) for n in m.N), sense = minimize)

print(f'Nvar: {len([k[index] for k in m.component_objects(Var) for index in k if k[index].is_binary()==True])}')

print(f'Nvar: {len([k[index] for k in m.component_objects(Var) for index in k])}')
print(f'Constraint: {len([k[index] for k in m.component_objects(Constraint) for index in k])}')


solver = SolverFactory('gurobi')
solver.options['MIPGap']= 1e-3
solver.solve(m,tee=True)

# plot
res_ = []

n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
res_ = {}

scens_ = {i : {} for i in range(1,len(n_leaf)+1)}
loc_ = 1
for n_ in n_leaf:
    # get path from root to n_
    path_root_to_n_leaf = nx.shortest_path(G,source = 1, target = n_)
    scens_[loc_] = path_root_to_n_leaf
    res_[loc_] = { node_time[ns_]:[] for ns_ in path_root_to_n_leaf}
    loc_+=1

loc_ = 1
for n_ in n_leaf:
    for ns_ in path_root_to_n_leaf:
        for r in m.R:
            if round(m.z[r, ns_](),1)==1:
                res_[loc_][node_time[ns_]].append(r)
    loc_+=1

# # res_ = []

n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
cap_ = []

scens_ = {i : {} for i in range(1,len(n_leaf)+1)}
loc_ = 1
for n_ in n_leaf:
    # get path from root to n_
    path_root_to_n_leaf = nx.shortest_path(G,source = 1, target = n_)
    scens_[loc_] = path_root_to_n_leaf
    res_[loc_] = { node_time[ns_]:[] for ns_ in path_root_to_n_leaf}
    loc_+=1

loc_ = 1
for n_ in n_leaf:
    for ns_ in path_root_to_n_leaf:
        cap_scen_ = 0
        for r in m.R:
            cap_scen_+= m.x[r,ns_]()

        res_[loc_][node_time[ns_]].append(cap_scen_)
    loc_+=1

n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
N_scenarios = len(n_leaf)
map_ny_to_k = {1: 'k24', 2: 'k25',3:'k26', 4: 'k27', 5:'k28', 6:'k29', 7: 'k30', 8: 'k31', 9:'k32'}

print(f' Scenario  -  Year  - Investment location - Capacity')
for n in range(len(n_leaf)):
    n_ = n_leaf[n]
    path_root_to_n_leaf = nx.shortest_path(G,source = 1, target = n_)
    for n_s in path_root_to_n_leaf:
        t_ = node_time[n_s]
        for r in m.R:
            if round(m.z[r,n_s](),1)==1:
                print(f' {n+1}  -  {map_ny_to_k[t_]} - {str(r)} - {m.x[r,n_s]()}  ')


nodes_to_t = {}

for i,j in nodes_per_t.items():
    for ii in j:
        nodes_to_t[ii] = map_ny_to_k[i]


res_ = {}

map_ny_to_k = {1: 'k24', 2: 'k25',3:'k26', 4: 'k27', 5:'k28', 6:'k29', 7: 'k30', 8: 'k31', 9:'k32'}

pr_ = {}
pb_ = {}

for sc_ in scens_.keys():
    res_[sc_] = {map_ny_to_k[k]: [] for k in range(1,10)}
    pr_[sc_] =  {map_ny_to_k[k]: [] for k in range(1,10)}
    pb_[sc_] =  {map_ny_to_k[k]: [] for k in range(1,10)}
    nodes_ = scens_[sc_]

    # loop over nodes
    for n_ in nodes_:
        # get time where node is
        t_n_ = nodes_to_t[n_]
        for r in m.R:
            if round(m.z[r,n_](),1)==1:
                res_[sc_][t_n_].append(r)
        
        pr_[sc_][t_n_].append(m.alpha['northdakota',n_])
        pb_[sc_][t_n_].append(m.p[n_])


import sys

data = {'scenario': [], 'Price': [], 'Year': [], 'Location': [], 'Capacity': []}
cost = {'Scenario': [], 'Year': [], 'Price': [], 'CAP': [], 'OP': [], 'DR': [], 'PC':[], 'TC':[], 'DC': []}

for n in range(len(n_leaf)):
    n_ = n_leaf[n]
    path_root_to_n_leaf = nx.shortest_path(G,source = 1, target = n_)
    for n_s in path_root_to_n_leaf:
        t_ = node_time[n_s]
        if round(sum(m.z[r,n_s]() for r in m.R),1)==0:
            data['scenario'].append(n+1)
            data['Year'].append(t_)
            data['Location'].append('None')
            data['Capacity'].append(0)
            data['Price'].append(a_[n_s])
        else:
            for r in m.R:
                if round(m.z[r,n_s](),1)==1:
                    data['scenario'].append(n+1)
                    data['Year'].append(t_)
                    data['Location'].append(r)
                    data['Capacity'].append(m.x[r,n_s]())
                    data['Price'].append(a_[n_s])


        cost['Scenario'].append(n+1)
        cost['Year'].append(t_)
        cost['CAP'].append(sum( m.x[r,n_s]()*m.sigma[r,n_s]+m.z[r,n_s]()*m.gamma[r,n_s] for r in m.R)/m.theta)
        cost['OP'].append(sum(m.x[r,n_p]()*m.zeta[r,n_p] for n_p in nx.shortest_path(G, source=1, target=n_s) for r in m.R))
        cost['DR'].append(sum(sum(m.y_rc[r,c,n_s]()*m.tau_rc[r,c] for c in m.C) for r in m.R))
        cost['PC'].append(sum(m.y_pd[p,d,n_s]()*m.alpha[p,n_s] for d in m.D for p in m.P))
        cost['TC'].append(sum(m.y_pd[p,d,n_s]()*m.tau_pd[p,d] for d in m.D for p in m.P))
        cost['DC'].append(sum(m.y_dc[d,c,n_s]()*m.tau_dc[d,c] for d in m.D for c in m.C))
        cost['Price'].append(a_[n_s])


    import csv

    # Specify the CSV file name
    # if sys.argv[2]=='renewable':
    #     csv_file = "results_renewable/investment_decisions_{}_no_mipgap.csv".format(sys.argv[1])
    # else:
    #     csv_file = "results/investment_decisions_{}_no_mipgap.csv".format(sys.argv[1])
    if sys.argv[2]=='renewable':
        csv_file = "results_aug_25_renewable/investment_decisions_{}_no_mipgap.csv".format(sys.argv[1])
    else:
        csv_file = "results_aug25/investment_decisions_{}_no_mipgap.csv".format(sys.argv[1])


    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the header row (keys of the dictionary)
        writer.writerow(data.keys())

        # Write the data rows (values of the dictionary)
        writer.writerows(zip(*data.values()))


    import csv

    # Specify the CSV file name
    # if sys.argv[2]=='renewable':
    #     csv_file = "results_renewable/costs_{}.csv".format(sys.argv[1])
    # else:
    #     csv_file = "results/costs_{}.csv".format(sys.argv[1])
    if sys.argv[2]=='renewable':
        csv_file = "results_aug_25_renewable/costs_{}.csv".format(sys.argv[1])
    else:
        csv_file = "results_aug25/costs_{}.csv".format(sys.argv[1])

    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the header row (keys of the dictionary)
        writer.writerow(cost.keys())

        # Write the data rows (values of the dictionary)
        writer.writerows(zip(*cost.values()))

# # plot stuff

n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
loc = 1
for n_ in n_leaf:
    y = []
    pr = []

    X = []
    Y1 = []
    Y2 = []

    parent_node_ = nx.shortest_path(G,source = 1, target = n_)
    for i_ in parent_node_:
        y.append(node_time[i_])
        pr.append(a_[i_])
        X.append(node_time[i_])
        Y1.append(sum(m.y_pd[p,d,i_]() for d in m.D for p in m.P))
        Y2.append(sum(sum(m.y_rc[r,c,i_]() for c in m.C) for r in m.R))

    plt.figure()
    n = 2 # len(vals)2
    _X = np.arange(len(X))
    width = 0.8
    i = 0
    plt.bar(X, Y1, width=width/float(n), color = 'black', align="edge", bottom = 0)   
    plt.bar(X, Y2, width=width/float(n), color = 'gray', align="edge", bottom = Y1)   
    plt.ylabel('Amount of ammonia', fontsize=12)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    text_values = ['2024', '2025', '2026', '2027','2028','2029', '2030', '2031', '2032']
    plt.xticks(_X + 0.5*width/2. + i/float(n)*width, text_values)

    # plt.xticks(_X, X)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Amount of ammonia', fontsize=12)
    
    # if sys.argv[2]=='renewable':
    #     plt.savefig('results_renewable/{}/ammonia_purch_shiped_with_years_scenario_{}_price_{}_renew.png'.format(sys.argv[1],loc, sys.argv[1]), dpi = 600)    
    #     plt.savefig('results_renewable/{}/ammonia_purch_shiped_with_years_scenario_{}_price_{}_renew.eps'.format(sys.argv[1],loc, sys.argv[1]), format = 'eps')
    # else:
    #     plt.savefig('results/{}/ammonia_purch_shiped_with_years_scenario_{}_price_{}_econ.png'.format(sys.argv[1],loc, sys.argv[1]), dpi = 600)    
    #     plt.savefig('results/{}/ammonia_purch_shiped_with_years_scenario_{}_price_{}_econ.eps'.format(sys.argv[1],loc, sys.argv[1]), format = 'eps')
    # loc+=1

n_leaf = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
loc = 1
sol = {}
for n_ in n_leaf:
    y = []
    pr = []

    X = []
    Y1 = []
    Y2 = []

    parent_node_ = nx.shortest_path(G,source = 1, target = n_)
    for i_ in parent_node_:
        y.append(node_time[i_])
        pr.append(a_[i_])
        X.append(node_time[i_])
        Y1.append(sum(m.y_pd[p,d,i_]() for d in m.D for p in m.P))
        Y2.append(sum(sum(m.y_rc[r,c,i_]() for c in m.C) for r in m.R))

    sol[loc] = {'y': y, 'pr': pr, 'X':X, 'Y1': Y1, 'Y2': Y2}
    loc+=1

import matplotlib.pyplot as plt
import numpy as np

# Create figure and subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid
axes = axes.flatten()  # Flatten for easy iteration

loc = 1
x = [2024, 2025, 2026, 2027,2028,2029, 2030, 2031, 2032]
for i, ax in enumerate(axes):
    ax2 = ax.twinx()  # Create secondary y-axis
    
    # Generate different data for each subplot
    # x  = sol[loc]['y']
    y11 = sol[loc]['pr']
    y21 = sol[loc]['Y1']
    y22 = sol[loc]['Y2']
    
    # Plot on primary y-axis (left)
    ax.plot(x, y11, 'b-', label=f'Ammonia price')
    # ax.grid()
    
    # Plot on secondary y-axis (right)
    # ax2.plot(x, y21, 'r-.', label=f'Amount of Ammonia')
    ax2.plot(x, [y22[k]/(y22[k]+y21[k]) for k in range(len(y21))], 'k-', label=f'Amount of Ammonia')
    
    
    col = i % 4  # Determine the column index (0-3)
    row = i // 4  # Determine the row index (0-2)
    
    # Formatting based on column position
    if col == 0:  # First column (only left y-axis)
        ax.set_ylabel("Ammonia Price", color='b', fontsize=12)
        # ax2.set_yticklabels([])  # Hide right y-axis labels
        ax2.set_ylabel("")
    elif col == 3:  # Last column (only right y-axis)
        ax2.set_ylabel("Percentage of renewable Ammonia", color='k', fontsize=12)
        # ax.set_yticklabels([])  # Hide left y-axis labels
        ax.set_ylabel("")
    else:  # Middle columns (hide both y-axes)
        # ax.set_yticklabels([])
        # ax2.set_yticklabels([])
        ax.set_ylabel("")
        ax2.set_ylabel("")
    
    # X-axis label behavior
    if row == 2:  # Bottom row (subplots 9-12)
        ax.set_xlabel("Year", fontsize=12)  # Set 'Year' for each subplot in the last row
    else:  # Remove x-axis labels for top rows
        ax.set_xticklabels([])

    # Set y-axis colors where applicable
    ax.tick_params(axis='y', colors='b')
    ax2.tick_params(axis='y', colors='k')
    ax2.set_ylim(-0.05, 1.05)  # Set range from 0 to 1
    ax2.set_yticks(np.arange(0, 1.1, 0.1))  # Set ticks every 0.1

    ax.set_ylim(100, 1200)  # Set range from 0 to 1
    ax.set_yticks(np.arange(100, 1200, 100))  # Set ticks every 0.1

    # Set subplot title
    ax.set_title(f"Scenario {i+1}")
    loc+=1

plt.tight_layout()  # Adjust layout to prevent overlap
# if sys.argv[2]=='renewable':
#     plt.savefig('results_renewable/all_scenarios_{}_renewable.png'.format(sys.argv[1]), dpi=600)
#     plt.savefig('results_renewable/all_scenarios_{}_renewable.eps'.format(sys.argv[1]), format = 'eps')
# else:
#     plt.savefig('results/all_scenarios_{}_econ.png'.format(sys.argv[1]), dpi=600)
#     plt.savefig('results/all_scenarios_{}_econ.eps'.format(sys.argv[1]), format = 'eps')
if sys.argv[2]=='renewable':
    plt.savefig('results_aug_25_renewable/all_scenarios_{}_renewable.png'.format(sys.argv[1]), dpi=600)
    plt.savefig('results_aug_25_renewable/all_scenarios_{}_renewable.eps'.format(sys.argv[1]), format = 'eps')
else:
    plt.savefig('results_aug25/all_scenarios_{}_econ.png'.format(sys.argv[1]), dpi=600)
    plt.savefig('results_aug25/all_scenarios_{}_econ.eps'.format(sys.argv[1]), format = 'eps')
plt.show()