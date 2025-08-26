import pandas as pd


def get_data():
    years = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='Sets', usecols='A')
    d_years  = list(years.Years)[:9]

    counties = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='Sets', usecols='C')
    d_counties = list(counties.Counties)

    sites = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='Sets', usecols='E')
    d_sites = list(sites.Sites)[:26]

    producers = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='Sets', usecols='I')
    d_producers = list(producers.Conventional_Production_Sites)[:10]

    distr_centers = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='Sets', usecols='J')
    d_dist_centers = list(distr_centers.Distribution_Sites)[:3]

    demand = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='demand', usecols='A:B')
    d_demand = {(c,k):0 for c in d_counties for k in d_years}

    for k in demand.iterrows():
        d_demand[k[1][0], 'k24'] = k[1][1]

    for c in d_counties:
        loc_ = 0
        for k in d_years:
            if k!='k24':
                loc_prev = loc_
                k_prev = d_years[loc_]
                d_demand[c,k] = d_demand[c,k_prev]*(1+0.5/100)
                loc_+=1
    #
    #  distribution cost from d->c
    # distribution cost
    tau_dc = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='distributionCost', usecols='A:D')

    d_tau_dc = { (d,c):0 for d in d_dist_centers for c in d_counties}
    for k in tau_dc.iterrows():
        k_ = k[0]
        d_tau_dc['Glenwood', tau_dc.counties[k_]] = tau_dc.glenwood[k_]
        d_tau_dc['Glenwood', tau_dc.counties[k_]] = tau_dc.glenwood[k_]
        d_tau_dc['Glenwood', tau_dc.counties[k_]] = tau_dc.glenwood[k_]

        d_tau_dc['Mankato', tau_dc.counties[k_]] = tau_dc.mankato[k_]
        d_tau_dc['Mankato', tau_dc.counties[k_]] = tau_dc.mankato[k_]
        d_tau_dc['Mankato', tau_dc.counties[k_]] = tau_dc.mankato[k_]

        d_tau_dc['Rosemount', tau_dc.counties[k_]] = tau_dc.rosemount[k_]
        d_tau_dc['Rosemount', tau_dc.counties[k_]] = tau_dc.rosemount[k_]
        d_tau_dc['Rosemount', tau_dc.counties[k_]] = tau_dc.rosemount[k_]


    lambd_par  = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='conventionalProductionParameter', usecols='A:B')
    d_lambda_p = {p:0 for p in d_producers}
    for k in lambd_par.iterrows():
        if k[0]<=9:
            producer_ = lambd_par.p[k[0]]
            limit_ = lambd_par.amount[k[0]]
            
            d_lambda_p[producer_] = limit_

    tau_pd = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='conventionalProductionParameter', usecols='F:I')
    d_tau_pd = {(p,d):0 for p in d_producers for d in d_dist_centers}

    for k in tau_pd.iterrows():
        if k[0]<=9:
            producer_ = tau_pd.producer[k[0]]
            d_tau_pd[producer_, 'Mankato'] = tau_pd.mankato[k[0]]
            d_tau_pd[producer_, 'Glenwood'] = tau_pd.glenwood[k[0]]
            d_tau_pd[producer_, 'Rosemount'] = tau_pd.rosemount[k[0]]


    tau_rc = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='renewableTransportationCost')
    d_tau_rc = { (r,c):0 for r in d_sites for c in d_counties}

    for i in range(1,28):
        for j in range(1,83):
            county_ = tau_rc.iloc[j,0]
            cand_site_ = tau_rc.iloc[0,i]
            d_tau_rc[county_, cand_site_] = tau_rc.iloc[j,i]

    reg_1 = ['elkriver', 'avon', 'virginia','willmar']
    reg_2 = ['marshall','pipestone','tracy','woodstock', 'lakebenton']
    reg_3 = ['blueearth', 'fairmont', 'windom', 'winnebago']
    reg_4 = ['chandler', 'lakewilson', 'luverne', 'wilmont', 'worthington']
    reg_5 = ['moorhead', 'hewitt', 'hoffman', 'crookston']
    reg_6 = ['adams', 'dexter', 'elba', 'lewiston', 'northfield']


    map_sites_to_regs = {}
    for c in d_sites:
        if c in reg_1:
            map_sites_to_regs[c] = 1
        if c in reg_2:
            map_sites_to_regs[c] = 2
        if c in reg_3:
            map_sites_to_regs[c] = 3
        if c in reg_4:
            map_sites_to_regs[c] = 4
        if c in reg_5:
            map_sites_to_regs[c] = 5
        if c in reg_6:
            map_sites_to_regs[c] = 6

    prod_params = pd.read_excel('Ammonia_Supply_Chain_Transition_Parameters.xlsx', sheet_name='renewableProductionParameters')


    # sigma_rk  1->9 3|8

    d_sigma = {}

    for i in range(1,10):
        for c in d_sites:    
            d_sigma[ c, prod_params.iloc[0,i]] = prod_params.iloc[map_sites_to_regs[c],i]

    d_gamma = {}
    for i in range(1,10):
        for c in d_sites:
            d_gamma[c, prod_params.iloc[0,i]] = prod_params.iloc[map_sites_to_regs[c]+9,i]

    d_zeta = {}
    for i in range(1,10):
        for c in d_sites:
            d_zeta[c, prod_params.iloc[0,i]] = prod_params.iloc[map_sites_to_regs[c]+18,i]
            
            
    d_omega = {}

    for i in range(1,10):
        for c in d_sites:    
            d_omega[ c, prod_params.iloc[0,i]] = prod_params.iloc[map_sites_to_regs[c],i+11]

    d_xi = {}
    for i in range(1,10):
        for c in d_sites:
            d_xi[c, prod_params.iloc[0,i]] = prod_params.iloc[map_sites_to_regs[c]+9,i+11]

    d_Xi = {}
    for i in range(1,10):
        d_Xi[prod_params.iloc[0,i]] = prod_params.iloc[17+i,12]



    return d_years,d_counties,d_sites,d_producers, d_dist_centers, d_demand,d_tau_dc,d_lambda_p, d_tau_pd,d_tau_rc, d_sigma, d_gamma, d_zeta,  d_omega, d_xi, d_Xi