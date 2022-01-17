import os, sys, warnings, inspect, pickle
import numpy as np
import pandas as pd
from glob import glob
import _pickle as pkl
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.linalg import eigh
from scipy.stats import kendalltau, ttest_ind
from sklearn.cluster import KMeans

def return_model(mode, **kwargs):
    '''
    Define a model to be used in computation of data values
    '''
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode=='logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', -1)
        C = kwargs.get('C', 0.05) # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C,
                                   max_iter=max_iter, random_state=666)
    elif mode=='linear':
        n_jobs = kwargs.get('n_jobs', -1)
        model = LinearRegression(n_jobs=n_jobs)
    elif mode=='ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    elif mode=='Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode=='RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='SVC':
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 0.05) # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = SVC(kernel=kernel, max_iter=max_iter, C=C, random_state=666)
    elif mode=='LinearSVC':
        C = kwargs.get('C', 0.05) # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = LinearSVC(loss='hinge', max_iter=max_iter, C=C, random_state=666)
    elif mode=='GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode=='KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        n_jobs = kwargs.get('n_jobs', -1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    elif mode=='NB':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid mode!")
    return model

def beta_constant(a, b):
    '''
    the second argument (b; beta) should be integer in this function
    '''
    beta_fct_value=1/a
    for i in range(1,b):
        beta_fct_value=beta_fct_value*(i/(a+i))
    return beta_fct_value

def compute_weight_list(m, alpha=1, beta=1):
    '''
    Given a prior distribution (beta distribution (alpha,beta))
    beta_constant(j+1, m-j) = j! (m-j-1)! / (m-1)! / m # which is exactly the Shapley weights.

    # weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j)*binom{n-1}{j} in the paper).
    '''
    weight_list=np.zeros(m)
    normalizing_constant=1/beta_constant(alpha, beta)
    for j in np.arange(m):
        # when the cardinality of random sets is j
        weight_list[j]=beta_constant(j+alpha, m-j+beta-1)/beta_constant(j+1, m-j)
        weight_list[j]=normalizing_constant*weight_list[j] # we need this '/m' but omit for stability # normalizing
    return weight_list

def check_convergence(mem):
    """
    Compute Gelman-Rubin statistic
    Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)
    """
    if len(mem) < 1000:
        return 100
    n_chains=10
    (N,n_to_be_valued)=mem.shape
    if (N % n_chains) == 0:
        n_MC_sample=N//n_chains
        offset=0
    else:
        n_MC_sample=N//n_chains
        offset=(N%n_chains)
    mem=mem[offset:]
    percent=25
    while True:
        IQR_contstant=np.percentile(mem.reshape(-1), 50+percent) - np.percentile(mem.reshape(-1), 50-percent)
        if IQR_contstant == 0:
            percent += 10
            if percent == 105:
                assert False, 'CHECK!!! IQR is zero!!!'
        else:
            break

    mem_tmp=mem.reshape(n_chains, n_MC_sample, n_to_be_valued)
    GR_list=[]
    for j in range(n_to_be_valued):
        mem_tmp_j_original=mem_tmp[:,:,j].T # now we have (n_MC_sample, n_chains)
        mem_tmp_j=mem_tmp_j_original/IQR_contstant
        mem_tmp_j_mean=np.mean(mem_tmp_j, axis=0)
        s_term=np.sum((mem_tmp_j-mem_tmp_j_mean)**2)/(n_chains*(n_MC_sample-1)) # + 1e-16 this could lead to wrong estimator
        
        mu_hat_j=np.mean(mem_tmp_j)
        B_term=n_MC_sample*np.sum((mem_tmp_j_mean-mu_hat_j)**2)/(n_chains-1)
        
        GR_stat=np.sqrt((n_MC_sample-1)/n_MC_sample + B_term/(s_term*n_MC_sample))
        GR_list.append(GR_stat)
    GR_stat=np.max(GR_list)
    print(f'Total number of random sets: {len(mem)}, GR_stat: {GR_stat}', flush=True)
    return GR_stat    

def check_convergence_univariate(mem, n_chains=10):
    """
    Compute Gelman-Rubin statistic
    Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)
    """
    mem=np.array(mem).reshape(-1)
    if len(mem) < 1000:
        return 100

    N=len(mem)
    if (N % n_chains) == 0:
        n_MC_sample=N//n_chains
        offset=0
    else:
        n_MC_sample=N//n_chains
        offset=(N%n_chains)
    mem=mem[offset:]

    if np.std(mem.reshape(-1)) == 0:
        return 0

    percent=25
    while True:
        IQR_contstant=np.percentile(mem.reshape(-1), 50+percent) - np.percentile(mem.reshape(-1), 50-percent)
        if IQR_contstant == 0:
            percent += 10
            if percent == 105:
                assert False, 'CHECK!!! IQR is zero!!!'
        else:
            break
    
    mem_tmp_j_original=mem.reshape(n_chains, n_MC_sample)
    mem_tmp=mem_tmp_j_original/IQR_contstant # np.median(mem_tmp_j_original)
    mem_tmp_j=mem_tmp.T # now we have (n_MC_sample, n_chains)
    mem_tmp_j_mean=np.mean(mem_tmp_j, axis=0)
    s_term=np.sum((mem_tmp_j-mem_tmp_j_mean)**2)/(n_chains*(n_MC_sample-1)) 

    mu_hat_j=np.mean(mem_tmp_j)
    B_term=n_MC_sample*np.sum((mem_tmp_j_mean-mu_hat_j)**2)/(n_chains-1)
    
    GR_stat=np.sqrt((n_MC_sample-1)/n_MC_sample + B_term/(s_term*n_MC_sample))
    print(f'Total number of random sets: {len(mem)}, GR_stat: {GR_stat}',flush=True)
    return GR_stat    

def point_removal_exp_core(shap_engine, order, X_test, y_test, n_minimum):
    # Data are removed one by one till n_minimum samples remained
    X_init, y_init=shap_engine.X, shap_engine.y
    shap_engine.model.fit(X_init, y_init) # performance with all data points
    vals=[shap_engine.value(X=X_test, y=y_test)]
    for top_k_sources in np.arange(shap_engine.n_sources-1, n_minimum, -1):
        if not top_k_sources:
            continue
        sub_index_list=[]
        for idx in order[:(top_k_sources+1)]:
            sub_index_list.append(shap_engine.sources[idx])
        sub_index_list=np.hstack(sub_index_list)
        if np.std(y_init[sub_index_list]) != 0:
            shap_engine.model.fit(X_init[sub_index_list], y_init[sub_index_list])
            vals.append(shap_engine.value(X=X_test, y=y_test))
        else:
            vals.append(shap_engine.random_score)
    return np.array(vals) 

def point_removal_exp(shap_engine, X_test, y_test):
    n_minimum=shap_engine.n_sources//2
    perf_func=lambda order: point_removal_exp_core(shap_engine, order, X_test, y_test, n_minimum)
    remove_dict=dict()
    
    # for each quantified value, we sort training points
    for method_name in list(shap_engine.results.keys()):
        # Remove least important data point first (value order: decreasing)
        remove_dict[method_name+'_small']=perf_func(np.argsort(shap_engine.results[method_name])[::-1])
    remove_dict['random']=np.mean([perf_func(np.random.permutation(shap_engine.n_sources)) for _ in range(10)], axis=0)

    return remove_dict

def point_addition_exp_core(shap_engine, order, X_test, y_test, n_minimum):
    # Data are added
    sub_index_list=[]
    for idx in range(n_minimum):
        sub_index_list.append(shap_engine.sources[idx])
    sub_index_list=np.hstack(sub_index_list)
    X_init, y_init=shap_engine.X[sub_index_list], shap_engine.y[sub_index_list]
    try:
        shap_engine.model.fit(X_init, y_init)
        val=shap_engine.value(X=X_test, y=y_test)
    except:
        val=shap_engine.random_score
    vals=[val]

    order = [j for j in order if j not in np.arange(n_minimum)]
    for top_k_sources in np.arange(shap_engine.n_sources-n_minimum):
        if not top_k_sources:
            continue
        sub_index_list=[]
        for idx in order[:(top_k_sources+1)]:
            sub_index_list.append(shap_engine.sources[idx])
        sub_index_list=np.hstack(sub_index_list)
        X_batch=np.concatenate([X_init, shap_engine.X[sub_index_list]])
        y_batch=np.concatenate([y_init, shap_engine.y[sub_index_list]])
        try:
            shap_engine.model.fit(X_batch, y_batch)
            val=shap_engine.value(X=X_test, y=y_test)
        except:
            val=shap_engine.random_score
        vals.append(val)
        if len(vals)==100:
            break
    return np.array(vals) 

def point_addition_exp(shap_engine, X_test, y_test):
    n_minimum=int(shap_engine.n_sources*0.05)
    perf_func=lambda order: point_addition_exp_core(shap_engine, order, X_test, y_test, n_minimum)
    add_dict=dict()

    # for each quantified value, we sort training points
    for method_name in list(shap_engine.results.keys()):
        # Add most important data point first (value order: decreasing)
        add_dict[method_name+'_large']=perf_func(np.argsort(shap_engine.results[method_name])[::-1])
    add_dict['random']=np.mean([perf_func(np.random.permutation(shap_engine.n_sources)) for _ in range(10)], axis=0)

    return add_dict

def subsampling_exp(shap_engine, X_test, y_test):
    '''
    Subsampling experiment
    '''
    wrm_dict=dict()
    for method_name in shap_engine.results.keys():
        value=shap_engine.results[method_name]
        
        threshold=0 # np.percentile(value, 10)
        weight_based_on_value=value*(value>threshold)+threshold*(value<=threshold)
        try:
            rnd_ind=np.random.choice(len(value), (len(value)//4), 
                                      replace=True, 
                                      p=weight_based_on_value/np.sum(weight_based_on_value))
            shap_engine.model.fit(shap_engine.X[rnd_ind], 
                                  shap_engine.y[rnd_ind], 
                                  sample_weight=1/weight_based_on_value[rnd_ind])
        except:
            rnd_ind=np.random.choice(len(value), (len(value)//4), replace=True)
            shap_engine.model.fit(shap_engine.X[rnd_ind], shap_engine.y[rnd_ind])
        wrm_dict[method_name]=shap_engine.model.score(X=X_test, y=y_test)
    
    rnd_ind=np.random.choice(len(value), (len(value)//4), replace=True)
    shap_engine.model.fit(shap_engine.X[rnd_ind], shap_engine.y[rnd_ind])
    wrm_dict['Uniform']=shap_engine.model.score(X=X_test, y=y_test)

    shap_engine.model.fit(shap_engine.X, shap_engine.y)
    wrm_dict['Full']=shap_engine.model.score(X=X_test, y=y_test)
    return wrm_dict

def compute_f1_score(list_a, list_b):
    '''
    Comput F1 score for noisy detection task
    list_a : true flipped data points
    list_b : predicted flipped data points
    '''
    n_a, n_b=len(list_a), len(list_b)
    
    # among A, how many B's are selected
    n_intersection=len(set(list_b).intersection(list_a))
    recall=n_intersection/(n_a+1e-16)
    # among B, how many A's are selected
    precision=n_intersection/(n_b+1e-16)
    
    if recall > 0 and precision > 0:
        f1_score=1/((1/recall + 1/precision)/2)
    else:
        f1_score=0.
    return recall, precision, f1_score

def noisy_detection_exp(shap_engine, noisy_index):
    '''
    Noisy point detection experiment
    '''
    noisy_dict=dict()
    for method_name in shap_engine.results.keys():
        value=shap_engine.results[method_name].reshape(-1,1)
        kmeans=KMeans(n_clusters=2, random_state=0).fit(value)
        threshold=np.min(kmeans.cluster_centers_)
        guess_index=np.where(value.reshape(-1) < threshold)[0]
        noisy_dict[method_name]=list(compute_f1_score(noisy_index, guess_index))
    return noisy_dict

def summary_experiments(shap_engine, noisy_index, X_test, y_test):
    exp_dict=dict()
    # Noisy point detection
    exp_dict['noisy']=noisy_detection_exp(shap_engine, noisy_index)
    # Weighted risk minimization experiments
    exp_dict['subsampling']=subsampling_exp(shap_engine, X_test, y_test)
    # Point removal/addition experiments
    exp_dict['point_removal']=point_removal_exp(shap_engine, X_test, y_test)
    exp_dict['point_addition']=point_addition_exp(shap_engine, X_test, y_test)

    return exp_dict
    

