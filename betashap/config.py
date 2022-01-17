import numpy as np
import copy

def generate_config(expno_name,
                     problem='classification', 
                     model_family='logistic', 
                     metric='accuracy',
                     dataset='covertype',
                     min_cardinality=1, 
                     is_noisy=True,
                     n_data_to_be_valued=200,
                     n_val=200,
                     n_test=1000,
                     n_episode=30):    
    assert problem in ['regression','classification'], 'Check problem'
    assert model_family in ['linear','logistic','RandomForest','GB','LinearSVC','KNN'], 'Check model_family'
    assert metric in ['r2','negative_mse','accuracy','f1','auc','likelihood','prediction'], 'Check metric'

    #EXP CONFIGS
    exp = dict()
    exp['expno']=expno_name
    exp['n_runs']=n_episode
    
    #RUN CONFIGS
    runs=[]
    run_temp = dict()
    run_temp['use-gpu']=False
    run_temp['runpath']='path' # please change this part
    run_temp['problem']=problem
    run_temp['model_family']=model_family
    run_temp['metric']=metric
    run_temp['dataset']=dataset
    run_temp['is_noisy']=is_noisy
    run_temp['weights_list']=[(-1,-1), (1, 16), (1, 4), (1,1), (4,1), (16, 1)]
    
    # general shapley parameters
    run_temp['min_cardinality']=min_cardinality
    for seed in range(n_episode):
        run = copy.deepcopy(run_temp) 
        run['seed'] = seed
        run['r_id'] = seed
        run['dargs'] = {'n_data_to_be_valued':n_data_to_be_valued, 
                        'n_val':n_val, 
                        'n_test':n_test,
                        'rid': seed}
        runs.append(run)
    return exp, runs 

'''
Classification
'''

n_episode=5 # you may want to change the number of independent runs here.
clf_metric='accuracy'
def config001CL():
    exp, runs=generate_config(expno_name='001CL', 
                               problem='classification', 
                               model_family='logistic', 
                               metric=clf_metric,
                               dataset='gaussian',
                               n_episode=n_episode)
    return exp, runs    

def config002CL():
    exp, runs=generate_config(expno_name='002CL', 
                               problem='classification', 
                               model_family='logistic', 
                               metric=clf_metric,
                               dataset='covertype',
                               n_episode=n_episode)
    return exp, runs    

def config003CL():
    exp, runs=generate_config(expno_name='003CL', 
                               problem='classification', 
                               model_family='LinearSVC', 
                               metric=clf_metric,
                               dataset='gaussian',
                               n_episode=n_episode)
    return exp, runs    

def config004CL():
    exp, runs=generate_config(expno_name='004CL', 
                               problem='classification', 
                               model_family='LinearSVC', 
                               metric=clf_metric,
                               dataset='covertype',
                               n_episode=n_episode)
    return exp, runs    




