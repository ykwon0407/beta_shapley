import numpy as np
import pandas as pd
import pickle

def make_balance_sample(data, target):
    p = np.mean(target)
    if p < 0.5:
        minor_class=1
    else:
        minor_class=0
    
    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class=len(index_minor_class)
    n_major_class=len(target)-n_minor_class
    new_minor=np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)

    data=np.concatenate([data, data[new_minor]])
    target=np.concatenate([target, target[new_minor]])
    return data, target

def load_data(problem, dataset, **dargs):
    '''
    (X,y): data to be valued
    (X_val, y_val): data to be used for evaluation
    (X_test, y_test): data to be used for point removal/addition experiments (same dist with validation)
    '''
    if problem=='classification':
        n_data_to_be_valued=dargs.get('n_data_to_be_valued', 200)
        n_val=dargs.get('n_val', 200)
        n_test=dargs.get('n_test', 1000)
        rid=dargs.get('rid', 1) # this is used as an identifier for cifar100_test dataset
        (X, y), (X_val, y_val), (X_test, y_test)=load_classification_dataset(n_data_to_be_valued=n_data_to_be_valued,
                                                                            n_val=n_val,
                                                                            n_test=n_test,
                                                                            rid=rid,
                                                                            dataset=dataset)
        # training is flipped
        flipped_index=np.random.choice(np.arange(n_data_to_be_valued), n_data_to_be_valued//10, replace=False) 
        y[flipped_index]=(1 - y[flipped_index])

        # validation is also flipped
        flipped_val_index=np.random.choice(np.arange(n_val), n_val//10, replace=False) 
        y_val[flipped_index]=(1 - y_val[flipped_val_index])
        return (X, y), (X_val, y_val), (X_test, y_test), flipped_index
    else:
        raise NotImplementedError('Check problem')


def load_classification_dataset(n_data_to_be_valued=200, 
                                 n_val=100,
                                 n_test=1000, 
                                 rid=1,
                                 dataset='gaussian',
                                 clf_path='clf_path'):
    '''
    This function loads classification (or density estimation) datasets for the point addition experiments.
    n_data_to_be_valued: The number of data points to be valued.
    n_val: the number of data points for evaluation of the utility function.
    n_test: the number of data points for evaluation of performances in point addition experiments.
    clf_path: path to classification datasets.

    You may need to prepare datasets first. Please run 'prep_non_reg_data.py' first.
    As for the datasets 'cifar10', 'fashion' and 'mnist', 
    we extract features from trained weights using 'torchvision.models.resnet18(pretrained=True)'.
    '''
    if dataset == 'gaussian':
        print('-'*50)
        print('GAUSSIAN-C')
        print('-'*50)
        n, input_dim=50000, 5
        data = np.random.normal(size=(n,input_dim))
        beta_true = np.array([2.0, 1.0, 0.0, 0.0, 0.0]).reshape(input_dim,1)
        p_true = np.exp(data.dot(beta_true))/(1.+np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'covertype':
        print('-'*50)
        print('Covertype')
        print('-'*50)
        # (581012, 54)
        from sklearn.datasets import fetch_covtype
        data, target=fetch_covtype(data_home=clf_path, return_X_y=True)
        target = ((target==1) + 0.0).astype(int)
        data, target=make_balance_sample(data, target)
    else:
        assert False, f"Check {dataset}"

    if dataset not in ['cifar100_test']:
        idxs=np.random.permutation(len(data))
        data, target=data[idxs], target[idxs]

    X=data[:n_data_to_be_valued]
    y=target[:n_data_to_be_valued]
    X_val=data[n_data_to_be_valued:(n_data_to_be_valued+n_val)]
    y_val=target[n_data_to_be_valued:(n_data_to_be_valued+n_val)]
    X_test=data[(n_data_to_be_valued+n_val):(n_data_to_be_valued+n_val+n_test)]
    y_test=target[(n_data_to_be_valued+n_val):(n_data_to_be_valued+n_val+n_test)]

    print(f'number of samples: {len(X)}')
    X_mean, X_std= np.mean(X, 0), np.std(X, 0)
    normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
    X, X_val, X_test = normalizer_fn(X), normalizer_fn(X_val), normalizer_fn(X_test)

    return (X, y), (X_val, y_val), (X_test, y_test)



