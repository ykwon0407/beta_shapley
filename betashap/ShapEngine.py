import numpy as np
import os, sys, time, warnings
import pickle as pkl
from collections import defaultdict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.base import clone

# custom 
import utils
warnings.filterwarnings("ignore")

class ShapEngine(object):
    def __init__(self, X, y, X_val, y_val, 
                 problem='classification', model_family='logistic', metric='accuracy', 
                 min_cardinality=1, GR_threshold=1.0005, sources=None, seed=None,
                 max_iters=100, **kwargs):
        """
        Args:
            (X,y): (inputs,outputs) to be valued.
            (X_val,y_val): (inputs,outputs) to be used for utility evaluation.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            max_iters: maximum number of iterations (for a fixed cardinality)
            **kwargs: Arguments of the model
        """
        self.X,self.y=X, y
        self.X_val,self.y_val=X_val, y_val
        self.problem=problem
        self.model_family=model_family
        self.metric=metric
        self.min_cardinality=min_cardinality
        self.GR_threshold=GR_threshold
        self.seed=seed
        self.sources=sources
        self.max_iters=max_iters

        self._initialize_instance()
        self.model=utils.return_model(self.model_family, **kwargs)
        self.random_score=self.init_score()
       
    def _initialize_instance(self):
        if self.problem == 'regression':
            assert (self.min_cardinality is not None), 'Check min_cardinality'
            self.is_regression = True
            if self.metric not in ['r2','negative_mse']:
                assert False, 'Invalid metric for regression!'
        elif self.problem == 'classification':
            self.is_regression=False
            self.num_classes=len(set(self.y_val))
            if self.num_classes > 2:
                assert self.metric != 'f1', 'Invalid metric for multiclass!'
                assert self.metric != 'auc', 'Invalid metric for multiclass!'
            if self.metric not in ['accuracy','f1','auc','likelihood']:
                assert False, 'Invalid metric for classification!'
        else:
            raise NotImplementedError("Check problem")

        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialize sources
        self.n_points=len(self.X)
        if self.sources is None:
            print('Source is initialized. A unit of sample is one data point')
            self.sources={i: np.array([i]) for i in range(self.n_points)}
        else:
            print('Source is initialized. A unit of sample is given')
            for i in self.sources.keys():
                print(f'class {i}: {len(self.sources[i])} samples')
        self.n_sources=len(self.sources)
        
        # Create placeholder for results.
        self.results=defaultdict(list)
        self.time_results=defaultdict(list)
        self.GR_results=defaultdict(list)
        self.weights=defaultdict(list)
            
    def init_score(self):
        """
        Gives the utility of a random guessing model. (Best constant prediction)
        We suppose that the higher the score is, the better it is.
        """
        if self.problem == 'regression':
            if self.metric == 'r2':
                return 0.0
            elif self.metric == 'negative_mse':
                return -np.mean((self.y_val-np.mean(self.y_val))**2)
            else:
                raise NotImplementedError("Check metric")
        elif self.problem == 'classification':
            if self.metric == 'accuracy':
                hist = np.bincount(self.y_val)/len(self.y_val)
                return np.max(hist)
            elif self.metric == 'f1':
                rnd_f1s = []
                for _ in range(1000):
                    rnd_y = np.random.permutation(self.y_val)
                    rnd_f1s.append(f1_score(self.y_val, rnd_y))
                return np.mean(rnd_f1s)
            elif self.metric == 'auc':
                return 0.5
            elif self.metric == 'likelihood':
                hist = np.bincount(self.y_val)/len(self.y_val)
                return np.sum(hist*np.log(hist+1e-16))
            else:
                raise NotImplementedError("Check metric")
        else:
            raise NotImplementedError("Check problem")
        
    def value(self, X=None, y=None):
        """
        Computes the utility of the given model
        """
        if X is None:
            X = self.X_val
        if y is None:
            y = self.y_val
        
        if self.problem == 'regression':
            if self.metric == 'r2':
                # sklearn default
                return self.model.score(X, y)
            elif self.metric == 'negative_mse':
                y_pred = self.model.predict(X)
                return -np.mean((y-y_pred)**2)
            elif self.metric == 'prediction':
                # Output: n_val by 1 matrix
                return self.model.predict(X)
            else:
                raise NotImplementedError("Check metric")
        elif self.problem == 'classification':
            if self.metric == 'accuracy':
                # sklearn default
                return self.model.score(X, y)
            elif self.metric == 'f1':
                return f1_score(y, self.model.predict(X))
            elif self.metric == 'auc':
                probs = self.model.predict_proba(X)
                return roc_auc_score(y, probs[:,-1])
            elif self.metric == 'likelihood':
                probs = self.model.predict_proba(X)
                true_probs = probs[np.arange(len(y)), y]
                return np.mean(np.log(true_probs))
            elif self.metric == 'prediction':
                # Output: n_val by 1 matrix
                probs = self.model.predict_proba(X)
                true_probs = probs[np.arange(len(y)), 1]
                return true_probs
            else:
                raise NotImplementedError('Invalid metric!')
        else:
            raise NotImplementedError("Check problem")

    def run(self, lld_run=True, knn_run=True, weights_list=None):
        """
        Calculates data sources(points) values.
        Args:
            lld_run: If True, computes and saves log-likelihood difference scores.
            knn_run: If True, computes and saves KNN-Shapley.
            weights_list: Computes and saves Beta Shapley values.
        """
        
        if self.problem=='classification':
            if weights_list:
                alpha, beta=1, 1
                model_name=f'{beta},{alpha}'
                self.weight_list=utils.compute_weight_list(m=self.n_sources, 
                                                            alpha=alpha, 
                                                            beta=beta)
                self._calculate_marginal_contributions(model_name=model_name)

                for weight in weights_list:
                    alpha, beta=weight
                    if alpha > 0 and beta > 0:
                        model_name=f'Beta({beta},{alpha})'
                        self.weight_list=utils.compute_weight_list(m=self.n_sources, 
                                                                alpha=alpha, 
                                                                beta=beta)
                    else:
                        model_name=f'LOO-First'
                        self.weight_list=np.zeros(self.n_sources) 
                        self.weight_list[self.min_cardinality]=1
                    self._compute_beta_shap_from_MC(model_name=model_name)

        if lld_run is True and self.metric != 'prediction':
            self._calculate_lld()

        if knn_run is True:
            self._calculate_knn(n_neighbors=10)
    
    def _calculate_lld(self):
        """
        Calculate the log-likelihood difference. (Or Leave-one-out)
        For regression, we assume the homogeneous error setting
        (i.e. variance is same across data points)
        """
        print('Start: Log-likelihood difference score calculation!')
        time_init=time.time()
        self.model.fit(self.X, self.y)
        baseline_value=self.value()
        vals_lld=np.zeros(self.n_sources)
        for i in self.sources.keys():
            X_batch=np.delete(self.X, self.sources[i], axis=0)
            y_batch=np.delete(self.y, self.sources[i], axis=0)
            self.model.fit(X_batch, y_batch)
            removed_value = self.value()
            vals_lld[i]=(baseline_value - removed_value)
        self.results['LOO-Last']=vals_lld
        self.time_results['LOO-Last']=time.time()-time_init
        print('Done: Log-likelihood difference score calculation!',flush=True)

    def _calculate_knn(self, n_neighbors=10):
        """
        Calculate the KNN-SHAP.
        from https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py
        """
        print('Start: KNN-SHAP score calculation!')
        time_init=time.time()
        n_val=len(self.X_val)
        knn_mat=np.zeros((self.n_sources, n_val))
        for i, (X_val_sample, y_val_sample) in enumerate(zip(self.X_val, self.y_val)):
            diff=(self.X - X_val_sample).reshape(self.n_sources, -1)
            dist=np.einsum('ij, ij->i', diff, diff)
            idx=np.argsort(dist)
            ans=self.y[idx]
            knn_mat[idx[self.n_sources - 1]][i] = float(ans[self.n_sources - 1] == y_val_sample) / self.n_sources
            cur = self.n_sources - 2
            for j in range(self.n_sources - 1):
                const_factor=min(cur+1, n_neighbors)/(cur+1)
                knn_mat[idx[cur]][i] = knn_mat[idx[cur + 1]][i] + float(int(ans[cur] == y_val_sample) - int(ans[cur + 1] == y_val_sample)) / n_neighbors * const_factor
                cur -= 1 
        self.results['KNN']=np.mean(knn_mat, axis=1)
        self.time_results['KNN']=time.time()-time_init
        print('Done: KNN-SHAP score calculation!',flush=True)

    def _calculate_marginal_contributions(self, model_name='model_name'):
        """
        Calculate Marginal Contributions.
        """
        print(f'Start: Marginal Contribution Calculation!', flush=True)
        time_init = time.time()
        self.prob_marginal_contrib=np.zeros((self.n_sources, self.n_sources))
        self.prob_marginal_contrib_count=np.zeros((self.n_sources, self.n_sources))
        self.prob_marginal_contrib_mean=np.zeros((0, self.n_sources))
        self.prob_GR_dict=dict()
        
        for iters in range(100*self.max_iters):
            # we check the convergence every 100 random sets.
            self.prob_GR_dict[100*iters]=utils.check_convergence(self.prob_marginal_contrib_mean)
            if self.prob_GR_dict[100*iters] < self.GR_threshold:
                break
            else:
                marginal_contribs=self._calculate_marginal_contributions_core()
                self.prob_marginal_contrib_mean=np.concatenate([self.prob_marginal_contrib_mean, marginal_contribs])
        
        self.time_results[f'Beta({model_name})']=time.time()-time_init
        self.GR_results[f'Beta({model_name})']=self.prob_GR_dict
        self.MC_obs_card_mat=self.prob_marginal_contrib
        self.MC_count_obs_card_mat=self.prob_marginal_contrib_count
        print(f'The total/max iterations: {(100*iters)}/{(100*100*self.max_iters)}',flush=True)
        print(f'Done: Marginal Contribution Calculation! ', flush=True)

    def _compute_beta_shap_from_MC(self, model_name='model_name'):
        """
        Compute Beta Shapley from marginal contributions.
        """
        shap_value_weight=np.sum(self.prob_marginal_contrib*self.weight_list, axis=1)
        self.weights[model_name]=self.weight_list
        self.results[model_name]=shap_value_weight/np.mean(np.sum(self.prob_marginal_contrib_count, axis=1))

    def _calculate_marginal_contributions_core(self):
        """
        Compute marginal contribution for Beta Shapley. 
        """
        marginal_contribs=np.zeros((0, self.n_sources))
        for _ in range(100):
            # for each iteration, we use random permutation of indices.
            idxs=np.random.permutation(self.n_sources)
            marginal_contribs_tmp=np.zeros(self.n_sources)
            X_batch=np.zeros((0,) + tuple(self.X.shape[1:]))
            if self.is_regression is not True:
                y_batch = np.zeros(0, int)
            else:
                y_batch = np.zeros((0,)+ tuple(self.y.shape[1:]))
            truncation_counter=0
            for n, idx in enumerate(idxs):
                X_batch=np.concatenate([X_batch, self.X[self.sources[idx]]])
                y_batch=np.concatenate([y_batch, self.y[self.sources[idx]]])
                if n < (self.min_cardinality-1):
                    continue
                # When n == (self.min_cardinality-1), we compute performance based on 'self.min_cardinality' samples
                if n == (self.min_cardinality-1):
                    # first old score is based on 'self.min_cardinality' samples
                    try:
                        self.model.fit(X_batch, y_batch)
                        old_score=self.value() 
                    except:
                        old_score=self.random_score
                    continue
                    
                try:
                    # 'new_score' is the performance with 'idx' sample.
                    # Baseline model ('old_score') is based on 'n' samples
                    # self.weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j) in the paper).
                    self.model.fit(X_batch, y_batch)
                    new_score=self.value()    
                except:
                    new_score=self.random_score
                marginal_contribs_tmp[idx]=(new_score-old_score)*self.weight_list[n]

                # When the cardinality of random set is 'n', 
                self.prob_marginal_contrib[idx, n]+=marginal_contribs_tmp[idx]
                self.prob_marginal_contrib_count[idx, n]+=1

                distance_to_full_score=np.abs(new_score*self.weight_list[n]/(sum(marginal_contribs_tmp)+1e-12)) # np.abs((new_score-old_score)/(new_score+1e-12))
                # If updates are too small then we assume it contributes 0.
                if distance_to_full_score <= 1e-6:
                    truncation_counter += 1
                    if truncation_counter > 10:
                        print(f'Among {self.n_sources}, {n} samples are observed!', flush=True)
                        break
                else:
                    truncation_counter = 0
                # update score
                old_score=new_score
            marginal_contribs=np.concatenate([marginal_contribs, marginal_contribs_tmp.reshape(1,-1)])

        return marginal_contribs


