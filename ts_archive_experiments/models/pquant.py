import time

import mlflow
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm

from models.quant import QuantClassifier

def haar_downsample(X): # 1D Haar wavelet downsampling to each channel
    B, C, _ = X.shape
    kernel = torch.tensor([1., 1.], device=X.device).reshape(1, 1, 2) / 2 # Haar lowpass filter [1, 1] / 2, repeated for each channel
    kernel = kernel.repeat(C, 1, 1)      # shape: (channels, 1, 2)
    return F.conv1d(X, kernel, stride=2, groups=C)

def grad_mag(X):
    g = X.diff()
    return torch.sqrt(g**2 + 1e-8)

def laplacian(X): # 1D Laplacian filter [1, -2, 1] to each channel
    B, C, _ = X.shape
    kernel = torch.tensor([1., -2., 1.], device=X.device).reshape(1, 1, 3) # Laplacian kernel: [1, -2, 1]
    kernel = kernel.repeat(C, 1, 1)   # shape: (channels, 1, 3)
    return F.conv1d(X, kernel, padding=1, groups=C)

def normalize_energy(X):
    return X / (torch.norm(X, dim=-1, keepdim=True) + 1e-8)


class PrunedQuant(QuantClassifier):

    def __init__(self, prune_rate=0.8, classifier='XRF', num_estimators=100, max_depth=20, max_features=0.1, criterion="entropy", random_state=None, limit_mb=-1, **kwargs):

        super().__init__(classifier, num_estimators, max_depth, max_features, criterion, random_state, limit_mb, **kwargs)
        self.classifier = self.clsf_cls(n_estimators=0, criterion=self.criterion, max_features=self.max_features, n_jobs=-1, warm_start=True, random_state=self.random_state) # unrestricted tree depth for deriving feature importance
        self.prune_rate = prune_rate

    def fit(self, training_data, **kwargs):
        # fit initial random forest
        super().fit(training_data)
        num_batches = training_data._num_batches
        num_estimators_per_batch = self._set_num_estimators(num_batches)

        results = {
            'n_par_pre_prune':  self.count_params(),
            'n_ft_pre_prune': self.classifier.n_features_in_,
            # 'train_accuracy_pre_prune': np.mean(scores),
        }

        # identify mean feature importance per interval
        imp = self.classifier.feature_importances_
        avg_imp = {}
        ft_offset = 0 # increases with each representation
        for t_idx, transf in self.transform.models.items():
            self.transform.models[t_idx].important_intervals = [] # placeholder, later to be filled with intervals
            for interval_idx in np.unique(transf.ft_map):
                interval_ft_idc = np.where(np.equal(transf.ft_map, interval_idx))[0]
                ft_start, ft_end = ft_offset + interval_ft_idc.min(), ft_offset + interval_ft_idc.max()
                avg_imp[(t_idx, interval_idx)] = np.mean(imp[ft_start:ft_end+1])
            ft_offset += len(transf.ft_map)
            
        # prune intervals with low importance
        sorted_imp = sorted(avg_imp.items(), key=lambda item: item[1], reverse=True)
        for (transf_idx, interv_idx), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.prune_rate)))]:
            self.transform.models[transf_idx].important_intervals.append(interv_idx)

        # refit with pruned intervals
        self.classifier = self.clsf_cls(n_estimators=0, criterion=self.criterion, max_features=self.max_features, max_depth=self.max_depth, n_jobs=-1, warm_start=True, random_state=self.random_state)
        scores = []
        for i, (X, Y) in tqdm(enumerate(training_data), total=num_batches):
            self.classifier.n_estimators += num_estimators_per_batch[i]
            Z = self.transform.transform(torch.tensor(X.astype(np.float32)))
            t0 = time.time()
            self.classifier.fit(Z, Y)
            # scores.append(self.classifier.score(Z, Y))
            t0 = time.time() # mlflow.log_metric(f"quant_fit_time_batch_{i}", time.time() - t0)

        results.update({
            # 'train_accuracy_post_prune': np.mean(scores),
            'n_par_post_prune': self.count_params(),
            'n_ft_post_prune': self.classifier.n_features_in_
        })
        for key, val in results.items():
            mlflow.log_metric(f"pquant_{key}", val)
        
        print('PRUNING', results)
        # print(f'N SAMPLES {Z.shape[0] * num_batches} PRE PRUNING: {n_ft_pre_prune} features, {n_par_pre_prune} parameters - POST PRUNING: {n_ft_post_prune} features, {n_par_post_prune} parameters', flush=True)
        t0 = time.time() # mlflow.log_metric(f"quaaaaant_time", time.time() - t0000)
