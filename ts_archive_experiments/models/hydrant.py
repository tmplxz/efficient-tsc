import time
from itertools import product
import os
import json
import math

import numpy as np
import torch
import mlflow
from tqdm import tqdm

from models.quant import QuantClassifier
from models.hydra import HydraMultivariateGPU
from models.phydra import prepare_group_info_io


def estimate_hydra_subbatches(n_instances, n_channels, length, num_kernels_per_group, num_groups, max_memory_gb=24.0, overhead=3, dtype=torch.float32):
    """
    Estimate how many subbatches are needed to keep hydra memory usage below max_memory_gb.
    """
    # Get dtype size in bytes
    bytes_per_elem = torch.finfo(dtype).bits // 8
    total_bytes = n_instances * n_channels * length * bytes_per_elem * num_kernels_per_group * num_groups * overhead # expected tensor sizes + some overhead
    total_gb = total_bytes / (1024**3)
    subbatch = math.ceil(total_gb / max_memory_gb) # Compute required number of splits so each fits under limit
    if subbatch > 1:
        print(f'Hydrant will perform Hydra transformations on {subbatch} sub-batches per {n_instances} instances batch, to not extend memory capacity of {max_memory_gb} GB.')
    return subbatch # TODO improve by also checking whether pruned models can handle larger batches?


class Hydrant(QuantClassifier):

    def __init__(self, config):
        seed = config['seed'] if config['seed'] >= 0  else None
        super().__init__(classifier=config['classifier'], num_estimators=config['num_estimators'], max_depth=config['max_depth'], max_features=config['max_features'], criterion=config['criterion'], random_state=seed)
        self.classifier = self.clsf_cls(n_estimators=0, criterion=self.criterion, max_features=self.max_features, n_jobs=-1, warm_start=True, random_state=self.random_state) # unrestricted tree depth for deriving feature importance
        self.config = config
        self.hydra_transf = HydraMultivariateGPU(config)
        self.prune_rate = config['prune_rate']
        self.hydra_subbatch = estimate_hydra_subbatches(config['batch_size'], config['n_channels'], config['length'], config['num_kernels_per_group'], config['num_groups'])
        mlflow.log_param('hydra_subbatch', self.hydra_subbatch)

    def transform_(self, X, Y=None, iter=-1):
        X_ = torch.tensor(X.astype(np.float32, copy=False))
        ZQ = self.transform.fit_transform(X_, Y) if iter == 0 else self.transform.transform(X_)
        if self.hydra_subbatch == 1: # no sub-batching
            ZH, _ = self.hydra_transf(X_.to(device=self.hydra_transf.W.device))
        else: # write sub-batch results into prepared tensor on specified GPU
            i, ZH = 0, torch.zeros((ZQ.shape[0], self.hydra_transf.num_features), device=self.config['device'])
            for X__ in torch.chunk(X_, self.hydra_subbatch, dim=0):
                ZH[i:(i+X__.shape[0])] = self.hydra_transf(X__.to(device=self.config['device']))[0]
                i += X__.shape[0]
        ZH = ZH.to('cpu')
        
        # OTHER SUB-BATCHING IMPLEMENTATIONS (but empirically slower)
        # write into list on CPU, and afterwards concat
        # ZH_ = [ self.hydra_transf(X__.to(device=self.config['device']))[0] for X__ in torch.chunk(X_, self.hydra_subbatch, dim=0) ] # smaller batches for Hydra, to not crash memory
        # ZH_ = torch.cat(ZH_).to('cpu')
        # write into prepared tensor on CPU
        # ZH__, i = torch.zeros((ZQ.shape[0], self.hydra_transf.num_features)), 0
        # for X__ in torch.chunk(X_, self.hydra_subbatch, dim=0):
        #     ZH__[i:(i+X__.shape[0])] = self.hydra_transf(X__.to(device=self.config['device']))[0].to('cpu')
        #     i += X__.shape[0]

        return torch.cat([ZH, ZQ], 1)

    def fit(self, training_data, **kwargs):

        if self.limit_mb > 0:
            training_data.set_batch_size(self.limit_mb)
        else:
            training_data._reset()

        num_batches = training_data._num_batches
        num_estimators_per_batch = self._set_num_estimators(num_batches)
        for i, (X, Y) in tqdm(enumerate(training_data), total=num_batches, desc='Batch-wise Hydrant training'):
            self.classifier.n_estimators += num_estimators_per_batch[i]
            Z = self.transform_(X, Y, i) # Y and i are only given to transform_ to fit_transform Quant in the first batch
            self.classifier.fit(Z, Y)

        if self.prune_rate > 0: # prune everything but the most important features for inference efficiency

            results = {
                'n_par_pre_prune':  self.count_params(),
                'n_ft_pre_prune': self.classifier.n_features_in_,
                'n_qft_pre_prune': self.classifier.n_features_in_ - self.hydra_transf.num_features,
                'n_hft_pre_prune': self.hydra_transf.num_features,
                # 'train_accuracy_pre_prune': np.mean(scores),
            }

            print(f'Pruning {self.prune_rate*100:.0f}% of {results["n_ft_pre_prune"]} Hydrant features')
            
            # PRUNE QUANT INTERVALS
            avg_imp = {}
            ft_offset = 0 # increases with each representation
            for t_idx, transf in self.transform.models.items():
                self.transform.models[t_idx].important_intervals = [] # placeholder
                for interval_idx in np.unique(transf.ft_map):
                    interval_ft_idc = np.where(np.equal(transf.ft_map, interval_idx))[0]
                    ft_start, ft_end = ft_offset + interval_ft_idc.min(), ft_offset + interval_ft_idc.max()
                    avg_imp[(t_idx, interval_idx)] = np.mean(self.classifier.feature_importances_[ft_start:ft_end+1])
                ft_offset += len(transf.ft_map)

            # prune intervals with low importance
            sorted_imp = sorted(avg_imp.items(), key=lambda item: item[1], reverse=True)
            for (transf_idx, interv_idx), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.prune_rate)))]:
                self.transform.models[transf_idx].important_intervals.append(interv_idx)

            # PRUNE HYDRA KERNELS
            imp = torch.tensor(self.classifier.feature_importances_[-self.hydra_transf.num_features:])
            kernel_imp = imp.view(self.hydra_transf.num_dilations, self.hydra_transf.divisor, self.hydra_transf.k, self.hydra_transf.h, 2) # reformat to access min and max response counts of kernels
            mean_kernel_imp = kernel_imp.mean(dim=-1) # average the min and max response count importance -> shape (D, divisor, k, h)
            
            # identify groups with highest mean importance
            imp_per_group = {}
            for div, group in product(range(self.hydra_transf.divisor), range(self.hydra_transf.h)):
                imp_per_group[(div, group)] = mean_kernel_imp[:, div, :, group].mean().item()
            sorted_imp = sorted(imp_per_group.items(), key=lambda item: item[1], reverse=True)
            important_groups = {}
            num_feat = 0
            for (div, group), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.prune_rate)))]:
                important_groups.setdefault(str(div), []).append(group) # str(div) instead of int because of json saving/loading
                num_feat += 1

            # Collect pruned kernels and info
            important_group_info, all_kernels, current_offset = {}, [], 0
            for dil in range(self.hydra_transf.num_dilations):
                for div, groups in important_groups.items():
                    orig_kernels = self.hydra_transf.W[dil, int(div)]
                    div_h = len(groups)
                    keep_kernels = orig_kernels.view(self.hydra_transf.k, self.hydra_transf.h, 1, self.hydra_transf.l)[:, groups].view(self.hydra_transf.k * div_h, 1, self.hydra_transf.l)
                    # Flatten and store
                    keep_kernels_flat = keep_kernels.flatten()
                    end_offset = current_offset + keep_kernels_flat.numel()
                    
                    important_group_info[f"{dil}_{div}"] = {
                        'start': torch.tensor(current_offset, device=self.config['device']),
                        'end': torch.tensor(end_offset, device=self.config['device']),
                        'h': torch.tensor(div_h, device=self.config['device']),
                        'groups': torch.tensor(groups, device=self.config['device']),
                        'shape': keep_kernels.shape
                    }
                    
                    all_kernels.append(keep_kernels_flat)
                    current_offset = end_offset

            # Concatenate all into single tensor
            important_group_info['use_diff'] = any([key.endswith('1') for key in important_group_info.keys()])
            self.hydra_transf = HydraMultivariateGPU(self.config, torch.cat(all_kernels), important_group_info)

            # retrain classifier on new transformations
            self.classifier = self.clsf_cls(n_estimators=0, criterion=self.criterion, max_features=self.max_features, max_depth=self.max_depth, n_jobs=-1, warm_start=True, random_state=self.random_state)

            for i, (X, Y) in tqdm(enumerate(training_data), total=num_batches, desc='Batch-wise Hydrant training after pruning'):
                self.classifier.n_estimators += num_estimators_per_batch[i]
                Z = self.transform_(X)
                self.classifier.fit(Z, Y)

            results.update({
                # 'train_accuracy_post_prune': np.mean(scores),
                'n_par_post_prune': self.count_params(),
                'n_ft_post_prune': self.classifier.n_features_in_,
                'n_qft_post_prune': self.classifier.n_features_in_ - self.hydra_transf.num_features,
                'n_hft_post_prune': self.hydra_transf.num_features,
            })

            print('PRUNING HYDRANT RESULTS:', results)
            for key, val in results.items():
                mlflow.log_metric(f"hydrant_{key}", val)

        self._is_fitted = True

    def _predict(self, test_data, **kwargs):
        
        pred = []
        for i, (X, Y) in tqdm(enumerate(test_data), total=np.ceil(test_data.shape[0]/test_data.batch_size)):
            Z = self.transform_(X)
            pred.append(self.classifier.predict(Z))
            # if i == 3:
            #     for fn, arr in zip(['X', 'ZQ', 'ZH', 'Y', 'P'], [X, ZQ.numpy(), ZH.cpu().numpy(), Y, pred[-1]]):
            #         if os.path.isfile(f'{fn}.npy'):
            #             arr2 = np.load(f'{fn}.npy')
            #             assert np.all(np.isclose(arr, arr2))
            #         np.save(f'{fn}.npy', arr)

        pred = np.concatenate(pred, axis=0)
       
        return pred
    
    def save_to_disk(self, path):
        fsizes = super().save_to_disk(path) # quant and classifier
        torch.save(self.hydra_transf.state_dict(), os.path.join(path, "hydra.pth")) # hydra weights
        fsizes += os.path.getsize(os.path.join(path, f"hydra.pth"))
        if self.hydra_transf.important_groups is not None: # hydra transform
            imp_group_info = prepare_group_info_io(self.hydra_transf.important_groups)
            json.dump(imp_group_info, open(os.path.join(path, "imp_groups.json"), 'w'))
            self.hydra_transf.important_groups = prepare_group_info_io(imp_group_info, False, self.config["device"])
            fsizes += os.path.getsize(os.path.join(path, f"imp_groups.json"))
        return fsizes

    def load_from_disk(self, path):
        fsizes = super().load_from_disk(path) # load quant
        if os.path.isfile(os.path.join(path, "imp_groups.json")): # load results after pruning
            imp_group_info = json.load(open(os.path.join(path, "imp_groups.json"), 'r'))
            imp_group_info = prepare_group_info_io(imp_group_info, False, self.config["device"])
            fsizes += os.path.getsize(os.path.join(path, f"imp_groups.json"))
            all_kernels = torch.load(os.path.join(path, "hydra.pth"))['W']
            self.hydra_transf = HydraMultivariateGPU(self.config, all_kernels, imp_group_info) # init new, based on pruned info
        self.hydra_transf.load_state_dict(torch.load(os.path.join(path, "hydra.pth")))
        fsizes += os.path.getsize(os.path.join(path, f"hydra.pth"))
        return fsizes
