from itertools import product
import os
import json

from models.hydra import HydraMultivariateGPU
from models.ridge import RidgeClassifier, torch

import mlflow

def prepare_group_info_io(important_groups, to_json=True, device='cpu'):
    for key, vals in important_groups.items():
        if key != 'use_diff':
            if to_json:
                important_groups[key]['groups'] = vals['groups'].cpu().tolist()
                important_groups[key]['shape'] = list(vals['shape'])
                for key2 in ['start', 'end', 'h']:
                    important_groups[key][key2] = important_groups[key][key2].item()
            else: # from json
                important_groups[key]['groups'] = torch.tensor(vals['groups']).to(device)
                important_groups[key]['shape'] = torch.Size(vals['shape'])
                for key2 in ['start', 'end', 'h']:
                    important_groups[key][key2] = torch.tensor(important_groups[key][key2], device=device)
    return important_groups

class PrunedHydra():

    def __init__(self, config, **kwargs):
        self.config = config
        self.trnsf = HydraMultivariateGPU(config)
        self.clsf = RidgeClassifier(transform=self.trnsf, device=config["device"], seed=config['seed'], **kwargs)
        
    def fit(self, training_data, **kwargs):
        # train initial model
        self.clsf.fit(training_data, **kwargs)
        results = {
            'n_par_pre_prune': self.count_params(),
            "n_ft_pre_prune": self.clsf.B.shape[0],
            # "acc_pre_pruning": self.clsf.score(training_data).item()
        }

        print(f'Pruning {self.config["prune_rate"]*100:.0f}% of {results["n_ft_pre_prune"]} Hydra features')
        
        # check feature importance
        imp = torch.mean(torch.abs(self.clsf.B), dim=1) # across classes -> (2048,)
        kernel_imp = imp.view(self.trnsf.num_dilations, self.trnsf.divisor, self.trnsf.k, self.trnsf.h, 2) # reformat to access min and max response counts of kernels
        mean_kernel_imp = kernel_imp.mean(dim=-1) # average the min and max response count importance -> shape (D, divisor, k, h)
        
        # identify groups with highest mean importance
        imp_per_group = {}
        for div, group in product(range(self.trnsf.divisor), range(self.trnsf.h)):
            imp_per_group[(div, group)] = mean_kernel_imp[:, div, :, group].mean().item()
        sorted_imp = sorted(imp_per_group.items(), key=lambda item: item[1], reverse=True)
        important_groups = {}
        num_feat = 0
        for (div, group), _ in sorted_imp[:(int(len(sorted_imp) * (1-self.config["prune_rate"])))]:
            important_groups.setdefault(str(div), []).append(group) # str(div) instead of int because of json saving/loading
            num_feat += 1

        # Collect pruned kernels and info
        important_group_info, all_kernels, current_offset = {}, [], 0
        for dil in range(self.trnsf.num_dilations):
            for div, groups in important_groups.items():
                orig_kernels = self.trnsf.W[dil, int(div)]
                div_h = len(groups)
                keep_kernels = orig_kernels.view(self.trnsf.k, self.trnsf.h, 1, self.trnsf.l)[:, groups].view(self.trnsf.k * div_h, 1, self.trnsf.l)
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
        self.trnsf = HydraMultivariateGPU(self.config, torch.cat(all_kernels), important_group_info)
        
        # create new ridge classifier
        self.clsf = RidgeClassifier(transform=self.trnsf, device=self.config['device'], seed=self.config['seed'], **kwargs)
        self.clsf.fit(training_data, **kwargs)
        results.update({
            'n_par_post_prune': self.count_params(),
            "n_ft_post_pruning": self.clsf.B.shape[0],
            # "acc_post_pruning": self.clsf.score(training_data).item()
        })
        print('PRUNING', results)
        for key, val in results.items():
            mlflow.log_metric(f"phydra_{key}", val)

    def save_to_disk(self, path):
        # after pruning, store the information in json and without torch datatypes
        imp_group_info = prepare_group_info_io(self.trnsf.important_groups)
        json.dump(imp_group_info, open(os.path.join(path, "imp_groups.json"), 'w'))
        self.trnsf.important_groups = prepare_group_info_io(imp_group_info, False, self.config["device"])
        fsize = self.clsf.save_to_disk(path) + os.path.getsize(os.path.join(path, "imp_groups.json"))
        return fsize
        
    def load_from_disk(self, path):
        # after pruning, load the information from json and store with torch datatypes
        if os.path.isfile(os.path.join(path, "imp_groups.json")):
            imp_group_info = json.load(open(os.path.join(path, "imp_groups.json"), 'r'))
            imp_group_info = prepare_group_info_io(imp_group_info, False, self.config["device"])
        # load weights
        state_dict = torch.load(os.path.join(path, "transform.pth"))
        self.trnsf = HydraMultivariateGPU(self.config, state_dict['W'], imp_group_info)
        self.trnsf.load_state_dict(state_dict)
        self.clsf = RidgeClassifier(transform=self.trnsf, device=self.config['device'], seed=self.config['seed'])
        fsize = self.clsf.load_from_disk(path) + os.path.getsize(os.path.join(path, "imp_groups.json"))
        return fsize

    def count_params(self):
        return self.clsf.count_params()
    
    def _predict(self, test_data, **kwargs):
        return self.clsf._predict(test_data, **kwargs)
