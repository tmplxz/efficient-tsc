# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# QUANT: A Minimalist Interval Method for Time Series Classification
# ECML PKDD 2024

import os
import joblib
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import torch, torch.nn.functional as F
from tqdm import tqdm

# == generate intervals ========================================================

def make_intervals(input_length, depth):

    exponent = \
    min(
        depth,
        int(np.log2(input_length)) + 1
    )

    intervals = []

    for n in 2 ** torch.arange(exponent):

        indices = torch.linspace(0, input_length, n + 1).long()

        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)

        intervals.append(intervals_n)

        if n > 1 and intervals_n.diff().median() > 1:

            shift = int(np.ceil(input_length / n / 2))

            intervals.append((intervals_n[:-1] + shift))

    return torch.cat(intervals)

# == quantile function =========================================================

def f_quantile(X, div = 4):

    n = X.shape[-1]

    if n == 1:

        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])
    
    else:
        
        num_quantiles = 1 + (n - 1) // div

        if num_quantiles == 1:

            quantiles = X.quantile(torch.tensor([0.5]), dim = -1).permute(1, 2, 0)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])
        
        else:
            
            quantiles = X.quantile(torch.linspace(0, 1, num_quantiles), dim = -1).permute(1, 2, 0)
            quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims = True)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

# == interval model (per representation) =======================================

class IntervalModel():

    def __init__(self, input_length, depth = 6, div = 4):

        assert div >= 1
        assert depth >= 1

        self.div = div

        self.intervals = make_intervals(input_length=input_length, depth=depth)
        self.important_intervals = None # will be transformed to a list after pruning
        self.ft_map = []

    def transform(self, X):

        features = []
        store_ft_map = self.ft_map == [] # in the first data pass, store a map of feature indices to intervals and representations

        if self.important_intervals is None: # before pruning
            for idx, (a, b) in enumerate(self.intervals):
                features.append(f_quantile(X[..., a:b], div=self.div).squeeze(1))
                if store_ft_map:
                    self.ft_map.extend([idx] * features[-1].shape[-1]) # store which features belong to which interval (idx)

        else: # after pruning
            for idx in self.important_intervals:
                features.append(f_quantile(X[..., self.intervals[idx][0]:self.intervals[idx][1]], div=self.div).squeeze(1))
        
        return torch.cat(features, -1)

# representation_functions 
def identity(X):
    return X

def avg_pool_diff(X):
    return F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1)

def diff2(X):
    return X.diff(n=2)

def fft_abs(X):
    return torch.fft.rfft(X).abs()

# == quant =====================================================================

class QuantTransform():

    def __init__(self, depth = 6, div = 4):

        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div
        self.representation_functions = (identity, avg_pool_diff, diff2, fft_abs)
        self.models = {}
        self.fitted = False

    def transform(self, X, test=False):
        assert self.fitted, "not fitted"
        # calculate all representations and apply individual interval quantile models
        features = []
        for idx, func in enumerate(self.representation_functions):
            if self.models[idx].important_intervals is None or len(self.models[idx].important_intervals) > 0:
                features.append( self.models[idx].transform(func(X)) )

        res = torch.cat(features, -1)
        return res
        
    def fit_transform(self, X, Y):

        features = []
        for idx, func in enumerate(self.representation_functions):
            # create interval transformers based on representation
            Z = func(X)
            self.models[idx] = IntervalModel(input_length=Z.shape[-1], depth=self.depth, div=self.div)
            features.append(self.models[idx].transform(Z))
        
        self.fitted = True
        res = torch.cat(features, -1)
        return res

# ==============================================================================

class QuantClassifier():

    def __init__(self, classifier='XRF', num_estimators=100, max_depth=20, max_features=0.1, criterion="entropy", random_state=None, limit_mb=-1, **kwargs):

        self.clsf_cls = ExtraTreesClassifier if classifier == 'XRF' else RandomForestClassifier
        self.num_estimators = num_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.num_estimators = num_estimators
        self.random_state = random_state
        self.limit_mb = limit_mb
        self.transform = QuantTransform()
        self.classifier = self.clsf_cls(n_estimators=0, criterion=self.criterion, max_features=self.max_features, max_depth=self.max_depth, n_jobs=-1, warm_start=True, random_state=self.random_state)
        self._is_fitted = False

    def fit(self, training_data, **kwargs):
        t0000 = time.time()

        if self.limit_mb > 0:
            training_data.set_batch_size(self.limit_mb)
        else:
            training_data._reset()
        # print(f"training_data.batch_size -> {training_data.batch_size}", flush = True)
        num_batches = training_data._num_batches
        num_estimators_per_batch = self._set_num_estimators(num_batches)
        # print(training_data.batch_size, num_batches, sum(num_estimators_per_batch), training_data.shape)
        t0 = time.time() # mlflow.log_metric(f"quant_init_time", time.time() - t0000)

        # print(f"num_batches -> {num_batches}", flush = True)
        # print(f"num_estimators_per_batch -> {num_estimators_per_batch}", flush = True)
        for i, (X, Y) in enumerate(tqdm(training_data, total=num_batches)):
            self.classifier.n_estimators += num_estimators_per_batch[i]
            
            # print(training_data.shape, X.shape)
            if i == 0:
                Z = self.transform.fit_transform(torch.tensor(X.astype(np.float32)), Y)
            else:
                Z = self.transform.transform(torch.tensor(X.astype(np.float32)))

            t0 = time.time()
            self.classifier.fit(Z, Y)
            t0 = time.time() # mlflow.log_metric(f"quant_fit_time_batch_{i}", time.time() - t0)

        self._is_fitted = True
        t0 = time.time() # mlflow.log_metric(f"quaaaaant_time", time.time() - t0000)

    def _set_num_estimators(self, num_batches):

        num_estimators_per = max(1, int(self.num_estimators / num_batches))

        num_estimators_per_batch = np.ones(num_batches, dtype = np.int32) * num_estimators_per

        _total = num_estimators_per_batch.sum()
        _diff = self.num_estimators - _total
        if _diff > 0:
            num_estimators_per_batch[:_diff] += 1

        return num_estimators_per_batch
    
    def _predict(self, test_data, **kwargs):
        
        # legacy code without batching
        # Z_ = self.transform.transform(torch.tensor(test_data.X.astype(np.float32)), test=True)
        # pred_ = self.classifier.predict(Z_)
        # score_ = np.mean(pred_ == test_data.Y)

        # new code with batching
        Y0, i = np.zeros((test_data.shape[0]), dtype=np.int64), 0
        for X, _ in tqdm(test_data, total=np.ceil(test_data.shape[0]/test_data.batch_size)):
            j = i + X.shape[0]
            Z = self.transform.transform(torch.tensor(X.astype(np.float32)), test=True)
            Y0[i:j] = self.classifier.predict(Z)
            i = j
        return Y0

    # def score(self, data):

    #     assert self._is_fitted

    #     num_incorrect = 0
    #     count = 0
        
    #     for X, Y in data:

    #         Z = self.transform.transform(torch.tensor(X.astype(np.float32)), test=True)

    #         num_incorrect += (self.classifier.predict(Z) != Y).sum()
    #         count += X.shape[0]

    #     return num_incorrect / count

    # def score_logloss(self, data):

    #     assert self._is_fitted

    #     num_incorrect = 0
    #     loss = 0
    #     count = 0
        
    #     for X, Y in data:

    #         Z = self.transform.transform(torch.tensor(X.astype(np.float32)), test=True)

    #         num_incorrect += (self.classifier.predict(Z) != Y).sum()
    #         # loss += F.nll_loss(torch.tensor(self.classifier.predict_proba(Z), dtype = torch.float32).log(), torch.tensor(Y, dtype = torch.int64), reduction = "sum")
    #         loss += F.nll_loss(torch.tensor(self.classifier.predict_proba(Z), dtype = torch.float32).clip(eps, 1 - eps).log(), torch.tensor(Y, dtype = torch.int64), reduction = "sum")
    #         count += X.shape[0]

    #     return num_incorrect / count, loss / count

    def save_to_disk(self, path):
        # Save sklearn classifier with joblib, and transform with torch if needed
        joblib.dump(self.classifier, os.path.join(path, "classifier.joblib"))
        # Save transform (Quant object) with torch or joblib
        joblib.dump(self.transform, os.path.join(path, "transform.joblib"))
        fsizes = [os.path.getsize(os.path.join(path, f"{f}.joblib")) for f in ['classifier', 'transform']]
        return sum(fsizes)

    def load_from_disk(self, path):
        self.classifier = joblib.load(os.path.join(path, "classifier.joblib"))
        self.transform = joblib.load(os.path.join(path, "transform.joblib"))
        fsizes = [os.path.getsize(os.path.join(path, f"{f}.joblib")) for f in ['classifier', 'transform']]
        return sum(fsizes)

    def count_params(self):
        # Return number of parameters in the classifier and transform
        num_params = 0
        for estimator in self.classifier.estimators_:
            num_params += estimator.tree_.node_count
        # Note: transform parameters are not counted here as they are not learned parameters
        return num_params
