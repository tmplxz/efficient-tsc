import os
import time

import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm

EPS = np.finfo(np.float32).eps

def binarize(Y, n):
    return -torch.ones(Y.shape[0], n).scatter(-1, torch.tensor(Y[:, None]).long(), -1)

def stratified_split(Y, validation_size, seed=None):
    rng = np.random.default_rng(seed) if seed >= 0 else np.random.default_rng()
    U, C = np.unique(Y, return_counts = True)
    _C = ((C / C.sum()) * validation_size).round().clip(1).astype(np.int64)
    VA = np.zeros(_C.sum(), dtype = np.int64)
    a = 0
    for i, y in enumerate(U):
        c = _C[i]
        b = a + c
        J = (Y == y).nonzero()[0]
        K = rng.choice(J, c, replace = False)
        VA[a:b] = K
        a = b
    return np.setdiff1d(np.arange(Y.shape[0]), VA), VA

class Scaler(nn.Module):

    def __init__(self, num_values, device, dtype, **kwargs):

        super().__init__()
        self.register_buffer("_mean", torch.zeros(num_values, dtype=dtype, device=device))
        self.register_buffer("_std", torch.zeros(num_values, dtype=dtype, device=device))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.int64, device=device))
        self.register_buffer("_eps", torch.tensor(kwargs.get("eps", EPS * 10), device=device))
        self.register_buffer("_with_std", torch.tensor(kwargs.get("with_std", True), device=device))

    def partial_fit(self, X):

        batch_size = X.shape[0]
        new_count = self._count + batch_size

        batch_mean = X.mean(0)
        batch_std = X.std(0) if batch_size > 1 else 0

        self._mean = self._mean + ((batch_mean - self._mean) * (batch_size / new_count))
        self._std = self._std + ((batch_std - self._std) * (batch_size / new_count))

        self._count = new_count

    def fit(self, X):
        self._mean = X.mean(0)
        self._std = X.std(0)

    def scale(self, X):
        if self._with_std:
            return (X - self._mean) / (self._std + self._eps)
        else:
            return (X - self._mean)
        
    def fit_transfrom(self, X):
        self.fit(X)
        return self.scale(X)

class RidgeClassifier():

    def __init__(self, transform, device="cpu", seed=None, **kwargs):

        self.transform = transform
        self.device = device
        self.X_scaler = None
        self.Y_scaler = None
        self.lambdas = kwargs.get("lambdas", torch.logspace(-6, 6, 21))
        self.B = None
        self.B0 = None
        self.seed = seed

    def fit(self, training_data, **kwargs):

        n, p, k = training_data.shape[0], self.transform.num_features, kwargs.get("num_classes", len(training_data.classes))
        max_val_size = kwargs.get("max_val_size", 8_192)
        val_size = min(int(n * 0.2), max_val_size)

        if n < p: # low amounts of data, so first transform all batches, then fit ridge

            # calculate transformation features
            X0, Y0 = torch.zeros((n, p), device = self.device), torch.zeros((n, k), device = self.device)
            i = 0
            for X, Y in tqdm(training_data, total=np.ceil(n/training_data.batch_size)):
                j = i + X.shape[0]
                _X = self.transform(torch.tensor(X.astype(np.float32, copy=False)).to(self.device))
                _Y = binarize(Y, k)
                X0[i:j], Y0[i:j] = _X, _Y
                i = j

            # scale features
            if self.X_scaler is None:
                self.X_scaler = Scaler(num_features=p, device=self.device, dtype=X0.dtype)
                self.Y_scaler = Scaler(num_features=p, device=self.device, dtype=Y0.dtype, with_std=False)
            X0 = self.X_scaler.fit_transfrom(X0)
            Y0 = self.Y_scaler.fit_transfrom(Y0)
            self.B0 = self.Y_scaler._mean.to(self.device)
            
            # Ridge regression shortcut (via eigendecomposition) Tew et al. @NeurIPS 2023
            S2, U = torch.linalg.eigh((X0 @ X0.T))
            S2 = S2.clip(EPS)
            S = S2.sqrt()
            V = (X0.T @ U) * (1 / S)
            R = U * S
            R2 = R ** 2
            RTY = R.T @ Y0
            best_alpha_hat = None
            best_error = np.inf
            for lambda_ in self.lambdas * np.sqrt(n):
                alpha_hat = (1 / (S2[:, None] + lambda_)) * RTY
                Y_hat = R @ alpha_hat
                E = Y0 - Y_hat
                diag_H = (R2 / (S2 + lambda_)).sum(1)
                E_loocv = E / (1 - diag_H[:, None]).clip(EPS)
                err_lambda = (E_loocv ** 2).mean()
                if err_lambda < best_error:
                    best_error = err_lambda
                    best_alpha_hat = alpha_hat

            self.B = V @ best_alpha_hat

        else: # n >= p => memory-efficient fitting, estimating the LOOCV error with a validation set - see Dempster et al. @ AALTD 2024

            # split data
            TR, VA = stratified_split(training_data.Y, val_size, self.seed)
            TR, VA = np.sort(TR), np.sort(VA)
            training_data_1, validation_data = training_data[TR], training_data[VA]
            n1, n2 = training_data_1.shape[0], validation_data.shape[0]

            # First pass: compute mean using Welford's algorithm (numerically stable)
            mean_X = torch.zeros(p, device=self.device, dtype=torch.float64)
            mean_Y = torch.zeros(k, device=self.device, dtype=torch.float64)
            count = 0
            
            for X, Y in tqdm(training_data_1, total=np.ceil(n1/training_data_1.batch_size)):
                _X = self.transform(torch.tensor(X.astype(np.float32, copy=False)).to(self.device).float())[0]
                _Y = binarize(Y, k).to(self.device)
                
                batch_size = _X.shape[0]
                _X_f64 = _X.double()
                _Y_f64 = _Y.double()
                
                # Welford update for mean
                for i in range(batch_size):
                    count += 1
                    delta_X = _X_f64[i] - mean_X
                    mean_X += delta_X / count
                    delta_Y = _Y_f64[i] - mean_Y
                    mean_Y += delta_Y / count
            
            # Initialize scaler with computed mean
            if self.X_scaler is None:
                self.X_scaler = Scaler(num_values=p, device=self.device, dtype=torch.float32)
                self.Y_scaler = Scaler(num_values=k, device=self.device, dtype=torch.float32, with_std=False)
            
            self.X_scaler._mean = mean_X.float()
            self.Y_scaler._mean = mean_Y.float()
            self.B0 = self.Y_scaler._mean.to(self.device)
            
            # Second pass: compute gram matrix with stable scaling
            XTX = torch.zeros((p, p), device=self.device, dtype=torch.float64)
            XTY = torch.zeros((p, k), device=self.device, dtype=torch.float64)
            
            for X, Y in tqdm(training_data_1, total=np.ceil(n1/training_data_1.batch_size)):
                _X = self.transform(torch.tensor(X.astype(np.float32, copy=False)).to(self.device).float())[0]
                _Y = binarize(Y, k).to(self.device)
                
                # Scale using the stable mean computed in first pass
                _X_scaled = self.X_scaler.scale(_X).double()
                _Y_scaled = self.Y_scaler.scale(_Y).double()
                
                XTX += _X_scaled.T @ _X_scaled
                XTY += _X_scaled.T @ _Y_scaled

            # XTX and XTY are now already scaled, now calculate eigenvalue decomp
            S2, V = torch.linalg.eigh(XTX.to(self.device))
            S2 = S2.clip(EPS)
            # calculate validation transformation features
            XV, YV = torch.zeros((n2, p), device=self.device), torch.zeros(n2, dtype=torch.int64, device=self.device)
            i = 0
            for i, (X, Y) in enumerate(validation_data):
                j = i + X.shape[0]
                X_, _ = self.transform(torch.tensor(X.astype(np.float32)).to(self.device))
                _XV = self.X_scaler.scale(X_)
                XV[i:j], YV[i:j] = _XV, torch.tensor(Y, dtype = torch.int64)
                i = j

            # perform ridge regression with LOOCV on validation set
            best_error = np.inf
            for lambda_ in self.lambdas * np.sqrt(n1):
                _XTXi = (V * (1 / (S2 + lambda_))) @ V.T
                _B = (_XTXi @ XTY).float()
                err_lambda = (YV != ((XV @ _B) + self.B0).argmax(-1)).float().mean()
                if err_lambda < best_error:
                    best_error = err_lambda
                    self.B = _B.clone()

            # check reproducibility of fitting the ridge regression
            # for arr, fname in zip([XTX, XTY, S2, V, XV, YV, self.B], ['XTX', 'XTY', 'S2', 'V', 'XV', 'YV', 'B']):
                # torch.save(arr, f"{fname}.pt")
                # tmp = torch.load(f"{fname}.pt")
                # assert torch.allclose(arr, tmp, atol=1e-05)

            # delete temporary data sets
            validation_data.close()
            training_data_1.close()

    def save_to_disk(self, path):
        for attr_name in ['transform', 'B0', 'B', 'X_scaler', 'Y_scaler']:
            if hasattr(getattr(self, attr_name), 'state_dict'):
                torch.save(getattr(self, attr_name).state_dict(), os.path.join(path, f"{attr_name}.pth"))
                # for testing reproducibility, check the similarity with a previously stored model:
                # for key, tensor in getattr(self, attr_name).state_dict().items():
                #     tmp = torch.load(PATH mlruns/0/a14c5a64e91b47d0b96257693207f363/artifacts', f'{attr_name}.pth'))
                #     assert torch.allclose(tensor, tmp[key])
            else:
                torch.save(getattr(self, attr_name), os.path.join(path, f"{attr_name}.pt"))
                # for testing reproducibility, check the similarity with a previously stored model:
                # tmp = torch.load(os.path.join('PATH mlruns/0/a14c5a64e91b47d0b96257693207f363/artifacts', f'{attr_name}.pt'))
                # assert torch.allclose(getattr(self, attr_name), tmp)
        fsizes = [os.path.getsize(os.path.join(path, fname)) for fname in ['X_scaler.pth', 'Y_scaler.pth', 'transform.pth', 'B.pt', 'B0.pt']]
        return sum(fsizes)
        
    def load_from_disk(self, path):
        for attr_name in ['transform', 'B0', 'B', 'X_scaler', 'Y_scaler']:
            if attr_name == 'X_scaler' and self.X_scaler is None: # init scalers based on loaded info from transform and B / B0
                self.X_scaler = Scaler(num_values=self.transform.num_features, device=self.device, dtype=torch.float32)
                self.Y_scaler = Scaler(num_values=self.B.shape[1], device=self.device, dtype=torch.float32, with_std=False)
            if hasattr(getattr(self, attr_name), 'load_state_dict'):
                getattr(self, attr_name).load_state_dict(torch.load(os.path.join(path, f"{attr_name}.pth")))
            else:
                setattr(self, attr_name, torch.load(os.path.join(path, f"{attr_name}.pt"), map_location=self.device))
        self.X_scaler.to(self.device).eval()
        self.transform.to(self.device).eval()
        fsizes = [os.path.getsize(os.path.join(path, fname)) for fname in ['X_scaler.pth', 'transform.pth', 'B.pt', 'B0.pt']]
        return sum(fsizes)

    def count_params(self):
        p_transf = self.transform.I.shape.numel() + self.transform.W.shape.numel()
        p_scaler = sum([ s._mean.numel() + s._std.numel() + s._count.numel() + s._eps.numel() for s in [self.X_scaler, self.Y_scaler]])
        return p_transf + p_scaler + self.B.numel() + self.B0.numel()
    
    def _predict_single(self, X):
        # t0 = time.time()
        X_ = torch.tensor(X.astype(np.float32, copy=False)).to(self.device)
        # t00 = time.time()
        _X, t_lookup = self.transform(X_)
        # t1 = time.time()
        _X = self.X_scaler.to(_X.device).scale(_X)
        # t2 = time.time()
        res = _X @ self.B + self.B0
        # t3 = time.time()
        return res, t_lookup, 0, 0, 0, 0 # , t00-t0, t1-t00, t2-t1, t3-t2

    def _predict(self, test_data, **kwargs):

        n, k = test_data.shape[0], kwargs.get("num_classes", len(test_data.classes))
        
        # legacy code that crashes on large data sets because it does not use data batching
        # _X2 = self.transform(torch.tensor(test_data.X.astype(np.float32, copy=False)).to(self.device))
        # _X2 = self.X_scaler.to(_X2.device).scale(_X2)
        # Y1 = _X2 @ self.B + self.B0

        # new code with correct batching
        Y0 = torch.zeros((n, k), device=self.device)
        i = 0
        t_lookup_total, t0, t1, t2, t3 = 0, 0, 0, 0, 0
        for cnt, (X, _) in enumerate(tqdm(test_data, total=np.ceil(n/test_data.batch_size))):
            j = i + X.shape[0]
            Y0[i:j,:], t_lookup, t0_, t1_, t2_, t3_ = self._predict_single(X)
            i = j
            # t0 += t0_
            # t1 += t1_
            # t2 += t2_
            # t3 += t3_
            # t_lookup_total += t_lookup
        
        # print(f"PRUNED HYDRA LOOKUP TIME {t_lookup_total:4.2f}s")
        print(f'HYDRA PREDICT OVER {np.ceil(n/test_data.batch_size)} BATCHES WITH {X.shape} VALUES AND {k} CLASSES, {self.transform.num_features} LATENT FEATS & {self.transform.W.numel()} TRANSF PARAMS : to_gpu {t0:4.2f} transform {t1:4.2f}s scale {t2:4.2f}s ridge {t3:4.2f}s lookup {t_lookup_total:4.2f}s')
        return Y0

    def score(self, data):

        incorrect = 0
        count = 0

        for X, Y in tqdm(data, total = np.ceil(data.shape[0] / data.batch_size)):

            incorrect += (torch.tensor(Y).to(self.device) != self._predict_single(X).argmax(-1)).sum()
            count += X.shape[0]

        return incorrect / count