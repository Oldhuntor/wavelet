import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



class DeanSubModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=True))  # hidden layers with bias
            layers.append(nn.ReLU())
            prev = h

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev, output_dim, bias=False)  # no bias on output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        out = self.output_layer(h)
        out = F.selu(out)  # SELU on output as required
        return out  # shape: [batch, output_dim]


class DeanEnsemble:
    def __init__(
        self,
        input_dim: int,
        ensemble_size: int = 50,
        hidden_dims: List[int] = [64, 32],
        feature_bagging: bool = True,
        bagging_ratio: float = 0.5,
        power: float = 2.0,
        device: Optional[torch.device] = None,
        seed: Optional[int] = 42,
    ):
        """
        :param input_dim: total number of features in the dataset
        :param ensemble_size: number of submodels
        :param hidden_dims: list specifying sizes of hidden layers for each submodel
        :param feature_bagging: whether to apply feature bagging
        :param bagging_ratio: fraction of features per submodel when bagging=True
        :param power: the integer 'power' used in aggregation (see Eq. 5)
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.hidden_dims = hidden_dims
        self.feature_bagging = feature_bagging
        self.bagging_ratio = bagging_ratio
        self.power = float(power)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # containers for submodels and their feature indices
        self.models: List[DeanSubModel] = []
        self.feature_sets: List[List[int]] = []
        self.qis: List[torch.Tensor] = []  # q_i after training (tensor scalar)

        for _ in range(self.ensemble_size):
            if self.feature_bagging:
                k = max(1, int(math.ceil(self.input_dim * self.bagging_ratio)))
                idx = sorted(random.sample(range(self.input_dim), k))
            else:
                idx = list(range(self.input_dim))

            sub = DeanSubModel(input_dim=len(idx), hidden_dims=self.hidden_dims, output_dim=1).to(self.device)
            self.models.append(sub)
            self.feature_sets.append(idx)

        # initialize qis as zeros; will be computed after training
        self.qis = [torch.tensor(0.0, device=self.device) for _ in range(self.ensemble_size)]


    def _train_single(
        self,
        model: DeanSubModel,
        train_x: torch.Tensor,
        batch_size: int = 1024,
        epochs: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ):
        """
        Trains the submodel to output target 1 for every training sample.
        Uses MSE loss: ||f(x) - 1||^2 (mean over batch) which matches the norm objective.
        """
        model.train()
        dataset = TensorDataset(train_x)  # targets are constant so not stored
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        target = torch.ones(1, device=self.device)  # scalar 1

        for epoch in range(epochs):
            epoch_losses = []
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)  # shape [batch, 1]
                # compute loss towards constant 1
                loss = criterion(out.squeeze(), target.expand_as(out.squeeze()))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            if verbose:
                print(f"  epoch {epoch+1}/{epochs}  loss={sum(epoch_losses)/len(epoch_losses):.6f}")


    def fit(
        self,
        X_train: torch.Tensor,
        per_model_batch_size: int = 2048,
        per_model_epochs: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ):
        """
        :param X_train: Tensor of shape [N_train, input_dim]
        """
        X_train = X_train.to(self.device)
        n = X_train.shape[0]

        # train each submodel on its own feature subset
        for i, model in enumerate(self.models):
            idx = self.feature_sets[i]
            x_sub = X_train[:, idx]  # shape [N, len(idx)]
            if verbose:
                print(f"[train] submodel {i+1}/{self.ensemble_size}  features={len(idx)}  epochs={per_model_epochs}")
            self._train_single(
                model,
                x_sub,
                batch_size=per_model_batch_size,
                epochs=per_model_epochs,
                lr=lr,
                weight_decay=weight_decay,
                verbose=verbose,
            )

            # compute q_i: mean over training outputs (should be approx 1)
            model.eval()
            with torch.no_grad():
                # compute outputs in one pass (batching if necessary)
                # we'll reuse DataLoader pattern to avoid memory issues
                ds = TensorDataset(x_sub)
                loader = DataLoader(ds, batch_size=per_model_batch_size, shuffle=False)
                outs = []
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    out = model(xb)  # [batch, 1]
                    outs.append(out.detach().cpu())
                outs = torch.cat(outs, dim=0)  # [N, 1] on CPU
                qi = outs.mean(dim=0).to(self.device)  # scalar tensor on device
                self.qis[i] = qi.squeeze()

        if verbose:
            qis_cpu = [float(qi.cpu().item()) for qi in self.qis]
            print("[fit] qis (per model means) summary: min {:.4f}, max {:.4f}, mean {:.4f}".format(
                min(qis_cpu), max(qis_cpu), sum(qis_cpu)/len(qis_cpu)
            ))

    # ---------------------------
    # Scoring function: computes score_F(x) as in equation (5):
    #   score_F(x) = (1/|F|) * sum_i ( || f_i(x) - q_i ||^power )
    # For scalar outputs, ||·|| can be absolute value; we implement abs(...)**power.
    # Returns a tensor of shape [batch, 1] with scores.
    # ---------------------------
    def score(self, X: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        :param X: tensor [B, input_dim] (or [N, input_dim])
        :return: scores tensor [B, 1]
        """
        X = X.to(self.device)
        B = X.shape[0]
        device = self.device

        # We'll compute outputs model-by-model and accumulate powered deviations.
        # To save memory, iterate models; each model processes only its feature subset.
        acc = torch.zeros(B, device=device)  # will accumulate sum_i |f_i - q_i|^power

        # compute per-model outputs in batches for big inputs
        for i, model in enumerate(self.models):
            idx = self.feature_sets[i]
            x_sub = X[:, idx]  # [B, len(idx)]
            # forward (allow model to handle full batch)
            model.eval()
            with torch.no_grad():
                out = model(x_sub).squeeze(dim=1)  # [B]
                qi = self.qis[i]  # scalar tensor on device
                # absolute deviation then power
                dev = torch.abs(out - qi) ** self.power  # [B]
                acc += dev

        # average over models (1/|F|)
        scores = acc / float(self.ensemble_size)
        return scores.view(-1, 1)  # [B,1]

    # ---------------------------
    # Utility: convenience function to score single numpy / torch sample
    # ---------------------------
    def score_numpy(self, X_np):
        t = torch.from_numpy(X_np).float()
        return self.score(t).cpu().numpy()



if __name__ == "__main__":
    # Example: synthetic dataset
    N_train = 5000
    input_dim = 100
    X_train = torch.randn(N_train, input_dim)  # normal training data

    dean = DeanEnsemble(
        input_dim=input_dim,
        ensemble_size=80,
        hidden_dims=[64, 32],
        feature_bagging=True,
        bagging_ratio=0.3,
        power=3.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=123,
    )

    # Fit: smaller lr and reasonably large batch size (as recommended)
    dean.fit(
        X_train=X_train,
        per_model_batch_size=2048,   # relatively large batch
        per_model_epochs=6,         # a few epochs - training each model is quick
        lr=1e-4,                    # lower-than-standard LR
        weight_decay=0.0,
        verbose=True,
    )

    # Score some samples (normal)
    X_test = torch.randn(10, input_dim)
    scores = dean.score(X_test)  # shape [10,1]
    print("scores:", scores.squeeze().tolist())

