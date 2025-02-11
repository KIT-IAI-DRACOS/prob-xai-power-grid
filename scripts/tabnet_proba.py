import torch
import numpy as np
from pytorch_tabnet.tab_model import filter_weights, TabModel


class TabNetRegressorProba(TabModel):
    """
    Probabilistic version of the TabNet regressor model.
    """

    def __post_init__(self):
        super(TabNetRegressorProba, self).__post_init__()
        self._task = "regression"
        self._default_loss = torch.nn.GaussianNLLLoss  # torch.nn.functional.mse_loss
        self._default_metric = "mse"
        self.eps = 1e-6  # add an epsilon term for numerical stability

    def prepare_target(self, y):
        return y

    # negative log likelihood as loss for the probabilistic model
    def compute_loss(self, y_pred, y_true):
        mean = y_pred[:, 0].reshape(-1, 1)
        cov = y_pred[:, 1].reshape(-1, 1)
        cov = cov**2 + self.eps
        assert mean.shape == y_true.shape, f"{mean.shape} != {y_true.shape}"
        loss = torch.nn.GaussianNLLLoss()
        return loss(mean, y_true, cov)

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        if len(y_train.shape) != 2:
            msg = (
                "Targets should be 2D : (n_samples, n_regression) "
                + f"but y_train.shape={y_train.shape} given.\n"
                + "Use reshape(-1, 1) for single regression."
            )
            raise ValueError(msg)
        self.output_dim = 2  # y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        outputs[:, 1] = outputs[:, 1] ** 2 + self.eps
        return outputs

    def predict_mean(self, X_test):
        return self.predict(X_test)[:,0]
    
    def predict_std(self, X_test):
        return self.predict(X_test)[:,1]

    """
        def predict_std(self, outputs):
            return outputs[:, 1] ** 2 + self.eps
    """

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
