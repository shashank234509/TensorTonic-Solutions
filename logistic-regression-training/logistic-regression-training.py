import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X=np.asarray(X,float)
    Y=np.asarray(y,float)

    R,C=X.shape
    W=np.zeros(C)
    B=0.0

    for _ in range(steps):
        logit=X@W + B
        P=_sigmoid(logit)
        grad_w=X.T@(P-Y)/R
        grad_b=np.mean(P-Y)

        W-=lr*grad_w
        B-=lr*grad_b

    return W,B
    
