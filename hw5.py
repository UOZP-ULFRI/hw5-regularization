import numpy as np


class LinearRegression:
    def __init__(self, l2_regularization_coefficient: float = 0):
        self.l2_lambda = l2_regularization_coefficient
        self.coefs = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """
        Implement a fit method that uses a closed form solution to calculate coeficients and the intercept.
        Use vectorized operations to calculate weights. Do not forget about the intercept.
        Assign coeficients (np.array) to the self.coefs variable and intercept (float) to the self.intercept variable.

        Arguments
        ---------
        X: np.ndarray
            Array of features with rows as samples and features as columns.
        y: np.array
            Array of response variables with length of samples.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.array:
        """
        Make prediction using learned coeficients and the intercept.
        Use vectorized operations.

        Arguments
        ---------
        X: np.ndarray
            Test data with rows as samples and columns as features

        Returns
        -------
        np.array
            Predicted values for given samples.
        """
        raise NotImplementedError


## You can resuse your code from HW4 and adjust it for l2_lambda parameter
def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2_lambda: float) -> float:
    """
    A cost function in matrix/vector form. Stick to the notation from instructions to
    keep the lambda parameter equivalent between implementations.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters
    l2_lambda: float
        L2 regularization parameter

    Returns
    -------
    float
        The value of the cost function
    """
    raise NotImplementedError


def gradient(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2_lambda: float
) -> np.ndarray:
    """Gradient of cost function in matrix/vector form.
    Stick to the notation from instructions to
    keep the lambda parameter equivalent between implementations.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters
    l2_lambda: float
        L2 regularization parameter

    Returns
    -------
    numpy array of shape (n_features,)
    """
    raise NotImplementedError


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr=0.01,
    l2_lambda: float = 0.0,
    tol=1e-5,
    max_iter=1000,
):
    """Implementation of gradient descent.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    lr: float
        The learning rate.
    l2_lambda: float
        L2 regularization parameter.
    tol: float
        The stopping criterion (tolerance).
    max_iter: int
        The maximum number of passes (aka epochs).

    Returns
    -------
    numpy array of shape (n_features,)
    """
    raise NotImplementedError


class LinearRegressionGD:
    def __init__(self, l2_regularization_coefficient: float = 0) -> None:
        self.coefs = None
        self.intercept = None
        self.l2_lambda = l2_regularization_coefficient

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        The fit method of LinearRegression accepts X and y
        as input and save the coefficients of the linear model.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
            Training data
        y: numpy array of shape (n_samples,)
            Target values

        Returns
        -------
        None
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
            New samples

        Returns
        -------
        numpy array of shape (n_samples,)
            Returns predicted values.
        """
        raise NotImplementedError


## exercise 4 - new code
def find_best_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    candidate_lambdas: np.ndarray = np.array([0, 0.01, 1, 10]),
):
    """
    Find the optimal L2 lambda parameter on train and validation dataset.
    For each lambda, instantiate a new LinearRegression model with that parameter,
    train it on the train set and evaluate performance with MSE on the validation set.
    Return lambda with the best MSE on the validation set.

    Arguments
    ---------
    X_train: np.ndarray
        Train dataset with rows as samples and columns as features.
    y_train: np.ndarray
        Train target vector with length of validation samples.
    X_validation: np.ndarray
        Validation dataset with rows as samples and columns as features.
    y_validation: np.ndarray
        Validation target vector with length of validation samples.
    candidate_lambdas: np.ndarray
        A list of float values for L2 regularization parameter (lambda).

    Returns
    -------
    float
        The best lambda from candidates according to MSE on validation set.
    """
    raise NotImplementedError
