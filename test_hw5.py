"""Test script for HW5"""
import numpy as np
import unittest


class A_ArgumentTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = np.array(
            [[0.7, 0.9], [0.1, 0.7], [0.2, 0.8], [0.0, 0.1], [0.5, 0.0], [0.6, 0.6]]
        )
        self.weights = np.array([1, 2])
        self.y = np.array([2.5, 1.5, 1.8, 0.2, 0.5, 1.8])

    def test_010_fit_arguments_LR(self):
        from hw5 import LinearRegression

        lr = LinearRegression(0)
        fit_return = lr.fit(self.X, self.y)

        self.assertEqual(
            fit_return,
            None,
            f"fit() method should return None, but got: {type(fit_return)}",
        )
        self.assertEqual(
            type(lr.coefs),
            np.ndarray,
            f"fit() method should assign np.ndarray to self.coefs but assigned: {type(lr.coefs)}",
        )
        self.assertEqual(
            type(lr.intercept),
            np.float64,
            f"fit() method should assign float to self.intercept but assigned: {type(lr.intercept)}",
        )

    def test_020_fit_arguments_LR_GD(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(0)
        fit_return = lr.fit(self.X, self.y)

        self.assertEqual(
            fit_return,
            None,
            f"fit() method should return None, but got: {type(fit_return)}",
        )
        self.assertEqual(
            type(lr.coefs),
            np.ndarray,
            f"fit() method should assign np.ndarray to self.coefs but assigned: {type(lr.coefs)}",
        )
        self.assertEqual(
            type(lr.intercept),
            np.float64,
            f"fit() method should assign float to self.intercept but assigned: {type(lr.intercept)}",
        )


class B_InterceptTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = np.array(
            [[0.7, 0.9], [0.1, 0.7], [0.2, 0.8], [0.0, 0.1], [0.5, 0.0], [0.6, 0.6]]
        )
        self.intercept = 5.0
        self.weights = np.array([1.0, 2.0])
        self.y = np.array([7.5, 6.5, 6.8, 5.2, 5.5, 6.8])

        # without regularization
        self.X_test = np.array([[0.8, 0.5], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]])
        self.y_test = np.array([6.8, 5.7, 6.5, 6.2])

        # with regularization
        self.weights_reg = {
            1: np.array([0.40046129, 0.87664647]),
            10: np.array([0.06390273, 0.1440971]),
        }
        self.intercept_reg = {1: 5.790237870282883, 10: 6.286517209960804}
        self.y_test_reg = {
            1: np.array([6.54893014, 6.08570555, 6.41364697, 6.30108098]),
            10: np.array([6.40968794, 6.33450745, 6.38725879, 6.36971714]),
        }

    def test_110_regularized_intercept_LR(self):
        from hw5 import LinearRegression

        lr = LinearRegression(1)

        lr.fit(self.X, self.y)

        intercept = lr.intercept

        if intercept < self.intercept:
            raise ValueError(
                f"Check your implementation. Seems like your intercept is regularized. Think about how to remove it from regularization."
            )

    def test_120_regularized_intercept_LR_GD(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(1)

        lr.fit(self.X, self.y)

        intercept = lr.intercept

        if intercept < self.intercept:
            raise ValueError(
                f"Check your implementation. Seems like your intercept is regularized. Think about how to remove it from regularization."
            )

    def test_121_unregularized_coefs_LR_GD(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(0)
        lr.fit(self.X, self.y)
        coefs_noreg = lr.coefs.copy()

        np.testing.assert_almost_equal(
            coefs_noreg,
            self.weights,
            3,
            f"Gradient LR seem to produce different results than expected. If close, try adjusting the threshold for convergence.",
        )

    def test_122_regularized_coefs_LR_GD(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(0)
        lr.fit(self.X, self.y)
        coefs_noreg = lr.coefs.copy()

        lr = LinearRegressionGD(1)
        lr.fit(self.X, self.y)
        coefs_reg = lr.coefs.copy()

        np.testing.assert_almost_equal(
            coefs_noreg,
            self.weights,
            3,
            f"Gradient LR seem to produce different results than expected. If close, try adjusting the threshold for convergence.",
        )

        if np.isclose(coefs_noreg, coefs_reg, 1e-4).sum() > 0:
            raise ValueError(
                f"Regularized and unregularized coefficients are close. Check if your derivative is correct."
            )

        np.testing.assert_almost_equal(
            coefs_reg,
            self.weights_reg[1],
            5,
            f"Regularized Gradient LR seem to produce different results than expected. If close, try adjusting the threshold for convergence or check your gradient for errors (if completely off).",
        )


class C_SimpleTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = np.array(
            [[0.7, 0.9], [0.1, 0.7], [0.2, 0.8], [0.0, 0.1], [0.5, 0.0], [0.6, 0.6]]
        )
        self.intercept = 5.0
        self.weights = np.array([1.0, 2.0])
        self.y = np.array([7.5, 6.5, 6.8, 5.2, 5.5, 6.8])

        # without regularization
        self.X_test = np.array([[0.8, 0.5], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]])
        self.y_test = np.array([6.8, 5.7, 6.5, 6.2])

        # with regularization
        self.weights_reg = {
            1: np.array([0.40046129, 0.87664647]),
            10: np.array([0.06390273, 0.1440971]),
        }
        self.intercept_reg = {1: 5.790237870282883, 10: 6.286517209960804}
        self.y_test_reg = {
            1: np.array([6.54893014, 6.08570555, 6.41364697, 6.30108098]),
            10: np.array([6.40968794, 6.33450745, 6.38725879, 6.36971714]),
        }

    def test_210_no_regularization_correct_fit(self):
        from hw5 import LinearRegression

        lr = LinearRegression(0)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, self.weights, decimal=5)
        np.testing.assert_almost_equal(fit_intercept, self.intercept, decimal=5)

    def test_211_no_regularization_correct_predict(self):
        from hw5 import LinearRegression

        lr = LinearRegression(0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test, decimal=5)

    def test_220_regularization_1_correct_fit(self):
        from hw5 import LinearRegression

        lr = LinearRegression(1.0)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, self.weights_reg[1], decimal=5)
        np.testing.assert_almost_equal(fit_intercept, self.intercept_reg[1], decimal=5)

    def test_221_regularization_1_correct_prediction(self):
        from hw5 import LinearRegression

        lr = LinearRegression(1.0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test_reg[1], decimal=5)

    def test_230_regularization_10_correct_fit(self):
        from hw5 import LinearRegression

        lr = LinearRegression(10.0)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, self.weights_reg[10], decimal=5)
        np.testing.assert_almost_equal(fit_intercept, self.intercept_reg[10], decimal=5)

    def test_231_regularization_10_correct_prediction(self):
        from hw5 import LinearRegression

        lr = LinearRegression(10.0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test_reg[10], decimal=5)

    def test_240_regularization_1e9_correct_fit(self):
        from hw5 import LinearRegression

        lr = LinearRegression(1e9)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, np.array([0.0, 0.0]), decimal=5)
        np.testing.assert_almost_equal(fit_intercept, 6.383333332291889, decimal=5)

    def test_231_regularization_1e9_correct_prediction(self):
        from hw5 import LinearRegression

        lr = LinearRegression(1e9)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(
            y_pred, np.array([6.383333332291889] * 4), decimal=5
        )


class F_BestLambdaParameterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lambdas = np.array(
            [0, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 2, 3, 4, 5, 10, 20, 50, 100]
        )

    def print_lambda_choice(self, text, n_samples, noise, best_lambda, correct):

        is_correct = "correct" if correct else "wrong"
        print(
            f"""
When choosing the best lambda on the dataset with
    {text}
n_samples: {n_samples}
noise:     {noise}
You chose lambda: {best_lambda} ({is_correct})
Is this expected? Why?
        """
        )

    def get_data(self, n_samples: int, noise_spread: float) -> tuple:

        np.random.seed(0)
        X = np.abs(np.random.normal(1, 3, (n_samples, 50)))
        y = (
            5
            + 0.1 * X[:, 0]
            - X[:, 2]
            + 10 * X[:, 7]
            + np.random.normal(0, noise_spread, n_samples)
        )

        HALF = n_samples // 2

        return (X[:HALF], X[HALF:], y[:HALF], y[HALF:])

    def test_510_enough_samples_no_noise(self):
        from hw5 import find_best_lambda

        n_samples = 200
        noise = 0
        correct_lambda = 0

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "enough samples and no noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_511_enough_samples_a_bit_noise(self):
        from hw5 import find_best_lambda

        n_samples = 200
        noise = 3
        correct_lambda = 1

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "enough samples and a bit of noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_512_enough_samples_some_noise(self):
        from hw5 import find_best_lambda

        n_samples = 200
        noise = 4
        correct_lambda = 4

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "enough samples and some noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_513_enough_samples_a_lot_noise(self):
        from hw5 import find_best_lambda

        n_samples = 200
        noise = 7
        correct_lambda = 20

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "enough samples and a lot of noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_520_few_samples_no_noise(self):
        from hw5 import find_best_lambda

        n_samples = 100
        noise = 0
        correct_lambda = 0.01

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "few samples and no noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_521_few_samples_a_bit_noise(self):
        from hw5 import find_best_lambda

        n_samples = 100
        noise = 3
        correct_lambda = 10

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "few samples and a bit of noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_522_few_samples_some_noise(self):
        from hw5 import find_best_lambda

        n_samples = 100
        noise = 4
        correct_lambda = 20

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "few samples and some noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)

    def test_523_few_samples_a_lot_noise(self):
        from hw5 import find_best_lambda

        n_samples = 100
        noise = 7
        correct_lambda = 50

        X_train, X_test, y_train, y_test = self.get_data(n_samples, noise)

        best_lambda = find_best_lambda(X_train, y_train, X_test, y_test, self.lambdas)

        self.print_lambda_choice(
            "few samples and a lot of noise",
            n_samples,
            noise,
            best_lambda,
            correct_lambda == best_lambda,
        )

        self.assertEqual(best_lambda, correct_lambda)


class D_CostFunctionTest(unittest.TestCase):
    def test_310_cost_grad_lambda_0(self):
        from hw5 import gradient, cost
        from test_vars import X, y

        X = np.array(X)
        y = np.array(y)

        _, cols = X.shape
        theta0 = np.ones(cols)

        grad = gradient(X, y, theta0, 0)

        def cost_(theta):
            return cost(X, y, theta, 0)

        eps = 10 ** -4
        theta0_ = theta0
        grad_num = np.zeros(grad.shape)
        for i in range(grad.size):
            theta0_[i] += eps
            h = cost_(theta0_)
            theta0_[i] -= 2 * eps
            l = cost_(theta0_)
            theta0_[i] += eps
            grad_num[i] = (h - l) / (2 * eps)

        np.testing.assert_almost_equal(grad, grad_num, decimal=3)

    def test_320_cost_grad_lambda_1(self):
        from hw5 import gradient, cost
        from test_vars import X, y

        X = np.array(X)
        y = np.array(y)

        _, cols = X.shape
        theta0 = np.ones(cols)

        grad = gradient(X, y, theta0, 1)

        def cost_(theta):
            return cost(X, y, theta, 1)

        eps = 10 ** -4
        theta0_ = theta0
        grad_num = np.zeros(grad.shape)
        for i in range(grad.size):
            theta0_[i] += eps
            h = cost_(theta0_)
            theta0_[i] -= 2 * eps
            l = cost_(theta0_)
            theta0_[i] += eps
            grad_num[i] = (h - l) / (2 * eps)

        np.testing.assert_almost_equal(grad, grad_num, decimal=3)


class E_LinRegGDTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = np.array(
            [[0.7, 0.9], [0.1, 0.7], [0.2, 0.8], [0.0, 0.1], [0.5, 0.0], [0.6, 0.6]]
        )
        self.intercept = 5.0
        self.weights = np.array([1.0, 2.0])
        self.y = np.array([7.5, 6.5, 6.8, 5.2, 5.5, 6.8])

        # without regularization
        self.X_test = np.array([[0.8, 0.5], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]])
        self.y_test = np.array([6.8, 5.7, 6.5, 6.2])

        # with regularization
        self.weights_reg = {
            1: np.array([0.40046129, 0.87664647]),
            10: np.array([0.06390273, 0.1440971]),
        }
        self.intercept_reg = {1: 5.790237870282883, 10: 6.286517209960804}
        self.y_test_reg = {
            1: np.array([6.54893014, 6.08570555, 6.41364697, 6.30108098]),
            10: np.array([6.40968794, 6.33450745, 6.38725879, 6.36971714]),
        }

    def test_410_GD_no_regularization_correct_fit(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(0)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, self.weights, decimal=5)
        np.testing.assert_almost_equal(fit_intercept, self.intercept, decimal=5)

    def test_411_GD_no_regularization_correct_predict(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test, decimal=5)

    def test_420_GD_regularization_1_correct_fit(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(1.0)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, self.weights_reg[1], decimal=5)
        np.testing.assert_almost_equal(fit_intercept, self.intercept_reg[1], decimal=5)

    def test_421_GD_regularization_1_correct_prediction(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(1.0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test_reg[1], decimal=5)

    def test_430_GD_regularization_10_correct_fit(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(10.0)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_weights, self.weights_reg[10], decimal=5)
        np.testing.assert_almost_equal(fit_intercept, self.intercept_reg[10], decimal=5)

    def test_431_GD_regularization_10_correct_prediction(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(10.0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test_reg[10], decimal=5)

    def test_440_GD_regularization_200_correct_fit(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(200)
        lr.fit(self.X, self.y)

        fit_weights = lr.coefs
        fit_intercept = lr.intercept

        if np.isnan(fit_intercept):
            suggestion = (
                "Think about the function you are trying to optimize and print out the gradient in each step."
                + "Perhaps you should adjust your learning rate in each step when regularization parameter is high."
            )
            raise ValueError(
                f"Algorithm did not converge, which resulted in intercept: {fit_intercept}.\n{suggestion}"
            )

        np.testing.assert_almost_equal(fit_weights, np.array([0.0, 0.0]), decimal=1)
        np.testing.assert_almost_equal(fit_intercept, 6.383333332291889, decimal=1)

    def test_441_GD_regularization_200_correct_prediction(self):
        from hw5 import LinearRegressionGD

        lr = LinearRegressionGD(200)
        lr.fit(self.X, self.y)

        fit_intercept = lr.intercept

        if np.isnan(fit_intercept):
            suggestion = (
                "Think about the function you are trying to optimize and print out the gradient in each step."
                + "Perhaps you should adjust your learning rate in each step when regularization parameter is high."
            )
            raise ValueError(
                f"Algorithm did not converge, which resulted in intercept: {fit_intercept}.\n{suggestion}"
            )

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, np.array([6.3833] * 4), decimal=1)


if __name__ == "__main__":
    unittest.main()
