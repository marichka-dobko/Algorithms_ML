import numpy as np


class WeightedLinearRegression():
    def __init__(self, X_train, y_train, X_pred):
        self.data = X_train
        self.labels = y_train
        self.test_data = X_pred

    def predict_linear(self, theta, X):
        # Compute the hypothesis function for linear regression.
        h = X.dot(theta)  # sigmoid function
        return h

    def get_example_weights(self, X, x_pred, tau):
        # Compute the weight for each example, given the
        # prediction point (x_pred).
        weights = [0 for el in range(len(X))]

        for i in range(len(X)):
            x_diff = X[i] - x_pred
            weights[i] = np.exp(-1 * (np.dot(x_diff.T, x_diff)) / (2 * tau ** 2))

        weights = np.array(weights)

        return weights

    def cost_function(self, theta, X, y, weights):
        # Given the currently learned model weights (theta),
        # compute the overall loss on the training set (X),
        # taking the weights into account.
        m = y.size
        J = np.sum((weights * (np.square(self.predict_linear(theta, X) - y)))) / (2 * m)

        return J

    def cost_function_gradient(self, theta, X, y, weights):
        # Given the currently learned model weights (theta),
        # compute the gradient of the cost function on the
        # training set (X), taking the weights into account.
        # gradient= (ℎ𝜃(𝑥) − 𝑦) ⋅ 𝑥 *W

        cost_grad = np.multiply((self.predict_linear(theta, X) - y), X.T).dot(weights)

        return cost_grad

    def update_model_weights(self, theta, learning_rate, cost_gradient):
        # Given the learning rate and the gradient of the
        # cost function, take one gradient descent step and
        # return the updated vector theta.
        theta = theta - learning_rate * cost_gradient

        return theta

    def gradient_descent(self, X, y, weights, loss_fun, grad_fun, learning_rate, convergence_threshold, max_iters,
                         verbose=False):
        theta = np.zeros(X.shape[1])
        losses = []
        for i in range(max_iters):

            loss = loss_fun(theta, X, y, weights)
            losses.append(loss)

            if verbose:
                print("Iteration: {0:3} Loss: {1}".format(i + 1, loss))

            if len(losses) > 2 and np.abs(losses[-1] - losses[-2]) <= convergence_threshold:
                break

            grad = grad_fun(theta, X, y, weights)
            theta = self.update_model_weights(theta, learning_rate, grad)

        return theta, np.array(losses)

    def predict_weighted_linear(self, verbose=False):
        # x_predict = x_pred[-1]
        weights = self.get_example_weights(self.data, np.array(self.test_data), tau=0.1)
        theta, losses = self.gradient_descent(
            self.data,
            self.labels,
            weights,
            loss_fun=self.cost_function,
            grad_fun=self.cost_function_gradient,
            learning_rate=0.005,
            convergence_threshold=0.0001,
            max_iters=500,
            verbose=verbose
        )
        # all_pred = [self.predict_linear(theta, np.array(x_pred_i))[0] for x_pred_i in x_pred]
        return self.predict_linear(theta, np.array(self.test_data))

# wlr = WeightedLinearRegression(X_train, y_train, X_pred)
# predicted = wlr.predict_weighted_linear()
