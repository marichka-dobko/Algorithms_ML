import scipy.io.wavfile
import numpy as np
import copy

# Example of cocktail party problem


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unmixer(X, W):
    """
    :param X: data
    :param W: empty identity matrix
    :return: matrix for unmixed sounds
    """
    alpha = 0.01
    likelihoods, loss_history = [], [0]
    convergence = False
    best_l, iteration = 0, 0
    W_last = np.identity(5)
    i = 0

    while not convergence:
        l_val = []
        for xi in X:
            W += alpha * (np.outer(1 - 2 * sigmoid(np.dot(W, xi.T)), xi) + np.linalg.inv(W.T))
            likelihood = np.nansum(np.log(np.diff(sigmoid(W.T.dot(xi))))) + np.log(np.linalg.det(W))
            l_val.append(likelihood)
            iteration += 1
        i += 1
        print(i, np.linalg.norm(W - W_last))

        if np.abs(np.linalg.norm(W - W_last) - loss_history[-1]) < 0.001:
            convergence = True
        else:
            loss_history.append(np.linalg.norm(W - W_last))
            W_last = copy.deepcopy(W)

    print("Num of iterations: ", iteration)

    return W

dataset = []
for i in range(1,6):
    sample_rate, wav_data = scipy.io.wavfile.read('mixed/mix'+str(i)+'.wav')
    dataset.append(wav_data)

dataset = np.array(dataset).T
maxs = np.max(np.abs(dataset), axis=0).astype(np.int64)
data_normalized = 0.99 * dataset / maxs

W = np.identity(5)

W = unmixer(data_normalized, W)

unmixed = np.dot(data_normalized, W.T)
maxs = np.max(np.abs(unmixed), axis=0).astype(np.int64)
unmixed_normalized = (0.99 * np.abs(unmixed) / maxs).T

for i in range(unmixed_normalized.shape[0]):
    track = unmixed[:,i]
    scipy.io.wavfile.write('unmixed/unmixed'+str(i)+'.wav', sample_rate, track)
