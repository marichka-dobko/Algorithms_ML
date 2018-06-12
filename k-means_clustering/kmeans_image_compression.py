import numpy as np
import matplotlib.image as img
import scipy.misc

INPUT_IMAGE_FILE = "mandrill-large.png"
NUM_COLORS = NUM_CLUSTERS = 16
RANDOM_SEED = 42


def get_closest_centroids(X, centroids):
    # Find the index of the closest centroid for each data point. The function should return np.array.
    # 𝑐(𝑖) = arg min𝑗‖ 𝑥(𝑖) − 𝜇𝑗 ‖^2
    closest = np.array([np.linalg.norm(X - c, axis=1) for c in centroids]).argmin(axis=0)

    return closest


def move_centroids(X, closest_centroids, num_clusters):
    # Recompute the coordinates of each centroid. The function should return np.array.
    new_centroids = [X[np.where(closest_centroids == k)[0]].mean(axis=0) for k in range(num_clusters)]
    return np.array(new_centroids)


def kmeans_objective(X, centroids, closest_centroids):
    # Compute the K-Means objective function.
    matrix = np.empty((closest_centroids.shape[0], 3))
    for i, el in enumerate(closest_centroids):
        matrix[i] = centroids[el]

    return np.sum((X - matrix) ** 2)


def main():
    # Read image, which will be compressed
    input_img = img.imread(INPUT_IMAGE_FILE)
    color_depth = input_img.shape[-1]
    X = input_img.reshape(-1, color_depth)

    np.random.seed(RANDOM_SEED)

    # Initialize centroids
    centroids = X[np.random.choice(np.arange(X.shape[0]), size=NUM_CLUSTERS, replace=False)]

    closest_centroids = np.zeros(len(X))

    objective_history = [1]
    convergence = False
    iteration = 0
    epsilon = 1

    while not convergence:
        # Run k-means iteration until convergence.
        closest_centroids = get_closest_centroids(X, centroids)
        new_centroids = move_centroids(X, closest_centroids, NUM_CLUSTERS)

        # Compute the objective.
        objective = kmeans_objective(X, new_centroids, closest_centroids)
        if np.abs(objective_history[-1] - objective) < epsilon:
            convergence = True
        objective_history.append(objective)

        # Update centroids
        centroids = new_centroids

        # Increase iteration counter
        iteration += 1

        print("Iteration: {0:2d}    Objective: {1:.3f}".format(iteration, objective))
    objective_history = objective_history[1:]

    output_img = centroids[closest_centroids].reshape(input_img.shape)
    scipy.misc.imsave('output_file.jpg', output_img)


if __name__ == '__main__':
    main()
