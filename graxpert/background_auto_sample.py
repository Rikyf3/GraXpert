import astropy.stats
import numpy as np
from auto_stretch import apply_stretch
from scipy.spatial.distance import cdist
from astropy.stats import biweight_midvariance


def intensity_based_sampling(n_points, image, window_size=100):
    epsilon = 1e-6
    height, width = image.shape
    padded_image = np.pad(image, window_size // 2, mode='reflect')  # Pad the image to handle edges
    median_image = np.zeros_like(image, dtype=float)

    # Calculate median intensity for each window
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            y_end = min(y + window_size, height)  # Handle windows at the image edges
            x_end = min(x + window_size, width)
            window = image[y:y_end, x:x_end]
            median_value = np.median(window)

            # Assign the median value to the entire window in the median_image
            median_image[y:y_end, x:x_end] = median_value

    # Invert and apply power transformation, similar to original method
    inverted_image = 1.0 / (median_image + epsilon)
    inverted_image = np.power(inverted_image, 0.3)

    # Normalize to create probabilities
    probabilities = inverted_image / np.sum(inverted_image)

    # Flatten the probabilities and create a cumulative distribution function (CDF)
    flat_probabilities = probabilities.flatten()
    cdf = np.cumsum(flat_probabilities)

    # Generate random numbers between 0 and 1
    random_values = np.random.rand(n_points)

    # Use the CDF to find the corresponding pixel indices
    indices = np.searchsorted(cdf, random_values)

    # Convert linear indices to (y, x) coordinates
    y_coords, x_coords = np.unravel_index(indices, image.shape)

    return np.stack([y_coords, x_coords], axis=-1)


def fit_linear_background(window, regularization_lambda=0.1):
    # Create coordinate matrices for the window
    window_x, window_y = np.mgrid[0:window.shape[0], 0:window.shape[1]]

    # Flatten the arrays
    x = window_x.flatten()
    y = window_y.flatten()
    z = window.flatten()
    N = len(z)

    # Construct the design matrix A for a linear fit
    A = np.column_stack((x, y, np.ones(N)))

    # Add L2 regularization term
    L2_reg = regularization_lambda * np.identity(A.shape[1])

    # Calculate the closed-form solution using the normal equation with regularization
    ATA_inv = np.linalg.inv(A.T @ A + L2_reg)
    params = ATA_inv @ A.T @ z

    # Reconstruct the background
    background = (params[0] * window_x +
                  params[1] * window_y +
                  params[2])

    window_corrected = window - background

    return window_corrected, background


def calculate_energy(point, image, window_size=75):
    image_shape = image.shape
    x, y = point
    half_window = window_size // 2

    # Extract the window around the point
    x_start = max(0, int(x) - half_window)
    x_end = min(image_shape[0], int(x) + half_window + 1)
    y_start = max(0, int(y) - half_window)
    y_end = min(image_shape[1], int(y) + half_window + 1)

    window = image[x_start:x_end, y_start:y_end]
    window, _ = fit_linear_background(window)

    return biweight_midvariance(window)


def simulated_annealing_points(image, N, T=1.0, cooling_rate=0.99, max_iterations=2000, window_size=75):
    image = apply_stretch(image)
    image_shape = image.shape

    # Initialize random points
    points = intensity_based_sampling(N, image, 4*window_size)
    rows, cols = image_shape
    corner_points = np.array([
        [20, 20],  # Top-left
        [20, cols - 20],  # Top-right
        [rows - 20, 20],  # Bottom-left
        [rows - 20, cols - 20]  # Bottom-right
    ])
    points = np.vstack((points, corner_points))
    points = points.astype(np.float32)

    for iteration in range(max_iterations):
        # Choose a random point to perturb
        idx = np.random.randint(N)
        old_point = points[idx].copy()
        old_energy = calculate_energy(old_point, image, window_size=window_size)

        # Perturb the point
        if T > 0.3:
            points[idx] += np.random.randn(2) * max(image_shape)/100
        else:
            points[idx] += np.random.randn(2) * max(image_shape)/500

        # Clip to image bounds
        points[idx] = np.clip(points[idx], [0, 0], np.array(image_shape) - 1)

        # Calculate new energy
        new_energy = calculate_energy(points[idx], image, window_size=window_size)

        # Acceptance probability
        delta_energy = new_energy - old_energy
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
            pass  # Accept the move
        else:
            points[idx] = old_point  # Reject the move

        # Cool down
        T *= cooling_rate

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Temperature {T:.4f}, delta_energy {delta_energy:.6f}")

    return points


def a_posteriori_filter(points, image, window_size=100, tol=1.0):
    image_median = np.median(image)
    image_mad = astropy.stats.median_absolute_deviation(image)
    filtered_points = []
    half_window = window_size // 2

    for point in points:
        x, y = point
        # Extract the window around the point
        x_start = max(0, int(x) - half_window)
        x_end = min(image.shape[0], int(x) + half_window + 1)
        y_start = max(0, int(y) - half_window)
        y_end = min(image.shape[1], int(y) + half_window + 1)

        window = image[x_start:x_end, y_start:y_end]

        # Calculate the median of the window
        window_median = np.median(window)

        # Keep the point only if the window median is below the image median
        if window_median <= image_median + tol * image_mad:
            filtered_points.append(point)

    return np.array(filtered_points)


def filter_close_points(points, image, min_distance=50, window_size=75):
    distance_matrix = cdist(points, points)
    np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances

    points_to_remove = set()
    for i in range(len(points)):
        if i in points_to_remove:
            continue

        close_points = np.where(distance_matrix[i] < min_distance)[0]
        if len(close_points) > 0:
            all_points = [i] + list(close_points)
            energies = [calculate_energy(points[j], image, window_size=window_size) for j in all_points]
            worst_point_index = np.argmax(energies)

            for j in all_points:
                if j != all_points[worst_point_index]:
                    points_to_remove.add(j)

    filtered_points = np.array([p for i, p in enumerate(points) if i not in points_to_remove])
    return filtered_points


def background_intelligent_selection(image, num_points_rows, tol, window_size):
    if image.shape[-1] == 3:
        image = np.mean(image, axis=-1)
    else:
        image = image[:, :, 0]
    dist = image.shape[1] // num_points_rows
    num_points_cols = image.shape[0] // dist
    num_points = num_points_cols * num_points_rows

    points = simulated_annealing_points(image, num_points, window_size=window_size)
    points = a_posteriori_filter(points, image, tol=tol, window_size=window_size)
    points = filter_close_points(points, apply_stretch(image), min_distance=min(image.shape) * 0.1, window_size=window_size)

    points = np.flip(points, axis=-1)

    result = []
    for i in range(points.shape[0]):
        result.append(np.array([points[i][0], points[i][1], 1], dtype=int))

    return result
