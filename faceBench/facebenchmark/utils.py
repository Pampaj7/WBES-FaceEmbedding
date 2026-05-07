import numpy as np


def compute_landmark_base_vertex_weights(mm, weight_power, weight_strategy):
    """
    Computes per-vertex weights for a face model based on the distances from the mean face shape.

    The function normalizes the mean face shape using the interocular distance (iod) computed from
    the landmarks corresponding to the left and right eye. Then it computes two types of weights:

    1. weights_min: computed as the inverse of the minimum distance from any landmark to each vertex.
    2. weights_mean: computed as the inverse of the absolute difference between the mean distance
       (averaged over a set of reference landmarks) and a fixed value (0.48).

    Depending on the 'weight_strategy' ('mixed', 'min', or 'mean'), a combination is used, and
    then a transformation is applied according to 'weight_power' ('square' or 'sqrt').

    Parameters
    ----------
    mm : dict
        A dictionary containing keys:
          - 'mean_face_shape': (N x 3) array (or list) of the mean face shape.
          - 'leye_oc_rel_index': index (or indices) for the left eye.
          - 'reye_oc_rel_index': index (or indices) for the right eye.
          - 'lmk_indices': list/array of landmark indices.
    weight_power : str
        How to scale the final weights. Options: 'square' or 'sqrt'.
    weight_strategy : str
        How to combine the two computed weights. Options: 'mixed', 'min', or 'mean'.

    Returns
    -------
    weights : ndarray
        The computed weights for each vertex.
    """

    # Convert mean_face_shape to a NumPy array if necessary.
    Xmean = mm['mean_face_shape']
    if isinstance(Xmean, list):
        Xmean = np.array(Xmean)

    # Get indices for the left and right eyes and landmarks.
    lix = mm['leye_oc_rel_index']
    rix = mm['reye_oc_rel_index']
    lmk_indices = mm['lmk_indices']

    # Normalize the mean face shape by the interocular distance.
    iod = np.linalg.norm(Xmean[lmk_indices[lix], :] - Xmean[lmk_indices[rix], :])
    Xmean = Xmean / iod
    # Extract the landmark coordinates from the normalized mean face shape.
    Xmean_lmks = Xmean[lmk_indices, :]

    Nl = len(lmk_indices)

    # Compute distances from each landmark to every vertex in the face shape.
    # adists: shape (Nl, number_of_vertices)
    adists = np.sqrt(np.sum((Xmean_lmks[:, None, :] - Xmean[None, :, :]) ** 2, axis=2))
    # Enforce a minimum threshold (dth = 0.01) on each computed distance.
    dth = 0.01
    adists[adists < dth] = dth
    # For each vertex, take the minimum distance from any landmark.
    madists = adists.min(axis=0)
    weights_min = 1.0 / madists

    # Define reference indices (if exactly 51 landmarks are available)
    ref_lis = np.array([0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30, 19, 22, 25, 28, 13,
                        14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50])
    if Nl == 51:
        # Use only the reference landmarks if available.
        indices = np.array(lmk_indices)[ref_lis]
    else:
        indices = np.array(lmk_indices)
    # Compute distances from each reference landmark to every vertex
    adists2 = np.sqrt(np.sum((Xmean[indices, None, :] - Xmean[None, :, :]) ** 2, axis=2))
    # For each vertex, compute the mean distance from the selected landmarks.
    mean_dists = np.mean(adists2, axis=0)
    weights_mean = 1.0 / np.abs(mean_dists - 0.48)

    # Combine the weights according to the strategy.
    if weight_strategy == 'mixed':
        weights = (weights_mean + weights_min) / 2.0
    elif weight_strategy == 'min':
        weights = weights_min
    elif weight_strategy == 'mean':
        weights = weights_mean
    else:
        raise ValueError("Invalid weight_strategy. Choose 'mixed', 'min', or 'mean'.")

    # Ensure that all weights are at least 1.
    weights[weights < 1] = 1

    # Modify weights based on weight_power.
    if weight_power == 'square':
        # Scale weights by squaring, normalized by the mean.
        weights = (weights ** 2) / (np.mean(weights ** 2) / np.mean(weights))
    elif weight_power == 'sqrt':
        # Take the square root of the weights.
        weights = np.sqrt(weights)
    else:
        raise ValueError("Invalid weight_power. Choose 'square' or 'sqrt'.")

    return weights