"""Calculate network-based features from regional deviations."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def calculate_neighborhood_deviation_for_subject(subject_data, connect_matrix):
    """Calculate neighbor-weighted deviations for one subject.

    subject_data has shape (n_regions,) and connect_matrix has shape
    (n_regions, n_regions), using the same ROI order.
    Only positive off-diagonal connections define neighbors. For each ROI,
    sum(weight * neighbor deviation) is divided by the number of neighbors,
    not the sum of weights. ROIs without neighbors receive NaN."""
    numRegions = subject_data.shape[0]
    neighbor_weighted_avg = np.zeros(numRegions)

    for i in range(numRegions):

        neighbor_indices = np.where((connect_matrix[i, :] > 0) &
                                    (np.arange(numRegions) != i))[0]

        N_i = len(neighbor_indices)

        if N_i > 0:

            weighted_sum = 0
            for j in neighbor_indices:
                p_j = subject_data[j]
                C_ij = connect_matrix[i, j]
                weighted_sum += p_j * C_ij

            neighbor_weighted_avg[i] = weighted_sum / N_i
        else:

            neighbor_weighted_avg[i] = np.nan

    return neighbor_weighted_avg

def calculate_neighbor_correlation_for_subject(subject_data, structural_connectivity_matrix):
    """Return the Spearman correlation and p-value for one subject.

    Correlate ROI deviations with their neighbor-weighted deviations,
    excluding ROIs with NaN neighbor values. Return (NaN, NaN) when fewer
    than two valid ROIs remain."""

    neighbor_weighted_avg = calculate_neighborhood_deviation_for_subject(
        subject_data, structural_connectivity_matrix
    )

    valid_indices = ~np.isnan(neighbor_weighted_avg)

    if np.sum(valid_indices) < 2:

        return np.nan, np.nan

    valid_subject_data = subject_data[valid_indices]
    valid_neighbor_avg = neighbor_weighted_avg[valid_indices]

    correlation, p_value = spearmanr(valid_subject_data, valid_neighbor_avg)

    return correlation, p_value

def calculate_network_correlations_for_group(network_name, network_matrix, group_data, group_name, subject_indices, group_labels):
    """Calculate one network's correlations for all subjects in a group.

    group_data has shape (n_subjects, n_regions). subject_indices contains
    subject IDs in the same row order. Return a DataFrame with subject IDs,
    group information, Spearman correlations, and p-values."""

    group_label = group_labels.get(group_name, f'Group {group_name}')

    num_subjects_in_group = group_data.shape[0]
    correlations = np.zeros(num_subjects_in_group)
    p_values = np.zeros(num_subjects_in_group)

    for i, sub_idx in enumerate(range(num_subjects_in_group)):
        correlation, p_value = calculate_neighbor_correlation_for_subject(
            group_data[sub_idx, :], network_matrix
        )
        correlations[i] = correlation
        p_values[i] = p_value

    result_df = pd.DataFrame({
        'SubjectID': subject_indices,
        'group': group_name,
        'group_label': group_label,
        f'{network_name}_correlation': correlations,
        f'{network_name}_pvalue': p_values
    })

    return result_df
