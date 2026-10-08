"""Network-based subtype clustering using W-score outputs.

Configure paths and analysis settings below. Clinical comparisons are optional.
See ../README.md for inputs, methods, and dependencies.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D
from neighbor_correlation import calculate_network_correlations_for_group
from plot_function import create_subtype_network_raincloud_plot

MAIN_DIR = Path(r" ") # change this path when running on another computer.
DATA_DIR = MAIN_DIR / "data"
WSCORE_DIR = MAIN_DIR / "output" / "wscore"
OUTPUT_DIR = MAIN_DIR / "output" / "clustering"
NETWORK_DIR = DATA_DIR / "networks"
RANDOM_STATE = 42
MAX_K = 8
STABILITY_REPEATS = 1000
SAMPLE_FRAC = 0.8

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_pca_subtypes(X_scaled, subject_ids, cluster_series, out_dir, group_name, colors=['#1f77b4', '#d62728', '#2ca02c', '#9467bd']):

    os.makedirs(out_dir, exist_ok=True)

    # ====================
    # PCA
    # ====================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # explained variance
    var_exp = pca.explained_variance_ratio_


    pca_df = pd.DataFrame({
        'SubjectID': subject_ids,
        'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
        'Cluster_Label': cluster_series.values
    })

    # Plot PCA coordinates.
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 18})

    plt.figure(figsize=(5, 5))

    subtype_labels = sorted(
        pca_df['Cluster_Label'].unique(),
        key=lambda x: int(str(x).replace('Subtype', ''))
    )


    for i, label in enumerate(subtype_labels):
        sub_df = pca_df[pca_df['Cluster_Label'] == label]

        plt.scatter(
            sub_df['PC1'],sub_df['PC2'],
            label=label,
            alpha=0.8,s=40,
            color=colors[i % len(colors)]
        )


    # Format axes and labels.
    plt.xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')

    legend_handles = []

    for i, label in enumerate(subtype_labels):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='o', linestyle='',
                markerfacecolor=colors[i % len(colors)],
                markeredgecolor='black',
                color='w', label=label, markersize=7
            )
        )

    plt.tight_layout()

    out_png = os.path.join(out_dir, f'PCA_subtypes_{group_name}.png')
    plt.savefig(out_png, dpi=1000)


    # Save the legend separately.
    fig_leg = plt.figure(figsize=(1.5,0.9))

    legend = fig_leg.legend(
        handles=legend_handles,
        loc='center', frameon=True, handletextpad=0.3,borderpad=0.2,
        fontsize=12, ncol=1
    )

    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(1)
    frame.set_facecolor('white')
    frame.set_alpha(1)
    frame.set_boxstyle('square')

    plt.axis('off')

    legend_png = os.path.join(out_dir,f'PCA_subtypes_{group_name}_legend.png')

    fig_leg.savefig(legend_png,dpi=1000,bbox_inches='tight',facecolor='white')

    plt.close(fig_leg)
    plt.close()

    # Save coordinates for downstream analysis.
    pca_df.to_csv(
        os.path.join(out_dir, f'PCA_coordinates_{group_name}.csv'),
        index=False, encoding='utf-8-sig')

    return pca_df

def compare_subtype_network_features(clustered_data, networks, out_dir, group_name):
    """Compare network features between two subtypes.

    Required columns: Cluster_Label and {network}_correlation."""
    os.makedirs(out_dir, exist_ok=True)

    feature_columns = [f'{net}_correlation' for net in networks.keys()]
    feature_columns = [c for c in feature_columns if c in clustered_data.columns]


    subtype_labels = sorted(
        clustered_data['Cluster_Label'].dropna().unique().tolist(),
        key=lambda x: int(str(x).replace('Subtype', ''))
    )
    subtype1, subtype2 = subtype_labels[0], subtype_labels[1]

    results = []

    for col in feature_columns:
        data1 = pd.to_numeric(
            clustered_data.loc[clustered_data['Cluster_Label'] == subtype1, col],
            errors='coerce'
        ).dropna()

        data2 = pd.to_numeric(
            clustered_data.loc[clustered_data['Cluster_Label'] == subtype2, col],
            errors='coerce'
        ).dropna()

        if len(data1) < 2 or len(data2) < 2:
            results.append({
                'network': col.replace('_correlation', ''), 'feature_col': col,
                'subtype1_mean': np.nanmean(data1) if len(data1) > 0 else np.nan,
                'subtype2_mean': np.nanmean(data2) if len(data2) > 0 else np.nan,
                'subtype1_n': len(data1), 'subtype2_n': len(data2),
                't_stat': np.nan, 'p_value': np.nan,
            })
            continue

        # Welch t-test
        t_stat, p_val = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')

        results.append({
            'network': col.replace('_correlation', ''), 'feature_col': col,
            'subtype1_label': subtype1, 'subtype2_label': subtype2,
            't_stat': t_stat, 'p_value': p_val,
        })

    result_df = pd.DataFrame(results)

    # FDR
    valid_mask = result_df['p_value'].notna()
    result_df['p_fdr'] = np.nan
    result_df['significant_fdr'] = False

    if valid_mask.sum() > 0:
        reject_fdr, pvals_fdr, _, _ = multipletests(
            result_df.loc[valid_mask, 'p_value'],
            alpha=0.05,
            method='fdr_bh'
        )
        result_df.loc[valid_mask, 'p_fdr'] = pvals_fdr
        result_df.loc[valid_mask, 'significant_fdr'] = reject_fdr

    result_df = result_df.sort_values('p_value', na_position='last')
    result_df.to_csv(os.path.join(out_dir, "subtype_network_comparison.csv"), index=False)

    return result_df

def plot_stability_silhouette(stability_df, valid_sil, plot_path):
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-white')

    fig, ax1 = plt.subplots(figsize=(6, 4))
    # Stability curve.
    ks = stability_df["k"].values
    means = stability_df["mean"].values
    ax1.plot(
        ks,means,'o-', color='#1f77b4',
        linewidth=1.5, markersize=5, markeredgewidth=1,
        label="Stability (ARI)"
    )
    # Silhouette curve.
    ax2 = ax1.twinx()

    sil_k = list(valid_sil.keys())
    sil_values = list(valid_sil.values())

    ax2.plot(
        sil_k, sil_values,'s-', color='#d62728',
        linewidth=1.5, markersize=5, markeredgewidth=1, label="Silhouette"
    )

    # Draw the left axis above the right axis.
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Make the right-axis background transparent and place it below the left axis.
    ax2.set_zorder(ax1.get_zorder() - 1)
    ax2.patch.set_visible(False)

    # Left axis.
    ax1.set_xlabel('Number of subtypes (k)', fontsize=14)
    ax1.tick_params(axis='x', labelsize=14, width=1.5, length=4)
    ax1.set_ylabel('Stability (ARI)', fontsize=14, color='#1f77b4')
    ax1.tick_params(axis='y', labelsize=14, colors='#1f77b4', width=1.5, length=4)

    # Configure the right axis position, spine, and colors.
    ax2.set_ylabel('Silhouette score', fontsize=14, color='#d62728')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax2.tick_params(axis='y', right=True, labelsize=14, colors='#d62728', width=1.5, length=4)

    # -----------legend-----------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    legend = ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='upper right', frameon=True, fontsize=14
    )

    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(1.5)
    frame.set_facecolor('white')
    frame.set_alpha(1)
    frame.set_boxstyle('square')

    plt.savefig(plot_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)


def _clustering_stability_two_subsamples(
    X_scaled: np.ndarray, subject_ids: np.ndarray, k: int,
    sample_frac: float = 0.8,
    n_repeats: int = 1000,
    metric: str = "ARI", random_state: int = 42, n_init: int = 20
):
    assert metric in ("ARI", "NMI")

    rng = np.random.RandomState(random_state)
    n = X_scaled.shape[0]
    m = int(np.floor(n * sample_frac))

    scores = []

    # Precompute the subject index array.
    all_idx = np.arange(n)

    for _ in range(n_repeats):
        idx1 = rng.choice(all_idx, size=m, replace=False)
        idx2 = rng.choice(all_idx, size=m, replace=False)

        # Cluster the two subsamples independently.
        km1 = KMeans(n_clusters=k, random_state=rng.randint(0, 10**9), n_init=n_init)
        km2 = KMeans(n_clusters=k, random_state=rng.randint(0, 10**9), n_init=n_init)

        lab1 = km1.fit_predict(X_scaled[idx1])
        lab2 = km2.fit_predict(X_scaled[idx2])

        # Find shared subjects using subject IDs.
        sub1_ids = subject_ids[idx1]
        sub2_ids = subject_ids[idx2]
        overlap = np.intersect1d(sub1_ids, sub2_ids, assume_unique=False)

        # Skip repetitions with insufficient overlap.
        if overlap.size < max(5, k * 2):
            continue

        # Align cluster labels for shared subjects.
        # Map subject IDs to cluster labels.
        map1 = dict(zip(sub1_ids, lab1))
        map2 = dict(zip(sub2_ids, lab2))

        y1 = np.array([map1[sid] for sid in overlap])
        y2 = np.array([map2[sid] for sid in overlap])

        if metric == "ARI":
            score = adjusted_rand_score(y1, y2)
        else:
            score = normalized_mutual_info_score(y1, y2)

        scores.append(score)

    scores = np.array(scores, dtype=float)
    return scores  # Return stability scores from all valid repetitions.


def _summarize_stability(scores: np.ndarray):
    """Return the mean, standard deviation, 95% CI, and valid repetition count."""
    if scores.size == 0:
        return dict(mean=np.nan, sd=np.nan, ci_low=np.nan, ci_high=np.nan, n=0)

    mean = float(np.mean(scores))
    sd = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    # Normal-approximation confidence interval for the mean.
    se = sd / np.sqrt(scores.size) if scores.size > 1 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return dict(mean=mean, sd=sd, ci_low=ci_low, ci_high=ci_high, n=int(scores.size))

def perform_clustering_analysis(
        group_data, networks,
        cluster_dir, group_name="CN_Aβ+"):

    print(f"\nClustering group: {group_name}")

    # --------------------------------------------------
    # 0. Preserve subject IDs.
    # --------------------------------------------------
    group_data = group_data.copy()
    group_data['SubjectID'] = group_data['SubjectID'].astype(str)

    subject_ids = group_data['SubjectID'].values
    subject_id_set = set(subject_ids)

    n_subjects = len(subject_ids)
    print(f"Subjects: {n_subjects}")

    # --------------------------------------------------
    # 1. Prepare the feature matrix from group_data.
    # --------------------------------------------------
    feature_columns = [f'{net}_correlation' for net in networks.keys()]

    features_df = group_data[['SubjectID'] + feature_columns].dropna()
    features_df = features_df.set_index('SubjectID')

    # Verify that retained subjects belong to the input cohort.
    assert set(features_df.index).issubset(subject_id_set)

    X = features_df.values

    # --------------------------------------------------
    # 2. Standardize features.
    # --------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 3. Evaluate candidate cluster counts using silhouette scores.
    # --------------------------------------------------
    print("\n=== Selecting the optimal number of clusters ===")

    max_k = min(MAX_K, len(X_scaled) - 1)
    k_range = range(2, max_k + 1)

    silhouette_scores = {}

    print("Calculating silhouette scores for candidate k values:")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)

        # Calculate cluster sizes.
        cluster_sizes = np.bincount(labels)

        # Calculate silhouette scores only when every cluster has at least two subjects.
        if len(cluster_sizes) == k and np.min(cluster_sizes) >= 2:
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
        else:
            silhouette_scores[k] = np.nan

    # --------------------------------------------------
    # 3.A Evaluate stability using repeated pairs of 80% subsamples.
    # --------------------------------------------------
    print("\n=== Clustering stability analysis ===")

    stability_metric = "ARI"         
    stability_sample_frac = SAMPLE_FRAC
    stability_repeats = STABILITY_REPEATS 
    min_k = 2
    max_k = min(MAX_K, len(X_scaled) - 1)
    k_range = range(min_k, max_k + 1)

    stability_summary = {}
    for k in k_range:
        # Evaluate stability for each candidate k.
        scores = _clustering_stability_two_subsamples(
            X_scaled=X_scaled,
            subject_ids=features_df.index.values.astype(str),  # Preserve the feature matrix subject order.
            k=k,
            sample_frac=stability_sample_frac,
            n_repeats=stability_repeats,
            metric=stability_metric,
            random_state=RANDOM_STATE, n_init=20)
        stability_summary[k] = _summarize_stability(scores)


    # Save the stability summary.
    stability_df = pd.DataFrame([
        {"k": k, **stability_summary[k]} for k in sorted(stability_summary.keys())
    ])
    stability_df.to_csv(
        os.path.join(cluster_dir, f"stability_{stability_metric}_{group_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # 3.1 Select k using the highest valid silhouette score.
    # --------------------------------------------------
    valid_sil = {k: v for k, v in silhouette_scores.items() if not np.isnan(v)}

    if valid_sil:
        best_k = max(valid_sil, key=valid_sil.get)
        best_sil_score = valid_sil[best_k]
        
        if best_sil_score < 0.25 and len(valid_sil) > 1:
            print(f"Warning: the best silhouette score is low ({best_sil_score:.3f}); retaining k={best_k}.")
    else:
        best_k = 2
        best_sil_score = 0
        print("Warning: no valid silhouette scores are available; using k=2.")

    # --------------------------------------------------
    # 3.3 Plot stability and silhouette scores.
    # --------------------------------------------------
    plot_path = os.path.join(cluster_dir, f'clustering_evaluation_{group_name}.png')
    plot_stability_silhouette(stability_df, valid_sil, plot_path)

    # --------------------------------------------------
    # 4. Fit the final clustering model.
    # --------------------------------------------------
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    final_labels = kmeans.fit_predict(X_scaled)

    # === SubjectID → Cluster ===
    cluster_series = pd.Series(final_labels,index=features_df.index,name='Cluster' )
    cluster_label_series = cluster_series.map(lambda x: f"Subtype{x + 1}")
    # Project standardized features onto two principal components.
    pca_df = plot_pca_subtypes(
        X_scaled,
        features_df.index.values,
        cluster_label_series,
        cluster_dir,
        group_name
    )

    pc1_s1 = pca_df[pca_df['Cluster_Label'] == 'Subtype1']['PC1']
    pc1_s2 = pca_df[pca_df['Cluster_Label'] == 'Subtype2']['PC1']

    _, p = ttest_ind(pc1_s1, pc1_s2)
    print("PC1 difference:", p)


    # --------------------------------------------------
    # 5. Assemble the final clustering results.
    # --------------------------------------------------
    clustering_results = features_df.copy()
    clustering_results['Cluster'] = cluster_series
    clustering_results['Cluster_Label'] = cluster_label_series
    clustering_results.reset_index(inplace=True)  # Restore SubjectID as a column.

    # Verify that output subjects belong to the input cohort.
    assert set(clustering_results['SubjectID']).issubset(subject_id_set)

    # --------------------------------------------------
    # 6. Save subject IDs for each subtype.
    # --------------------------------------------------
    subtype_subjects = {}
    for k in sorted(cluster_series.unique()):
        label = f"Subtype{k + 1}"
        ids = cluster_series[cluster_series == k].index.tolist()
        subtype_subjects[label] = ids

        out = pd.DataFrame({'SubjectID': ids})
        out.to_csv(
            os.path.join(cluster_dir, f'subjects_{label}.csv'),
            index=False,
            encoding='utf-8-sig'
        )

    # --------------------------------------------------
    # 8. Save the clustering results.
    # --------------------------------------------------
    clustering_results.to_csv(
        os.path.join(cluster_dir, 'clustering_results.csv'),
        index=False, encoding='utf-8-sig')


    return clustering_results, subtype_subjects


def main():

    file_wscore = WSCORE_DIR / "W_scores_threshold0.csv"
    file_wscore_demo = WSCORE_DIR / "W_scores_demo.csv"

    Dir_net = NETWORK_DIR
    outpath_subtype = OUTPUT_DIR
    os.makedirs(outpath_subtype, exist_ok=True)

    # Define clinical group labels.
    group_labels = { 1: 'CN Aβ+', 2: 'MCI Aβ+', 3: 'AD Aβ+'}

    print("1. Loading data...")
    wscore_data = pd.read_csv(file_wscore, index_col=0, header=0)
    wscore_data.index = wscore_data.index.astype(str)

    data_demo = pd.read_csv(file_wscore_demo, dtype={"Subject ID": str})
    data_demo = data_demo.rename(columns={"Subject ID": "SubjectID"})
    data_demo = data_demo.rename(columns={"Study Date": "StudyDate"})

    data_demo['SubjectID'] = data_demo['SubjectID'].astype(str).str.strip()
    subject_to_group = (data_demo.set_index('SubjectID')['Group_Code'])

    unique_groups = subject_to_group.unique()

    print("2. Loading network matrices...")
    gc = np.load(os.path.join(Dir_net, 'gc.npy'))
    rs = np.load(os.path.join(Dir_net, 'rs.npy'))
    ls = np.load(os.path.join(Dir_net, 'ls.npy'))
    mc = np.load(os.path.join(Dir_net, 'mc.npy'))
    hc = np.load(os.path.join(Dir_net, 'hc.npy'))
    ec = np.load(os.path.join(Dir_net, 'ec.npy'))
    ts = np.load(os.path.join(Dir_net, 'ts.npy'))
    sc = np.load(os.path.join(Dir_net, 'sc.npy'))
    euclidean = np.load(os.path.join(Dir_net, 'euclidean.npy'))

    # Define the network dictionary.
    networks = {
        "Euclidean": euclidean, "SC": sc, "GC": gc, "RS": rs, "LS": ls,
        "FC": hc, "MC": mc, "EC": ec, "TS": ts}

    # Calculate network features separately for each clinical group.
    all_group_results = []
    for group in unique_groups:
        group_label = group_labels.get(group, f'Group {group}')
        print(f"3. Processing group {group} ({group_label})...")

        group_subject_ids = subject_to_group[subject_to_group == group].index.tolist()
        group_wscore_data = wscore_data.loc[group_subject_ids].values

        # Calculate network correlations sequentially.
        group_results = []
        for net_name, net_matrix in networks.items():
            result = calculate_network_correlations_for_group(
                net_name, net_matrix, group_wscore_data, group, group_subject_ids, group_labels
            )
            group_results.append(result)

        # Initialize the group result table with subject IDs.
        group_final_result = pd.DataFrame({
            'SubjectID': group_subject_ids,
            'group': group, 'group_label': group_label  # Include the clinical group label.
        })

        # Merge results from each network.
        for net_name, result_df in zip(networks.keys(), group_results):
            # Normalize the subject ID column name.
            result_df = result_df.copy()
            if 'subject' in result_df.columns and 'SubjectID' not in result_df.columns:
                result_df = result_df.rename(columns={'subject': 'SubjectID'})

            # Keep subject IDs as strings.
            if 'SubjectID' in result_df.columns:
                result_df['SubjectID'] = result_df['SubjectID'].astype(str)

            # Identify network correlation columns.
            net_corr_columns = [col for col in result_df.columns
                                if '_correlation' in col and col != 'SubjectID']

            # Use the standard name when one correlation column is present.
            if len(net_corr_columns) == 1:
                # Retain correlation values.
                corr_values = result_df[net_corr_columns[0]].values
                # Rename the correlation column.
                result_df = result_df.drop(columns=net_corr_columns)
                result_df[f'{net_name}_correlation'] = corr_values
                net_corr_columns = [f'{net_name}_correlation']

            # Select columns to merge.
            columns_to_merge = ['SubjectID'] + net_corr_columns

            # Use the supplied group IDs if the result lacks a subject ID column.
            if 'SubjectID' not in result_df.columns:
                if len(result_df) == len(group_subject_ids):
                    result_df['SubjectID'] = group_subject_ids
                    columns_to_merge = ['SubjectID'] + net_corr_columns

            # Merge network results by subject ID.
            group_final_result = pd.merge(group_final_result, result_df[columns_to_merge],
                                          on='SubjectID', how='left',
                                          suffixes=('', f'_{net_name}'))

        all_group_results.append(group_final_result)

        # Combine results from all processed groups.
        final_result_all_groups = pd.concat(all_group_results, ignore_index=True)
        output_file = os.path.join(outpath_subtype, 'all_groups_network_correlations.csv')
        final_result_all_groups.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Network correlation results for all processed groups saved to: {output_file}")

    print("\nRunning clustering analysis...")

    group_data = final_result_all_groups.copy()

    group_name_for_file = ''

    # Run clustering.
    clustered_data, subtype_subjects = perform_clustering_analysis(
        group_data, networks, outpath_subtype, group_name_for_file
    )

    compare_subtype_network_features(
        clustered_data=clustered_data,
        networks=networks,
        out_dir=outpath_subtype, group_name=group_name_for_file
    )

    create_subtype_network_raincloud_plot(
        clustered_data=clustered_data,
        networks=networks, out_dir=outpath_subtype,
        group_col='Cluster_Label',
        subtype_order=('Subtype1', 'Subtype2'),
        colors=('#1f77b4', '#d62728')
    )


    print("\nAll calculations and analyses completed.")


if __name__ == "__main__":
    main()