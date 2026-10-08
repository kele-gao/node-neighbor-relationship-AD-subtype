
from matplotlib.legend_handler import HandlerTuple
import re
from scipy.stats import linregress, ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

def create_subtype_network_raincloud_plot(
        clustered_data,networks,out_dir,
        group_col='Cluster_Label',
        subtype_order=('Subtype1', 'Subtype2'),
        colors=('#1f77b4', '#d62728'),
        dpi=300):
    """
    多网络两亚型 raincloud plot
    """

    os.makedirs(out_dir, exist_ok=True)


    feature_cols = [f'{net}_correlation' for net in networks.keys()
                    if f'{net}_correlation' in clustered_data.columns]
    network_names = [c.replace('_correlation', '') for c in feature_cols]

    if len(feature_cols) == 0:
        print("没有找到可绘图的 network feature 列。")
        return

    plot_df = clustered_data[[group_col] + feature_cols].copy()
    plot_df = plot_df.melt(
        id_vars=[group_col],
        value_vars=feature_cols,
        var_name='Network',
        value_name='Value'
    )
    plot_df['Network'] = plot_df['Network'].str.replace('_correlation', '', regex=False)
    plot_df['Value'] = pd.to_numeric(plot_df['Value'], errors='coerce')
    plot_df = plot_df.dropna(subset=[group_col, 'Network', 'Value']).copy()

    if plot_df.empty:
        print("raincloud图没有可用数据。")
        return

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 20,

        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })

    fig, ax = plt.subplots(figsize=(14, 6))

    base_positions = np.arange(len(network_names), dtype=float)

    # 两个亚型相对于每个网络中心的位置
    group_offsets = [-0.2, 0.2]

    # 每个raincloud内部参数
    box_width = 0.10
    violin_width = 0.32
    scatter_sd = 0.028

    all_values = []

    # =========================
    # 先计算每个网络的p值
    # =========================
    p_dict = {}
    for net in network_names:
        x = plot_df[(plot_df['Network'] == net) &
                    (plot_df[group_col] == subtype_order[0])]['Value'].dropna().values
        y = plot_df[(plot_df['Network'] == net) &
                    (plot_df[group_col] == subtype_order[1])]['Value'].dropna().values

        if len(x) > 2 and len(y) > 2:
            _, p = ttest_ind(x, y, equal_var=False)
        else:
            p = np.nan
        p_dict[net] = p

    # =========================
    # 逐网络、逐亚型作图
    # =========================
    for j, net in enumerate(network_names):
        for i, subtype in enumerate(subtype_order):
            vals = plot_df[
                (plot_df['Network'] == net) &
                (plot_df[group_col] == subtype)
            ]['Value'].dropna().values

            if len(vals) == 0:
                continue

            all_values.extend(vals.tolist())

            center = base_positions[j] + group_offsets[i]

            # 1) 散点：放在左半边一点
            rng = np.random.default_rng(1000 + j * 10 + i)
            x_jitter = rng.normal(loc=center - 0.04, scale=scatter_sd, size=len(vals))
            ax.scatter(
                x_jitter, vals,
                color=colors[i],
                alpha=0.60,
                s=32,
                edgecolor='white',
                linewidth=0.5,
                zorder=3
            )

            # 2) 箱线图：放在中间略偏左
            box_pos = center - 0.01
            ax.boxplot(
                vals,
                positions=[box_pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                notch=False,
                medianprops={'color': 'black', 'linewidth': 1.6, 'zorder': 5},
                boxprops={'facecolor': colors[i], 'edgecolor': colors[i], 'linewidth': 1.4, 'zorder': 4},
                whiskerprops={'color': colors[i], 'linewidth': 1.4, 'zorder': 4},
                capprops={'color': colors[i], 'linewidth': 1.4, 'zorder': 4}
            )

            # 3) 半小提琴图：只保留右半边
            violin_pos = center + 0.015
            violin = ax.violinplot(
                vals,
                positions=[violin_pos],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )

            for pc in violin['bodies']:
                pc.set_facecolor(colors[i])
                pc.set_edgecolor(colors[i])
                pc.set_alpha(0.30)
                pc.set_zorder(2)

                verts = pc.get_paths()[0].vertices
                verts[:, 0] = np.where(verts[:, 0] > violin_pos, verts[:, 0], violin_pos)

        # =========================
        # 显著性星号
        # =========================
        y_max = plot_df[plot_df['Network'] == net]['Value'].dropna().max()
        p = p_dict.get(net, np.nan)
        if pd.isna(p):
            continue

        if p < 0.05:
            star = '*'
        else:
            star = ''

        if star:
            ax.text(
                base_positions[j], y_max + 0.05,
                star,ha='center', va='bottom',fontweight='bold')

    # =========================
    # 坐标轴与格式
    # =========================
    ax.set_xticks(base_positions)
    ax.set_xticklabels(network_names, rotation=0)
    ax.set_ylabel('Node-neighbour relationship')
    ax.set_xlim(-0.7, len(network_names) - 0.3)
    ax.set_ylim(-0.4, 0.8)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) #显示一位小数

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='both', width=1)

    plt.tight_layout()

    save_path = os.path.join(out_dir, '_subtype_network_raincloud.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    # =========================
    # 图例
    # =========================
    fig_leg = plt.figure(figsize=(2.2, 1.2))

    # legend handles
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o', linestyle='', color='w',
            label=subtype_order[0], markerfacecolor=colors[0], markeredgecolor='white', markersize=10
        ),

        Line2D(
            [0], [0], marker='o', linestyle='', color='w',
            label=subtype_order[1], markerfacecolor=colors[1], markeredgecolor='white', markersize=10)
    ]

    # =========================
    # 创建 legend
    # =========================
    legend = fig_leg.legend(
        handles=legend_handles,
        loc='center',
        frameon=True,
        borderpad=0.3,  # legend内部边距
        handletextpad=0.5,  # marker和文字距离
        labelspacing=0.4,  # 行间距
        borderaxespad=0.2,  # legend与边界距离
        fontsize=10,
        ncol=1
    )

    # legend边框设置
    frame = legend.get_frame()

    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(0.8)
    frame.set_alpha(1.0)
    frame.set_boxstyle('square')

    # marker进一步统一
    for handle in legend.legend_handles:
        handle.set_alpha(1)
        handle.set_markersize(7)


    # 去掉坐标轴
    plt.axis('off')

    # 保存
    fig_leg.savefig(
        os.path.join(
            out_dir,
            '_subtype_network_raincloud_legend.png'
        ),
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close(fig_leg)


def create_subtype_difference_radar_chart(
        clustered_data,
        networks,
        out_dir,
        group_col='Cluster_Label',
        subtype_order=('Subtype1', 'Subtype2'),
        line_color='#7f3c8d',
        fill_alpha=0.18,
        dpi=300):
    """
    两亚型差值雷达图：
    每个网络的值 = Subtype2均值 - Subtype1均值
    正值: Subtype2更高
    负值: Subtype1更高
    """
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = [f'{net}_correlation' for net in networks.keys()
                    if f'{net}_correlation' in clustered_data.columns]
    network_names = [c.replace('_correlation', '') for c in feature_cols]

    if len(feature_cols) == 0:
        print("没有找到可绘图的 network feature 列。")
        return

    subtype1_df = clustered_data[clustered_data[group_col] == subtype_order[0]].copy()
    subtype2_df = clustered_data[clustered_data[group_col] == subtype_order[1]].copy()

    if subtype1_df.empty or subtype2_df.empty:
        print("某个亚型没有数据，无法绘制差值雷达图。")
        return

    subtype1_mean = subtype1_df[feature_cols].mean()
    subtype2_mean = subtype2_df[feature_cols].mean()

    diff_values = (subtype2_mean - subtype1_mean).values.astype(float)

    n_networks = len(network_names)
    angles = np.linspace(0, 2 * np.pi, n_networks, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    diff_closed = diff_values.tolist() + diff_values[:1].tolist()

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)

    max_abs = np.max(np.abs(diff_values))
    if max_abs == 0:
        max_abs = 0.1
    lim = max_abs * 1.25

    ax.set_ylim(-lim, lim)

    # 网格和0基线
    ax.set_xticks(angles)
    ax.set_xticklabels(network_names, fontsize=12)

    rticks = np.linspace(-lim, lim, 5)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f'{x:.2f}' for x in rticks], fontsize=10)

    ax.plot(angles_closed, diff_closed, color=line_color, linewidth=2.2)
    ax.fill(angles_closed, diff_closed, color=line_color, alpha=fill_alpha)

    # 画0环
    theta_dense = np.linspace(0, 2*np.pi, 500)
    ax.plot(theta_dense, np.zeros_like(theta_dense), color='black', linewidth=1.2, linestyle='--')

    ax.set_title('Subtype2 - Subtype1', fontsize=14, pad=20)

    ax.yaxis.grid(True, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.xaxis.grid(True, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_facecolor('white')

    save_path = os.path.join(out_dir, 'CN_Abeta_pos_subtype_difference_radar.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()





def create_combined_radar_chart(group_features_dict, network_names, radar_dir,colors=['#1f77b4', '#d62728', '#2ca02c', '#9467bd']):
    """
    创建组合雷达图，显示所有组别和亚型的网络特征
    """
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    n_networks = len(network_names)
    all_features = {}

    for group_label, data in group_features_dict.items():
        if data is None or len(data) == 0:
            continue

        if 'Cluster_Label' in data.columns:
            # 对亚型标签进行排序，保证亚型1 → 亚型2 → …
            sorted_cluster_labels = sorted(
                data['Cluster_Label'].unique(),
                key=lambda x: int(x.replace('Subtype', ''))
            )
            stat_results = {}
            group_stat_results = {}

            # =========================
            # 计算亚型之间的统计差异
            # =========================
            feature_columns = [col for col in data.columns if col.endswith('_correlation')]

            subtype1 = data[data['Cluster_Label'] == 'Subtype1']
            subtype2 = data[data['Cluster_Label'] == 'Subtype2']

            for col in feature_columns:
                x = subtype1[col].dropna()
                y = subtype2[col].dropna()

                if len(x) > 3 and len(y) > 3:
                    _, p = ttest_ind(x, y)
                else:
                    p = np.nan

                stat_results[col] = p

            for cluster_label in sorted_cluster_labels:
                cluster_data = data[data['Cluster_Label'] == cluster_label]

                feature_columns = [col for col in data.columns if col.endswith('_correlation')]
                if feature_columns:
                    cluster_mean = cluster_data[feature_columns].mean().values
                    all_features[f'{group_label}_{cluster_label}'] = cluster_mean
        else:
            feature_columns = [col for col in data.columns if col.endswith('_correlation')]
            if feature_columns:
                group_mean = data[feature_columns].mean().values
                all_features[group_label] = group_mean

    # 以下绘图逻辑保持不变
    fig = plt.figure(figsize=(7, 5))  # 增大图形大小
    angles = np.linspace(0, 2 * np.pi, n_networks, endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(111, polar=True)
    ax.set_xticks(angles[:-1])

    # 先设置空标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    all_values = []
    for values in all_features.values():
        all_values.extend(values)

    y_min = np.floor(np.min(all_values) * 10) / 10
    y_max = np.ceil(np.max(all_values) * 10) / 10

    padding = 0.1 * (y_max - y_min)

    ax.set_ylim(y_min - padding, y_max + padding)

    # 设置圆圈
    r_ticks = np.linspace(y_min - padding, y_max + padding, 4)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([])

    # label 半径
    label_radius = ax.get_ylim()[1] * 1.05

    # 增加y轴上限，为标签留出更多空间
    ax.set_rlabel_position(30)

    line_styles = ['--', '-']

    legend_handles = []
    legend_labels = []

    for i, (feature_label, values) in enumerate(all_features.items()):
        values_closed = values.tolist() + values.tolist()[:1]

        if 'Subtype' in feature_label:
            line_style = line_styles[1]
            color_idx = int(feature_label.split('Subtype')[-1]) - 1
        else:
            line_style = line_styles[0]
            color_idx = i % len(colors)

        line = ax.plot(angles, values_closed, linewidth=2, linestyle=line_style,
                       color=colors[color_idx % len(colors)], label=feature_label)
        # ax.scatter(angles[:-1], values_closed[:-1], color=colors[color_idx % len(colors)],
        #            s=20, zorder=5)

        legend_handles.append(line[0])
        legend_labels.append(feature_label)

    # ax.grid(True, alpha=0.3)
    ax.set_facecolor('white') #设置背景色
    ax.yaxis.grid(True, color='black', linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
    ax.xaxis.grid(True, color='black', linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)

    # 手动添加标签，控制每个标签的位置
    label_radius = y_max + padding + 0.01  # 标签放置的半径

    for i, (angle, label) in enumerate(zip(angles[:-1], network_names)):
        # 找对应feature列
        feature_col = f"{label}_correlation"

        # 根据角度调整标签对齐方式
        if angle < np.pi / 2 or angle > 3 * np.pi / 2:  # 右侧区域
            ha = 'left'
        else:  # 左侧区域
            ha = 'right'

        # 调整垂直对齐方式
        if angle < np.pi:  # 上半部分
            va = 'bottom'
        else:  # 下半部分
            va = 'top'

        # p = stat_results.get(feature_col, np.nan)
        # if p < 0.05:
        #     star = '*'
        # else:
        #     star = ''

        # 添加标签
        ax.text(angle, label_radius, label,
                ha=ha, va=va, fontsize=15)


    output_file = os.path.join(radar_dir, 'all_groups_network_features_radar.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"组合雷达图已保存: {output_file}")

    # =========================
    # 单独保存 legend
    # =========================

    fig_legend = plt.figure(figsize=(4, 4))

    legend = fig_legend.legend(
        legend_handles,
        legend_labels,
        loc='center',
        fontsize=15,
        frameon=True
    )

    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(1.5)
    frame.set_facecolor('white')
    frame.set_alpha(1)
    frame.set_boxstyle('square')

    legend_file = os.path.join(radar_dir, 'all_groups_network_features_legend.png')

    fig_legend.savefig(
        legend_file,
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )

    plt.close(fig_legend)

def create_two_group_raincloud_visitnum_plot(
        df, value_col, group_col='Cluster_Label',
        group_order=None,
        output_path='.',
        file_name='raincloud_plot.png',
        y_label='Value',
        title='', colors=['#3B6FB6', '#C23B22', '#2ca02c', '#9467bd'],
        figsize=(5, 5), dpi=1000, p_text=None):
    """

    参数
    ----------
    df : pd.DataFrame 至少包含 group_col 和 value_col
    value_col : str 数值列名，如 'global_tau' 或 'global_tau_rate'
    group_col : str 分组列，默认 'Cluster_Label'
    group_order : list 组别顺序，如 ['Subtype1', 'Subtype2']
    output_path : str 输出目录
    file_name : str 输出文件名
    y_label : str y轴标签
    title : str 标题
    colors : list  两组颜色
    text : str  额外文本注释
    p_value : float 若提供，则在图中标注
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plot_font_size = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': plot_font_size,

        'axes.labelsize': plot_font_size,
        'axes.titlesize': plot_font_size,
        'xtick.labelsize': plot_font_size,
        'ytick.labelsize': plot_font_size,
        'legend.fontsize': plot_font_size,
        'figure.titlesize': plot_font_size,

        # 强制白色背景、无网格
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'axes.grid': False
    })

    os.makedirs(output_path, exist_ok=True)

    plot_df = df[[group_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[group_col, value_col]).copy()

    if plot_df.empty:
        print(f"警告: {value_col} 没有可用于绘图的数据。")
        return None

    if group_order is None:
        group_order = sorted(
            plot_df[group_col].dropna().unique().tolist(),
            key=lambda x: int(str(x).replace('Subtype', '')) if str(x).startswith('Subtype') else str(x)
        )

    fig, ax = plt.subplots(figsize=figsize)

    positions = np.array([0, 1], dtype=float)
    box_width = 0.1
    violin_width = 0.5

    all_values = []

    for i, grp in enumerate(group_order):
        vals = plot_df.loc[plot_df[group_col] == grp, value_col].dropna().values
        if len(vals) == 0:
            continue

        all_values.extend(vals.tolist())

        # ----------------- 散点 -----------------
        rng = np.random.default_rng(42 + i)
        x_jitter = rng.normal(loc=positions[i] - box_width, scale=0.035, size=len(vals))
        ax.scatter(
            x_jitter, vals,
            color=colors[i],
            alpha=0.6, s=40,
            edgecolor='white', linewidth=0.5, zorder=2)

        # ----------------- 箱线图 -----------------
        box_pos = positions[i] - box_width / 30
        bp = ax.boxplot(
            vals,
            positions=[box_pos],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            notch=False,
            medianprops={'color': 'black', 'linewidth': 2, 'zorder': 4},
            boxprops={'facecolor': colors[i], 'edgecolor': colors[i], 'linewidth': 2, 'zorder': 3},
            whiskerprops={'color': colors[i], 'linewidth': 2, 'zorder': 3},
            capprops={'color': colors[i], 'linewidth': 2, 'zorder': 3}
        )

        # ----------------- 半小提琴图 -----------------
        violin_pos = positions[i] + box_width / 30
        violin = ax.violinplot(
            vals,
            positions=[violin_pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        for pc in violin['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor(colors[i])
            pc.set_alpha(0.35)
            pc.set_zorder(1)

            # 只保留右半边
            verts = pc.get_paths()[0].vertices
            verts[:, 0] = np.where(verts[:, 0] > violin_pos, verts[:, 0], violin_pos)

    # ----------------- 坐标轴 -----------------
    ax.set_xticks(positions)
    xtick_labels = [
        {
            'Subtype1': 'connectome-discordant\nsubtype',
            'Subtype2': 'connectome-concordant\nsubtype'
        }.get(str(grp), f'{grp}')
        for grp in group_order
    ]
    ax.set_xticklabels(xtick_labels, fontsize=plot_font_size)
    ax.set_ylabel(y_label, fontsize=plot_font_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # 添加标题
    ax.set_title(title, pad=5, fontsize=plot_font_size)

    # y轴范围自动留白
    ax.set_ylim(0, 14)

    ax.tick_params(axis='x', which='major', width=1.5, length=5, labelsize=plot_font_size)
    ax.tick_params(axis='y', which='major', width=1.5, length=5, labelsize=plot_font_size)

    # 边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ----------------- 标注p值 -----------------
    if p_text is not None and len(all_values) > 0:
        y_min = np.nanmin(all_values)
        y_max = np.nanmax(all_values)
        y_range = y_max - y_min if y_max > y_min else 1.0

        line_y = y_max + 0.07 * y_range
        text_y = y_max + 0.12 * y_range

        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [line_y, line_y + 0.02 * y_range, line_y + 0.02 * y_range, line_y],
            color='black', linewidth=1.2
        )

        ax.text(np.mean(positions), text_y, p_text,
            ha='center', va='bottom', fontsize=plot_font_size)

    plt.tight_layout()
    save_path = os.path.join(output_path, file_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return


def create_two_group_raincloud_plot(
        df, value_col, group_col='Cluster_Label',
        group_order=None,
        output_path='.',
        file_name='raincloud_plot.png',
        y_label='Value',
        title='', colors=['#3B6FB6', '#C23B22', '#2ca02c', '#9467bd'],
        figsize=(4, 4), dpi=1000, p_value=None):
    """
    '#4C72B0', '#DD8452' 蓝，橙
    两组比较版 Raincloud plot
    适用于:
        - global tau level
        - global tau rate

    参数
    ----------
    df : pd.DataFrame 至少包含 group_col 和 value_col
    value_col : str 数值列名，如 'global_tau' 或 'global_tau_rate'
    group_col : str 分组列，默认 'Cluster_Label'
    group_order : list 组别顺序，如 ['Subtype1', 'Subtype2']
    output_path : str 输出目录
    file_name : str 输出文件名
    y_label : str y轴标签
    title : str 标题
    colors : list 两组颜色
    text : str 额外文本注释
    p_value : float 若提供，则在图中标注
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 15,

        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'figure.titlesize': 15,

        # 强制白色背景
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',

        # 不显示网格
        'axes.grid': False
    })

    os.makedirs(output_path, exist_ok=True)

    plot_df = df[[group_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[group_col, value_col]).copy()

    if plot_df.empty:
        print(f"警告: {value_col} 没有可用于绘图的数据。")
        return None

    if group_order is None:
        group_order = sorted(
            plot_df[group_col].dropna().unique().tolist(),
            key=lambda x: int(str(x).replace('Subtype', '')) if str(x).startswith('Subtype') else str(x)
        )

    fig, ax = plt.subplots(figsize=figsize)

    positions = np.array([0, 1], dtype=float)
    box_width = 0.1
    violin_width = 0.5

    all_values = []

    for i, grp in enumerate(group_order):
        vals = plot_df.loc[plot_df[group_col] == grp, value_col].dropna().values
        if len(vals) == 0:
            continue

        all_values.extend(vals.tolist())

        # ----------------- 散点 -----------------
        rng = np.random.default_rng(42 + i)
        x_jitter = rng.normal(loc=positions[i] - box_width, scale=0.035, size=len(vals))
        ax.scatter(
            x_jitter, vals,
            color=colors[i],
            alpha=0.6, s=40,
            edgecolor='white', linewidth=0.5, zorder=2)

        # ----------------- 箱线图 -----------------
        box_pos = positions[i] - box_width / 30
        bp = ax.boxplot(
            vals,
            positions=[box_pos],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            notch=False,
            medianprops={'color': 'black', 'linewidth': 2, 'zorder': 4},
            boxprops={'facecolor': colors[i], 'edgecolor': colors[i], 'linewidth': 2, 'zorder': 3},
            whiskerprops={'color': colors[i], 'linewidth': 2, 'zorder': 3},
            capprops={'color': colors[i], 'linewidth': 2, 'zorder': 3}
        )

        # ----------------- 半小提琴图 -----------------
        violin_pos = positions[i] + box_width / 30
        violin = ax.violinplot(
            vals,
            positions=[violin_pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        for pc in violin['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor(colors[i])
            pc.set_alpha(0.35)
            pc.set_zorder(1)

            # 只保留右半边
            verts = pc.get_paths()[0].vertices
            verts[:, 0] = np.where(verts[:, 0] > violin_pos, verts[:, 0], violin_pos)

    # ----------------- 坐标轴 -----------------
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{grp}' for grp in group_order])
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # y轴范围自动留白
    if len(all_values) > 0:
        y_min = np.nanmin(all_values)
        y_max = np.nanmax(all_values)
        y_range = y_max - y_min if y_max > y_min else 1.0
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.22 * y_range)


    # 边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')

    # ----------------- 标注p值 -----------------
    if p_value is not None and len(all_values) > 0:
        y_min = np.nanmin(all_values)
        y_max = np.nanmax(all_values)
        y_range = y_max - y_min if y_max > y_min else 1.0

        line_y = y_max + 0.07 * y_range
        text_y = y_max + 0.12 * y_range

        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [line_y, line_y + 0.02 * y_range, line_y + 0.02 * y_range, line_y],
            color='black', linewidth=1.2
        )

        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'

        ax.text(
            np.mean(positions), text_y, p_text,
            ha='center', va='bottom')

    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_path = os.path.join(output_path, file_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return

def format_region_name_for_paper(region_name):
    """
    将类似 CTX_LH_INFERIORTEMPORAL 转成论文中更常见的脑区名称
    """


    if pd.isna(region_name):
        return region_name

    name = str(region_name).upper()

    # 去掉前缀
    name = re.sub(r'^CTX_', '', name)
    name = re.sub(r'_SUVR', '', name)

    # 半球
    hemi = ''
    if name.startswith('LH_'):
        hemi = 'Left '
        name = name[3:]
    elif name.startswith('RH_'):
        hemi = 'Right '
        name = name[3:]

    # FreeSurfer / DK 常见脑区名映射
    region_map = {
        'BANKSSTS': 'banks of the superior temporal sulcus',
        'CAUDALANTERIORCINGULATE': 'caudal anterior cingulate cortex',
        'CAUDALMIDDLEFRONTAL': 'caudal middle frontal gyrus',
        'CUNEUS': 'cuneus',
        'ENTORHINAL': 'entorhinal cortex',
        'FUSIFORM': 'fusiform gyrus',
        'INFERIORPARIETAL': 'inferior parietal lobule',
        'INFERIORTEMPORAL': 'inferior temporal gyrus',
        'ISTHMUSCINGULATE': 'isthmus cingulate cortex',
        'LATERALOCCIPITAL': 'lateral occipital cortex',
        'LATERALORBITOFRONTAL': 'lateral orbitofrontal cortex',
        'LINGUAL': 'lingual gyrus',
        'MEDIALORBITOFRONTAL': 'medial orbitofrontal cortex',
        'MIDDLETEMPORAL': 'middle temporal gyrus',
        'PARAHIPPOCAMPAL': 'parahippocampal gyrus',
        'PARACENTRAL': 'paracentral lobule',
        'PARSOPERCULARIS': 'pars opercularis',
        'PARSORBITALIS': 'pars orbitalis',
        'PARSTRIANGULARIS': 'pars triangularis',
        'PERICALCARINE': 'pericalcarine cortex',
        'POSTCENTRAL': 'postcentral gyrus',
        'POSTERIORCINGULATE': 'posterior cingulate cortex',
        'PRECENTRAL': 'precentral gyrus',
        'PRECUNEUS': 'precuneus',
        'ROSTRALANTERIORCINGULATE': 'rostral anterior cingulate cortex',
        'ROSTRALMIDDLEFRONTAL': 'rostral middle frontal gyrus',
        'SUPERIORFRONTAL': 'superior frontal gyrus',
        'SUPERIORPARIETAL': 'superior parietal lobule',
        'SUPERIORTEMPORAL': 'superior temporal gyrus',
        'SUPRAMARGINAL': 'supramarginal gyrus',
        'FRONTALPOLE': 'frontal pole',
        'TEMPORALPOLE': 'temporal pole',
        'TRANSVERSETEMPORAL': 'transverse temporal cortex',
        'INSULA': 'insula',
    }

    pretty = region_map.get(name, name.lower().replace('_', ' '))
    return hemi + pretty

def plot_significant_tau_suvr_trajectories(
    subtype_subjects, suvr_file, output_dir,
    group_name="CN_Aβ+",
    region_cols=None,
    alpha=0.05,
    use_fdr=False,
    min_timepoints=2
):
    """
    绘制tau SUVR纵向变化轨迹图：
    1) 若global tau rate组间差异显著，则绘制global tau轨迹
    2) 若某个region tau rate组间差异显著，则绘制该region轨迹
    3) 额外新增：针对所有有差异的脑区，每个亚型画一个图，
       每个图中同时绘制所有显著脑区随时间变化的拟合线
    4) 图片输出到 tau_trajectory_plots 目录

    Parameters
    ----------
    subtype_subjects : dict
        {subtype_label: [SubjectID1, SubjectID2, ...]}
    suvr_file : str
        tau SUVR csv文件路径，需包含 PTID, SCANDATE, TRACER, QC_FLAG 及各ROI列
    output_dir : str
        输出目录
    group_name : str
        分组名称
    region_cols : list or None
        指定脑区列；若为None则自动寻找包含'_SUVR'的列
    alpha : float
        显著性阈值
    use_fdr : bool
        True时依据p_fdr判断显著；False时依据p_value判断显著
    min_timepoints : int
        纳入纵向轨迹所需最少时间点
    """
    os.makedirs(output_dir, exist_ok=True)
    traj_dir = os.path.join(output_dir, "tau_trajectory_plots")
    os.makedirs(traj_dir, exist_ok=True)

    # ========= 1. 读取数据 =========
    suvr_df = pd.read_csv(suvr_file)
    suvr_df['PTID'] = suvr_df['PTID'].astype(str).str.strip()

    if 'QC_FLAG' in suvr_df.columns:
        suvr_df = suvr_df[suvr_df['QC_FLAG'] != -1].copy()

    exclude_cols = [
        'TRACER_SUVR WARNING', 'TRACER_SUVR_WARNING',
        'INFERIORCEREBELLUM_SUVR',
        'RIGHT_CEREBELLUM_CORTEX_SUVR', 'LEFT_CEREBELLUM_CORTEX_SUVR',
        'RIGHT_CEREBELLUM_WHITE_MATTER_SUVR', 'LEFT_CEREBELLUM_WHITE_MATTER_SUVR',
        'CEREBRAL_WHITE_MATTER_SUVR',
        'CSF_SUVR', 'VENTRICLE_5TH_SUVR', 'VENTRALDC_SUVR',
        'LATERAL_VENTRICLE_SUVR',
        'VESSEL_SUVR', 'RIGHT_VESSEL_SUVR', 'LEFT_VESSEL_SUVR',
        'META_TEMPORAL_SUVR', 'PALLIDUM_SUVR', 'PUTAMEN_SUVR', 'OPTIC_CHIASM_SUVR', 'INF_LAT_VENT_SUVR',
        'HIPPOCAMPUS_SUVR', 'CAUDATE_SUVR', 'AMYGDALA_SUVR', 'ACCUMBENS_AREA_SUVR',
        'WM_HYPOINTENSITIES_SUVR', 'NON_WM_HYPOINTENSITIES_SUVR', 'THALAMUS_PROPER_SUVR',
        'CTX_ENTORHINAL', 'CTX_AMYGDALA', 'CTX_FUSIFORM', 'CTX_INFERIORTEMPORAL', 'CTX_MIDDLETEMPORAL',
        'CEREBELLUM_CORTEX_SUVR', 'CEREBELLUM_WHITE_MATTER_SUVR',
        'CHOROID_PLEXUS_SUVR', 'RIGHT_CHOROID_PLEXUS_SUVR', 'LEFT_CHOROID_PLEXUS_SUVR',
        'RIGHT_INF_LAT_VENT_SUVR', 'LEFT_INF_LAT_VENT_SUVR',
        'RIGHT_LATERAL_VENTRICLE_SUVR', 'LEFT_LATERAL_VENTRICLE_SUVR', 'RIGHT_VENTRALDC_SUVR',
        'LEFT_VENTRALDC_SUVR',
        'RIGHT_CEREBRAL_WHITE_MATTER_SUVR', 'LEFT_CEREBRAL_WHITE_MATTER_SUVR',
        'BRAINSTEM_SUVR', 'VENTRICLE_3RD_SUVR', 'VENTRICLE_4TH_SUVR',
        'CC_ANTERIOR_SUVR', 'CC_CENTRAL_SUVR', 'CC_MID_ANTERIOR_SUVR',
        'CC_MID_POSTERIOR_SUVR', 'CC_POSTERIOR_SUVR'
    ]

    if region_cols is None:
        region_cols = [c for c in suvr_df.columns if ('_SUVR' in c and c not in exclude_cols)]

    subtype_labels = list(subtype_subjects.keys())

    if 'TRACER' in suvr_df.columns:
        tracer_list = [x for x in suvr_df['TRACER'].dropna().unique()]
    else:
        tracer_list = [None]

    # ========= 工具函数 =========
    def safe_filename(name):
        name = str(name)
        name = re.sub(r'[\\/*?:"<>|]+', "_", name)
        name = name.replace(" ", "_")
        return name

    def prettify_region_name(region_name):
        name = str(region_name)
        name = name.replace('_SUVR', '')
        return name

    def prepare_long_df_for_tracer(tracer_df):
        """
        构建长表：每个subject每个timepoint一行，包含global_tau和各region
        """
        records = []

        for subtype_label in subtype_labels:
            subject_ids = [str(s).strip() for s in subtype_subjects[subtype_label]]

            for sid in subject_ids:
                subj_df = tracer_df[tracer_df['PTID'] == sid].copy()
                if subj_df.shape[0] < min_timepoints:
                    continue

                if 'SCANDATE' not in subj_df.columns:
                    continue

                subj_df['SCANDATE'] = pd.to_datetime(subj_df['SCANDATE'], errors='coerce')
                subj_df = subj_df.dropna(subset=['SCANDATE']).sort_values('SCANDATE').copy()
                if subj_df.shape[0] < min_timepoints:
                    continue

                t0 = subj_df['SCANDATE'].iloc[0]
                subj_df['time_years'] = (subj_df['SCANDATE'] - t0).dt.days / 365.25

                if subj_df['time_years'].nunique() < 2:
                    continue

                subj_df['global_tau'] = subj_df[region_cols].apply(
                    lambda row: np.nanmean(pd.to_numeric(row, errors='coerce')),
                    axis=1
                )

                for _, row in subj_df.iterrows():
                    one = {
                        'SubjectID': sid,
                        'Cluster_Label': subtype_label,
                        'SCANDATE': row['SCANDATE'],
                        'time_years': row['time_years'],
                        'global_tau': pd.to_numeric(row['global_tau'], errors='coerce')
                    }
                    for region in region_cols:
                        one[region] = pd.to_numeric(row[region], errors='coerce')
                    records.append(one)

        return pd.DataFrame(records)

    def compute_rate_comparison(long_df):
        """
        从long_df计算global rate比较和region rate比较
        """
        global_rate_records = []
        region_rate_records = []

        for subtype_label in subtype_labels:
            sub_df = long_df[long_df['Cluster_Label'] == subtype_label].copy()

            for sid, subj_df in sub_df.groupby('SubjectID'):
                subj_df = subj_df.sort_values('time_years').copy()

                x = pd.to_numeric(subj_df['time_years'], errors='coerce').values
                yg = pd.to_numeric(subj_df['global_tau'], errors='coerce').values
                vg = (~np.isnan(x)) & (~np.isnan(yg))

                if vg.sum() >= min_timepoints and np.ptp(x[vg]) > 0:
                    xg = x[vg]
                    ygv = yg[vg]

                    if len(xg) >= 3:
                        slope_g, _, _, p_g, _ = linregress(xg, ygv)
                    else:
                        slope_g = (ygv[-1] - ygv[0]) / (xg[-1] - xg[0])
                        p_g = np.nan

                    global_rate_records.append({
                        'SubjectID': sid,
                        'Cluster_Label': subtype_label,
                        'global_tau_rate': slope_g,
                        'global_tau_rate_p_value': p_g
                    })

                for region in region_cols:
                    yr = pd.to_numeric(subj_df[region], errors='coerce').values
                    vr = (~np.isnan(x)) & (~np.isnan(yr))

                    if vr.sum() < min_timepoints or np.ptp(x[vr]) == 0:
                        continue

                    xr = x[vr]
                    yrv = yr[vr]

                    if len(xr) >= 3:
                        slope_r, _, _, p_r, _ = linregress(xr, yrv)
                    else:
                        slope_r = (yrv[-1] - yrv[0]) / (xr[-1] - xr[0])
                        p_r = np.nan

                    region_rate_records.append({
                        'SubjectID': sid,
                        'Cluster_Label': subtype_label,
                        'brain_region': region,
                        'tau_rate': slope_r,
                        'rate_p_value': p_r
                    })

        global_rate_df = pd.DataFrame(global_rate_records)
        region_rate_df = pd.DataFrame(region_rate_records)

        global_cmp = None
        if not global_rate_df.empty:
            g1 = global_rate_df.loc[
                global_rate_df['Cluster_Label'] == subtype_labels[0], 'global_tau_rate'
            ].dropna()
            g2 = global_rate_df.loc[
                global_rate_df['Cluster_Label'] == subtype_labels[1], 'global_tau_rate'
            ].dropna()

            if len(g1) >= 2 and len(g2) >= 2:
                t_stat, p_val = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
                global_cmp = pd.DataFrame([{
                    'measure': 'global_tau_rate',
                    'statistic': t_stat,
                    'p_value': p_val,
                    'subtype1_mean': g1.mean(),
                    'subtype2_mean': g2.mean(),
                    'subtype1_n': len(g1),
                    'subtype2_n': len(g2)
                }])

        region_cmp_records = []
        if not region_rate_df.empty:
            for region in region_cols:
                d1 = region_rate_df[
                    (region_rate_df['Cluster_Label'] == subtype_labels[0]) &
                    (region_rate_df['brain_region'] == region)
                ]['tau_rate'].dropna()

                d2 = region_rate_df[
                    (region_rate_df['Cluster_Label'] == subtype_labels[1]) &
                    (region_rate_df['brain_region'] == region)
                ]['tau_rate'].dropna()

                if len(d1) >= 2 and len(d2) >= 2:
                    t_stat, p_val = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')
                    region_cmp_records.append({
                        'brain_region': region,
                        'statistic': t_stat,
                        'p_value': p_val,
                        'subtype1_mean_rate': d1.mean(),
                        'subtype2_mean_rate': d2.mean(),
                        'subtype1_n': len(d1),
                        'subtype2_n': len(d2)
                    })

        region_cmp_df = pd.DataFrame(region_cmp_records)
        if not region_cmp_df.empty:
            _, p_fdr, _, _ = multipletests(
                region_cmp_df['p_value'],
                alpha=alpha,
                method='fdr_bh'
            )
            region_cmp_df['p_fdr'] = p_fdr

        return global_rate_df, region_rate_df, global_cmp, region_cmp_df


    def plot_one_trajectory(long_df, value_col, tracer, p_text, out_png,
                            colors=['#1f77b4', '#d62728', '#2ca02c', '#9467bd'], figsize=(9, 6)):
        """
        绘制某个指标的纵向轨迹图：
        一个大图中左右两个子图，每个子图分别对应一个亚型
        """
        n_subtypes = len(subtype_labels)
        if n_subtypes != 2:
            raise ValueError("当前函数版本仅适用于两个亚型。")

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white', sharex=True)
        fig.patch.set_facecolor('white')

        # 统一y轴范围
        y_all = pd.to_numeric(long_df[value_col], errors='coerce').dropna()
        if len(y_all) > 0:
            y_min = np.min(y_all) - 0.2
            y_max = np.max(y_all) + 0.3
        else:
            y_min, y_max = None, None

        for ax, subtype_label in zip(axes, subtype_labels):
            ax.set_facecolor('white')

            sub_df = long_df[long_df['Cluster_Label'] == subtype_label].copy()

            # 个体轨迹
            from matplotlib.colors import Normalize

            # -------- 计算每个被试 baseline --------
            baseline_dict = {}

            for sid, subj_df in sub_df.groupby('SubjectID'):
                subj_df = subj_df.sort_values('time_years').copy()
                y = pd.to_numeric(subj_df[value_col], errors='coerce').dropna()

                if len(y) > 0:
                    baseline_dict[sid] = y.iloc[0]
                else:
                    baseline_dict[sid] = np.nan

            # 去掉 NaN
            baseline_dict = {k: v for k, v in baseline_dict.items() if not np.isnan(v)}

            if len(baseline_dict) == 0:
                continue

            # -------- 排序（从小到大）--------
            sorted_subjects = sorted(baseline_dict.keys(), key=lambda k: baseline_dict[k])

            # -------- 颜色映射 --------
            from matplotlib.colors import LinearSegmentedColormap

            if subtype_label == subtype_labels[0]:
                cmap = LinearSegmentedColormap.from_list(
                    "blue_grad", ["#9ecae1", "#1f77b4"]  # 更深一点的浅蓝
                )
            else:
                cmap = LinearSegmentedColormap.from_list(
                    "red_grad", ["#fcbba1", "#d62728"]  # 更深一点的浅红
                )

            norm = Normalize(
                vmin=min(baseline_dict.values()),
                vmax=max(baseline_dict.values())
            )

            # -------- 绘制轨迹 --------
            for sid in sorted_subjects:
                subj_df = sub_df[sub_df['SubjectID'] == sid].copy()
                subj_df = subj_df.sort_values('time_years')

                y = pd.to_numeric(subj_df[value_col], errors='coerce')
                x = pd.to_numeric(subj_df['time_years'], errors='coerce')

                valid = (~x.isna()) & (~y.isna())

                if valid.sum() >= 2:
                    color = cmap(norm(baseline_dict[sid]))

                    ax.plot(
                        x[valid], y[valid],
                        color=color,
                        linewidth=1,
                        alpha=0.8,
                        zorder=1.5
                    )
                    ax.scatter(
                        x[valid], y[valid],
                        color=color,
                        s=20,
                        alpha=0.8,
                        edgecolor=color,
                        linewidth=0.5,
                        zorder=2
                    )

            # 坐标轴美化
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=5)

            ax.set_xlim(-0.3, 8)
            ax.set_xticks(np.arange(0, 9, 1))

            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)

            ax.set_xlabel("Time from baseline (years)", fontsize=15)
            ax.set_title(str(subtype_label), fontsize=15, pad=5)

        # 左侧子图加 y 轴标签
        axes[0].set_ylabel('Tau SUVR', fontsize=15)

        # 整体标题
        region_name = format_region_name_for_paper(value_col)
        fig.suptitle(f" {region_name}", fontsize=16, y=0.95)

        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_subtype_multi_region_fits(long_df, sig_regions, tracer, out_png,figsize=(9, 6)):
        # 仅适用于两个亚型
        if len(subtype_labels) != 2:
            raise ValueError("当前函数版本仅适用于两个亚型。")

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white', sharex=True, sharey=True)
        fig.patch.set_facecolor('white')

        blue_palette = ["#deebf7", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]

        red_palette = ["#fee0d2", "#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"]

        # 为了后面单独保存 legend，收集 handles/labels
        legend_handle_dict = {}
        legend_labels = []

        for ax, subtype_label in zip(axes, subtype_labels):
            ax.set_facecolor('white')
            sub_df_all = long_df[long_df['Cluster_Label'] == subtype_label].copy()

            if sub_df_all.empty:
                continue

            # ---------- 按 baseline 平均值排序 ----------
            baseline_means = {}
            for region in sig_regions:
                tmp_baseline = sub_df_all[sub_df_all['time_years'] == 0][region].copy()
                tmp_baseline = pd.to_numeric(tmp_baseline, errors='coerce').dropna()

                if len(tmp_baseline) > 0:
                    baseline_means[region] = tmp_baseline.mean()
                else:
                    baseline_means[region] = np.nan

            sig_regions_sorted = sorted(
                sig_regions,
                key=lambda r: (np.isnan(baseline_means[r]), baseline_means[r])
            )

            # ---------- 渐变颜色 ----------
            # 根据亚型选择 palette
            if subtype_label == subtype_labels[0]:
                palette = blue_palette
            else:
                palette = red_palette

            # 如果脑区多于颜色 → 插值扩展
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("custom", palette)

            positions = np.linspace(0, 1, len(sig_regions_sorted))

            region_color_map = {
                region: cmap(p)
                for region, p in zip(sig_regions_sorted, positions)
            }

            # ---------- 画线 ----------
            for region in sig_regions_sorted:
                tmp = sub_df_all[['time_years', region]].copy()
                tmp['time_years'] = pd.to_numeric(tmp['time_years'], errors='coerce')
                tmp[region] = pd.to_numeric(tmp[region], errors='coerce')
                tmp = tmp.dropna(subset=['time_years', region])

                if tmp.shape[0] < 2:
                    continue

                x = tmp['time_years'].values
                y = tmp[region].values

                if np.unique(x).size < 2:
                    continue

                slope, intercept, _, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 200)
                y_fit = intercept + slope * x_fit

                line, = ax.plot(
                    x_fit,
                    y_fit,
                    color=region_color_map[region],
                    linewidth=1.5,
                    label=prettify_region_name(region),
                    zorder=3
                )

                region_name = prettify_region_name(region)

                # 为每个 region 分别保存 subtype1 和 subtype2 的 line handle
                if region_name not in legend_handle_dict:
                    legend_handle_dict[region_name] = {
                        'handles': {},
                        'baseline': baseline_means[region]  # ⭐ 存进去
                    }

                legend_handle_dict[region_name]['handles'][subtype_label] = line

            # ---------- 坐标轴美化 ----------
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5)
            ax.tick_params(labelleft=True)

            ax.set_xlim(-0.3, 8)
            ax.set_xticks(np.arange(0, 9, 1))
            ax.set_ylim(1, 2)
            ax.set_yticks(np.arange(1, 2.01, 0.2))

            ax.set_xlabel("Time from baseline (years)", fontsize=14)
            ax.set_title(f"{subtype_label}", fontsize=16, pad=10)

        # 左图加 y 轴标签
        axes[0].set_ylabel("Tau SUVR", fontsize=14)

        plt.tight_layout()

        # 保存主图
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')

        # ---------- 单独保存 legend ----------
        # 按 region 名称顺序生成 legend，每一行显示：蓝线 + 红线 + 一个标签
        ordered_region_names = sorted(
            legend_handle_dict.keys(),
            key=lambda r: (
                np.isnan(legend_handle_dict[r]['baseline']),
                -legend_handle_dict[r]['baseline']  # ⭐ 从大到小
            )
        )

        legend_handles = []
        legend_labels = []

        for region_name in ordered_region_names:
            h1 = legend_handle_dict[region_name]['handles'].get(subtype_labels[0], None)
            h2 = legend_handle_dict[region_name]['handles'].get(subtype_labels[1], None)

            # 只有两个亚型都存在时才加入；若你想允许单侧存在，也可以放宽
            if h1 is not None and h2 is not None:
                legend_handles.append((h1, h2))
                legend_labels.append(region_name)

        fig_legend = plt.figure(figsize=(3, max(2, len(legend_labels) * 0.23)))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        legend_labels = [format_region_name_for_paper(label) for label in legend_labels]
        ax_legend.legend(
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='center',
            frameon=False,
            fontsize=10,
            handlelength=3.0,
            handletextpad=0.8,
            labelspacing=0.35,
            borderaxespad=0.0,
            ncol=1
        )

        legend_out = out_png.replace('.png', '_legend.png')
        fig_legend.savefig(
            legend_out,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            pad_inches=0.01
        )
        plt.close(fig_legend)

        plt.close()

    # ========= 2. 分tracer处理 =========
    summary_rows = []

    for tracer in tracer_list:
        if tracer is not None:
            tracer_df = suvr_df[suvr_df['TRACER'] == tracer].copy()
            tracer_name = str(tracer)
        else:
            tracer_df = suvr_df.copy()
            tracer_name = "AllTracer"

        if tracer_df.empty:
            continue

        long_df = prepare_long_df_for_tracer(tracer_df)
        if long_df.empty:
            print(f"警告: tracer={tracer_name} 没有足够的纵向数据用于轨迹绘图")
            continue

        global_rate_df, region_rate_df, global_cmp, region_cmp_df = compute_rate_comparison(long_df)

        # 保存中间结果
        long_df.to_csv(
            os.path.join(traj_dir, f"{group_name}_{safe_filename(tracer_name)}_longitudinal_subject_level.csv"),
            index=False,
            encoding='utf-8-sig'
        )

        # ========= global tau 图：保留原逻辑 =========
        if global_cmp is not None and not global_cmp.empty:
            global_cmp['TRACER'] = tracer_name
            global_cmp.to_csv(
                os.path.join(traj_dir, f"{group_name}_{safe_filename(tracer_name)}_global_tau_rate_comparison.csv"),
                index=False,
                encoding='utf-8-sig'
            )

            p_val = global_cmp.loc[0, 'p_value']
            is_sig = p_val < alpha

            if is_sig:
                out_png = os.path.join(
                    traj_dir,
                    f"{safe_filename(tracer_name)}_global_tau.png"
                )
                plot_one_trajectory(
                    long_df=long_df,
                    value_col='global_tau',
                    tracer=tracer_name,
                    p_text=f"global tau rate difference: p = {p_val:.4g}",
                    out_png=out_png
                )
                summary_rows.append({
                    'TRACER': tracer_name,
                    'measure': 'global_tau',
                    'p_value': p_val,
                    'significant': True,
                    'output_png': out_png
                })

        # ========= region 图：保留原逻辑 + 新增 subtype 多脑区拟合图 =========
        region_cmp_df['TRACER'] = tracer_name
        region_cmp_df.to_csv(
            os.path.join(traj_dir, f"{group_name}_{safe_filename(tracer_name)}_region_tau_rate_comparison.csv"),
            index=False,
            encoding='utf-8-sig'
        )

        sig_col = 'p_value'
        sig_regions = region_cmp_df.loc[
            (region_cmp_df[sig_col] < alpha) &
            (region_cmp_df['brain_region'].str.startswith(('CTX_LH', 'CTX_RH'), na=False)),
            'brain_region'
        ].tolist()

        # ---------- 原有逻辑：每个显著region单独出图 ----------
        for region in sig_regions:
            region_p = region_cmp_df.loc[
                region_cmp_df['brain_region'] == region, sig_col
            ].iloc[0]

            plot_df = long_df[['SubjectID', 'Cluster_Label', 'time_years', region]].copy()
            plot_df = plot_df.rename(columns={region: 'plot_value'})

            out_png = os.path.join(
                traj_dir,
                f"{safe_filename(tracer_name)}_{safe_filename(region)}.png"
            )

            plot_one_trajectory(
                long_df=plot_df.rename(columns={'plot_value': region}),
                value_col=region,
                tracer=tracer_name,
                p_text=f"{region} rate difference ({sig_col}) = {region_p:.4g}",
                out_png=out_png
            )

            summary_rows.append({
                'TRACER': tracer_name,
                'measure': region,
                'p_value': region_p,
                'significant': True,
                'output_png': out_png
            })

            # ---------- 新增逻辑：每个亚型一张图，包含所有显著脑区拟合线 ----------
            out_png = os.path.join(
                traj_dir,
                f"{safe_filename(tracer_name)}_all_significant_regions_fit.png"
            )

            plot_subtype_multi_region_fits(
                long_df=long_df,
                sig_regions=sig_regions,
                tracer=tracer_name,
                out_png=out_png
            )

    return

def plot_delta_r2_bar(interaction_df, output_path, alpha=0.05):
    """
    绘制不同network加入后带来的ΔR²（模型解释力提升）条形图

    Parameters
    ----------
    interaction_df : pd.DataFrame
        至少包含以下列：
        - network
        - r2_base
        - r2_full
        - delta_r2
        - network_improves_model (可选)
        - modulation_supported_strict (可选)
    output_path : str
        输出图片路径
    alpha : float
        显著性阈值（目前主要用于可扩展标记）
    """

    plot_df = interaction_df.copy()

    required_cols = ['network', 'r2_base', 'r2_full', 'delta_r2']
    missing_cols = [c for c in required_cols if c not in plot_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"plot_delta_r2_bar 缺少必要列: {missing_cols}")

    # 按 ΔR² 从大到小排序
    plot_df = plot_df.sort_values('delta_r2', ascending=False).reset_index(drop=True)
    plot_df['network'] = plot_df['network'].str.replace('_correlation', '', regex=False)

    # 颜色设置：
    # 若模型整体改善且严格支持interaction，则深色；否则浅灰
    bar_colors = []
    for _, row in plot_df.iterrows():
        improves = bool(row['network_improves_model']) if 'network_improves_model' in plot_df.columns else False
        strict_sig = bool(row['modulation_supported_strict']) if 'modulation_supported_strict' in plot_df.columns else False

        if improves and strict_sig:
            bar_colors.append('#b22222')   # 深红：最值得关注
        elif improves:
            bar_colors.append('#f4a3a3')   # 浅红：整体改善但interaction不严格显著
        else:
            bar_colors.append('#d9d9d9')   # 灰：不支持模型改善

    plt.figure(figsize=(8, 5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    bars = ax.bar(
        plot_df['network'],
        plot_df['delta_r2'],
        color=bar_colors,
        edgecolor='black',
        linewidth=0.8
    )

    # 横线
    ax.axhline(0, color='black', linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=5)
    ax.set_ylim(0, 0.14)
    ax.set_ylabel('ΔR² (Model 2 - Model 1)', fontsize=15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_interaction_beta(interaction_beta_df, output_path, alpha=0.05):
    """
    绘制各 network 的 interaction β 及其 95% CI

    Parameters
    ----------
    interaction_beta_df : pd.DataFrame
        至少包含以下列：
        - network
        - beta
        - ci_low
        - ci_high
        - p_value
    output_path : str
        输出图片路径
    alpha : float
        显著性阈值
    """

    plot_df = interaction_beta_df.copy()

    required_cols = ['network', 'beta', 'ci_low', 'ci_high', 'p_value']
    missing_cols = [c for c in required_cols if c not in plot_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"plot_interaction_beta 缺少必要列: {missing_cols}")

    # 按 beta 从大到小排序
    plot_df = plot_df.sort_values('beta', ascending=False).reset_index(drop=True)
    plot_df['network'] = plot_df['network'].str.replace('_correlation', '', regex=False)

    x = np.arange(len(plot_df))
    y = plot_df['beta'].values
    yerr_low = y - plot_df['ci_low'].values
    yerr_high = plot_df['ci_high'].values - y

    # 颜色：显著深红，不显著灰
    colors = ['#b22222' if p < alpha else '#d9d9d9' for p in plot_df['p_value']]

    plt.figure(figsize=(8, 5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    ax.bar(
        x,
        y,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        width=0.7
    )

    ax.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        fmt='none',
        ecolor='black',
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3
    )

    ax.axhline(0, color='black', linewidth=1)

    # 星号标注显著性
    for i, p in enumerate(plot_df['p_value']):
        if p < alpha:
            y_text = plot_df.loc[i, 'ci_high']
            offset = max(np.nanmax(np.abs(y)) * 0.05, 0.005)
            ax.text(i, y_text + offset, '*', ha='center', va='bottom', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=5)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['network'], rotation=45, ha='right')
    ax.set_ylim(-0.02, 0.03)
    ax.set_ylabel('Interaction β', fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_interaction_curve(
    model,
    df_model,
    tau_rate_col,
    baseline_tau_col,
    coupling_col,
    sq_col,
    output_path,
    age_col='Age',
    sex_col='Sex_Code',
    edu_col='PTEDUCAT',
    hand_col='PTHAND',
    raw_df=None,
    standardize=True,
    x_label=None
):

    # raw_df: 原始数据，用来决定横轴显示范围，以及原始均值/标准差
    # df_model: 实际进入模型的数据（可能已经标准化）
    if raw_df is None:
        raw_df = df_model.copy()

    # ---------- 1. 原始尺度下的横轴范围 ----------
    x_raw_min = raw_df[baseline_tau_col].min()
    x_raw_max = raw_df[baseline_tau_col].max()
    x_raw_vals = np.linspace(x_raw_min, x_raw_max, 200)

    # ---------- 2. 获取原始数据的均值和标准差 ----------
    x_raw_mean = raw_df[baseline_tau_col].mean()
    x_raw_sd = raw_df[baseline_tau_col].std()

    coupling_raw_mean = raw_df[coupling_col].mean()
    coupling_raw_sd = raw_df[coupling_col].std()

    age_raw_mean = raw_df[age_col].mean()
    sex_raw_mean = raw_df[sex_col].mean()
    edu_raw_mean = raw_df[edu_col].mean()
    hand_raw_mode = raw_df[hand_col].mode().iloc[0]

    # ---------- 3. coupling 水平 ----------
    coupling_col_clean = coupling_col.replace('_correlation', '')
    coupling_levels_raw = {
        f'Low {coupling_col_clean}-constrained atrophy (-1 SD)': coupling_raw_mean - coupling_raw_sd,
        f'Mean {coupling_col_clean}-constrained atrophy': coupling_raw_mean,
        f'High {coupling_col_clean}-constrained atrophy (+1 SD)': coupling_raw_mean + coupling_raw_sd
    }

    plt.figure(figsize=(6, 5))

    colors = ['#f4a3a3', '#e06666', '#b22222']

    for (label, c_raw), color in zip(coupling_levels_raw.items(), colors):

        # 如果模型是标准化后拟合的，则预测时也必须标准化
        if standardize:
            x_for_model = (x_raw_vals - x_raw_mean) / x_raw_sd
            c_for_model = (c_raw - coupling_raw_mean) / coupling_raw_sd
            age_for_model = (age_raw_mean - raw_df[age_col].mean()) / raw_df[age_col].std()
            edu_for_model = (edu_raw_mean - raw_df[edu_col].mean()) / raw_df[edu_col].std()

            # sex 通常不要标准化；如果你建模时 sex 也没标准化，就直接保留原值
            sex_for_model = sex_raw_mean
            hand_for_model = hand_raw_mode
        else:
            x_for_model = x_raw_vals
            c_for_model = c_raw
            age_for_model = age_raw_mean
            sex_for_model = sex_raw_mean
            edu_for_model = edu_raw_mean
            hand_for_model = hand_raw_mode

        pred_df = pd.DataFrame({
            baseline_tau_col: x_for_model,
            sq_col: x_for_model ** 2,
            coupling_col: c_for_model,
            age_col: age_for_model,
            sex_col: sex_for_model,
            edu_col: edu_for_model,
            hand_col: hand_for_model
        })

        y_pred = model.predict(pred_df)
        plt.plot(
            x_raw_vals,
            y_pred,
            linewidth=1.5,
            label=label,
            color=color
        )

    if x_label is None:
        x_label = f'{baseline_tau_col} (raw)'

    plt.xlabel(x_label)
    plt.ylabel('Model-estimated tau accumulation rate')
    plt.legend(
        frameon=True,
        framealpha=1,
        edgecolor='black',
        facecolor='white',
        fancybox=False,  # 去掉圆角 → 更期刊风
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_subtype_quadratic_curves(
    df,
    baseline_tau_col,
    tau_rate_col,
    subtype_col,
    output_path, colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
):
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    plot_df = df[[baseline_tau_col, tau_rate_col, subtype_col]].dropna().copy()
    subtype_labels = sorted(plot_df[subtype_col].unique())

    plt.figure(figsize=(6, 5))

    for i, subtype in enumerate(subtype_labels):
        sub_df = plot_df[plot_df[subtype_col] == subtype].copy()
        if sub_df.shape[0] < 3:
            continue

        plt.scatter(
            sub_df[baseline_tau_col],
            sub_df[tau_rate_col],
            alpha=0.7,
            s=35,
            color=colors[i % len(colors)],
            label=f'{subtype}'
        )

        sq_col = f"{baseline_tau_col}_sq"
        sub_df[sq_col] = sub_df[baseline_tau_col] ** 2

        model_sub = smf.ols(
            f"{tau_rate_col} ~ {baseline_tau_col} + {sq_col}",
            data=sub_df
        ).fit()

        x_vals = np.linspace(
            sub_df[baseline_tau_col].min(),
            sub_df[baseline_tau_col].max(),
            200
        )
        pred_df = pd.DataFrame({
            baseline_tau_col: x_vals,
            sq_col: x_vals ** 2
        })
        y_pred = model_sub.predict(pred_df)

        plt.plot(
            x_vals,
            y_pred,
            color=colors[i % len(colors)],
            linewidth=2.5
        )

    plt.xlabel('Baseline tau level')
    plt.ylabel('Tau accumulation rate')
    plt.legend(
        frameon=True,
        framealpha=1,
        edgecolor='black',
        facecolor='white',
        fancybox=False,  # 去掉圆角 → 更期刊风
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
