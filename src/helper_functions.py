import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import List, Optional, Tuple
import seaborn as sns

# ==============================
# Summarize DataFrame
# ==============================

def summarize_dataframe(
    dataframe, 
    head=True, 
    head_size=5, 
    sample_size=5, 
    top_k=5
):
    """
    Provides a detailed overview of a pandas DataFrame, including:
    
    - Numerical columns: descriptive stats, percentiles, IQR, outlier bounds, outlier counts/proportions
    - Categorical/boolean columns: unique values, top-k frequent values
    - Optionally displays head or random sample

    Args:
        dataframe (pd.DataFrame): DataFrame to inspect.
        head (bool): If True, display the head; if False, display a random sample.
        head_size (int): Number of rows to display from head.
        sample_size (int): Number of rows to display from a random sample.
        top_k (int): Number of top frequent values to display for categorical columns.

    Returns:
        None
    """

    print("===== DATA OVERVIEW =====")
    overview = pd.DataFrame({
        'Num NAs': dataframe.isna().sum(),
        'Percent NAs': (dataframe.isna().mean() * 100).round(2),
        'Num unique': dataframe.nunique(),
        'Data Type': dataframe.dtypes
    })
    display(overview)
    print(f"\nDataFrame shape: {dataframe.shape}\n")

    # ======= NUMERICAL STATS =======
    df_numeric = dataframe.select_dtypes(include='number')
    if not df_numeric.empty:
        numeric_summary = []

        for col in df_numeric.columns:
            series = df_numeric[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            percentiles = series.quantile([0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]).to_dict()

            numeric_summary.append({
                'attribute': col,
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis(),
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'num_outliers': outliers.count(),
                'prop_outliers_%': (outliers.count() / series.count() * 100).round(2),
                'pct_2': percentiles[0.02],
                'pct_5': percentiles[0.05],
                'pct_10': percentiles[0.10],
                'pct_25': percentiles[0.25],
                'pct_50': percentiles[0.50],
                'pct_75': percentiles[0.75],
                'pct_90': percentiles[0.90],
                'pct_95': percentiles[0.95],
                'pct_98': percentiles[0.98]
            })

        df_stats = pd.DataFrame(numeric_summary)
        display(df_stats.round(2))

    # ======= CATEGORICAL / BOOLEAN STATS =======
    df_cat = dataframe.select_dtypes(exclude='number')
    if not df_cat.empty:
        print("===== CATEGORICAL / BOOLEAN STATS =====")
        for col in df_cat.columns:
            print(f"\nColumn: {col}")
            value_counts = dataframe[col].value_counts(dropna=False)
            top_values = value_counts.head(top_k)
            display(pd.DataFrame({
                'Value': top_values.index,
                'Count': top_values.values,
                'Percent': (top_values / len(dataframe) * 100).round(2)
            }))

    # ======= DATAFRAME SAMPLES =======
    if head:
        print(f"\nDataFrame head ({head_size} rows):")
        display(dataframe.head(head_size))
    else:
        print(f"\nRandom sample ({sample_size} rows):")
        display(dataframe.sample(sample_size))

# ==============================
# Boxplot + Outliers by Group
# ==============================

def boxplot_outliers_groups(df, column, label_col,
                            outlier_color='red', group_colors=None,
                            title=None, percentile_filter=(1, 99),
                            show_mean=True, figsize=(12, 6),
                            show_y_axis=True, width_ratios=[1, 2, 4]):
    """
    Plot a single row: bar chart, pie chart, horizontal boxplot
    - Bar chart: % of each group that are global outliers
    - Pie chart: distribution of global outliers
    - Horizontal boxplot: filtered by percentile

    Works with any binary group (or more than two groups if needed)
    """

    # Identify unique groups dynamically
    groups = sorted(df[label_col].unique())
    group_titles = [str(g) for g in groups]

    # Set group colors if not provided
    if group_colors is None:
        # Repeat colors if fewer than number of groups
        default_colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'violet']
        group_colors = default_colors[:len(groups)]

    # Compute global outliers BEFORE filtering
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers_mask = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)

    # Bar chart: fraction of each group that are outliers
    bar_values = [(outliers_mask[df[label_col] == g].sum() / 
                   (df[label_col] == g).sum()) for g in groups]

    # Pie chart: distribution of global outliers
    outliers_per_group = [outliers_mask[df[label_col] == g].sum() for g in groups]

    # Filter by percentiles
    low, high = percentile_filter
    filtered = df[(df[column] >= df[column].quantile(low / 100)) &
                  (df[column] <= df[column].quantile(high / 100))]

    # Range after filter
    min_val, max_val = filtered[column].min(), filtered[column].max()

    # Plot layout
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': width_ratios})

    # --- Bar chart
    axes[0].bar(group_titles, bar_values, color=outlier_color)
    for i, v in enumerate(bar_values):
        axes[0].text(i, v + 0.01, f'{v*100:.1f}%', ha='center')
    axes[0].set_ylabel('Fraction of group that are outliers')
    if not show_y_axis:
        axes[0].spines['left'].set_visible(False)
        axes[0].tick_params(left=False)

    # --- Pie chart
    axes[1].pie(outliers_per_group, labels=group_titles,
                colors=[outlier_color] * len(groups), autopct='%1.1f%%')
    axes[1].set_title('Distribution of global outliers')

    # --- Horizontal boxplot
    bplot = axes[2].boxplot(filtered[column], vert=False, patch_artist=True,
                            flierprops=dict(marker='o', markerfacecolor=outlier_color,
                                            markersize=6, linestyle='none'),
                            boxprops=dict(facecolor=group_colors[0], color='black'))

    if show_mean:
        mean_value = df[column].mean()  # global mean
        if min_val <= mean_value <= max_val:  # only plot if inside filter
            box_patch = bplot['boxes'][0]
            verts = box_patch.get_path().vertices
            y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
            offset = (y_max - y_min) * 0.05
            axes[2].plot([mean_value, mean_value], [y_min, y_max],
                         color='k', linestyle='--', linewidth=2)
            axes[2].text(mean_value, y_max + offset, 'mean', va='bottom', ha='left',
                         fontsize=10, fontweight='bold', color='k')

    axes[2].set_title(f'Boxplot of {column} ({low}-{high} percentile)')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()

# ==============================
# Histogram by Groups
# ==============================

def histogram_groups(df, column, label_col,
                     outlier_color='red', group_colors=None,
                     percentile_filter=(1, 99),
                     show_mean=True, show_median=True,
                     bins=30, figsize=(15, 5),
                     title=None):
    """
    Plot histograms:
    - All data (filtered distribution, but statistics from full df).
    - Each group (global statistics for each group)
    - Mean/median lines shown if inside filtered range.

    Works with any binary or categorical groups.
    """

    # --- Global outliers ---
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers_mask = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)

    # --- Groups ---
    groups = sorted(df[label_col].dropna().unique())
    group_titles = [str(g) for g in groups]

    if group_colors is None:
        default_colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'violet']
        group_colors = default_colors[:len(groups)]

    # --- Global statistics ---
    stats_global = {}
    stats_global["all"] = {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "p1": df[column].quantile(0.01),
        "p99": df[column].quantile(0.99)
    }
    for g in groups:
        subset = df.loc[df[label_col] == g, column]
        stats_global[g] = {
            "mean": subset.mean(),
            "median": subset.median(),
            "p1": subset.quantile(0.01),
            "p99": subset.quantile(0.99)
        }

    # --- Apply percentile filter ---
    filtered = df
    if percentile_filter and percentile_filter != (0, 100):
        low, high = percentile_filter
        filtered = df[(df[column] >= df[column].quantile(low / 100)) &
                      (df[column] <= df[column].quantile(high / 100))]

    # --- Bins range ---
    all_data = filtered[column].dropna()
    min_val, max_val = all_data.min(), all_data.max()
    if min_val == max_val:
        min_val, max_val = min_val - 0.5, max_val + 0.5
    bins_edges = np.linspace(min_val, max_val, bins + 1)

    # --- Figure ---
    fig, axes = plt.subplots(1, 1 + len(groups), figsize=figsize, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # -------- All data --------
    ax = axes[0]
    if len(all_data) > 0:
        weights = np.ones(len(all_data)) / len(all_data) * 100
        _, _, patches = ax.hist(all_data, bins=bins_edges,
                                color='lightgray', weights=weights)
        # Highlight global outliers
        outlier_vals = df.loc[outliers_mask, column]
        for patch, left, right in zip(patches, bins_edges[:-1], bins_edges[1:]):
            if ((left <= outlier_vals) & (outlier_vals < right)).any():
                patch.set_facecolor(outlier_color)

    ax.set_title("All data")
    ax.set_ylabel("Distribution (%)")

    # Legend for all data
    legend_lines, legend_labels = [], []
    if show_mean and min_val <= stats_global["all"]["mean"] <= max_val:
        line_mean = ax.axvline(stats_global["all"]["mean"], color='k', linestyle='--')
        legend_lines.append(line_mean)
        legend_labels.append(f"mean = {stats_global['all']['mean']:.2f}")
    if show_median and min_val <= stats_global["all"]["median"] <= max_val:
        line_med = ax.axvline(stats_global["all"]["median"], color='k', linestyle=':')
        legend_lines.append(line_med)
        legend_labels.append(f"median = {stats_global['all']['median']:.2f}")
    legend_lines.append(Patch(color=outlier_color))
    legend_labels.append("Global outliers")
    ax.legend(legend_lines, legend_labels)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------- Groups --------
    for i, g in enumerate(groups):
        ax = axes[i + 1]
        group_filtered = filtered.loc[filtered[label_col] == g, column].dropna()
        if len(group_filtered) > 0:
            weights = np.ones(len(group_filtered)) / len(group_filtered) * 100
            _, _, patches = ax.hist(group_filtered, bins=bins_edges,
                                    color=group_colors[i % len(group_colors)],
                                    weights=weights)
            # Highlight group outliers
            outlier_vals = df.loc[outliers_mask & (df[label_col] == g), column]
            for patch, left, right in zip(patches, bins_edges[:-1], bins_edges[1:]):
                if ((left <= outlier_vals) & (outlier_vals < right)).any():
                    patch.set_facecolor(outlier_color)

        ax.set_title(group_titles[i])

        # Legend for group
        legend_lines, legend_labels = [], []
        if show_mean and min_val <= stats_global[g]["mean"] <= max_val:
            line_mean = ax.axvline(stats_global[g]["mean"], color='k', linestyle='--')
            legend_lines.append(line_mean)
            legend_labels.append(f"mean = {stats_global[g]['mean']:.2f}")
        if show_median and min_val <= stats_global[g]["median"] <= max_val:
            line_med = ax.axvline(stats_global[g]["median"], color='k', linestyle=':')
            legend_lines.append(line_med)
            legend_labels.append(f"median = {stats_global[g]['median']:.2f}")
        legend_lines.append(Patch(color=outlier_color))
        legend_labels.append("Global outliers")
        ax.legend(legend_lines, legend_labels)

        for spine in ax.spines.values():
            spine.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()


# ==============================
# Histogram for single dataset
# ==============================

def plot_histogram_single(
    data,
    bins: int = -1,
    figsize: tuple = (8, 5),
    show_mean: bool = True,
    show_median: bool = True,
    color: str = 'skyblue',
    title: str = None,
    xlabel: str = "Value",
    density: bool = True
):
    """
    Plot clean histogram for a single dataset (e.g., posterior lift distribution).

    Args:
        data (array-like): Samples to plot
        bins (int): Number of histogram bins (-1 = auto)
        figsize (tuple): Figure size
        show_mean (bool): Draw vertical line for mean
        show_median (bool): Draw vertical line for median
        color (str): Histogram color
        title (str): Plot title
        xlabel (str): X-axis label
        density (bool): Normalize to density if True
    """
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if len(data) == 0:
        raise ValueError("Input data is empty or invalid.")

    # --- Auto bins (Freedman–Diaconis rule) ---
    if bins == -1:
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        n = len(data)
        bins = int((data.max() - data.min()) / (2 * iqr / n ** (1/3))) if iqr > 0 else 30

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, color=color, alpha=0.7, density=density)

    if show_mean:
        ax.axvline(data.mean(), color='k', linestyle='--', label=f"Mean = {data.mean():.4f}")
    if show_median:
        ax.axvline(np.median(data), color='r', linestyle=':', label=f"Median = {np.median(data):.4f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Frequency")

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # --- Clean axes ---
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()

# ==============================
# Histogram data1 vs data2 
# ==============================

def plot_histograms_data_1vsdata_2(
    samples_dict_or_df,
    group_names: list = None,
    bins: int = -1,
    figsize: tuple = (12, 5),
    show_mean: bool = True,
    show_median: bool = True,
    colors: list = None,
    title: str = None
):
    """
    Plot clean histograms for multiple sample groups (e.g. posterior distributions).

    Args:
        samples_dict_or_df (dict or pd.DataFrame): 
            - dict: {'group_label': np.array(samples)}
            - DataFrame: each column is a group
        group_names (list): Optional list of group order to plot
        bins (int): Number of histogram bins (-1 = auto)
        figsize (tuple): Figure size
        show_mean (bool): Draw vertical line for mean
        show_median (bool): Draw vertical line for median
        colors (list): Custom colors per group
        title (str): Overall figure title
    """

    # --- Normalize input ---
    if isinstance(samples_dict_or_df, pd.DataFrame):
        samples_dict = {col: samples_dict_or_df[col].dropna().values 
                        for col in samples_dict_or_df.columns}
    elif isinstance(samples_dict_or_df, dict):
        samples_dict = {k: np.asarray(v) for k, v in samples_dict_or_df.items()}
    else:
        raise TypeError("Input must be a dict or DataFrame.")

    # --- Groups ---
    if group_names is None:
        group_names = list(samples_dict.keys())
    n_groups = len(group_names)
    
    # --- Colors ---
    if colors is None:
        default_colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'violet']
        colors = default_colors[:n_groups]

    # --- Auto bins (Freedman–Diaconis rule) ---
    if bins == -1:
        bin_counts = []
        for g in group_names:
            data = np.asarray(samples_dict[g])
            if len(data) < 2:
                continue
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            n = len(data)
            if iqr > 0:
                bin_counts.append(int((data.max() - data.min()) / (2 * iqr / n ** (1/3))))
        bins = int(np.mean(bin_counts)) if bin_counts else 30

    # --- Plot ---
    fig, axes = plt.subplots(1, n_groups, figsize=figsize, sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, g, c in zip(axes, group_names, colors):
        data = np.asarray(samples_dict[g])
        ax.hist(data, bins=bins, color=c, alpha=0.7, density=True)

        if show_mean:
            ax.axvline(data.mean(), color='k', linestyle='--', label=f"Mean = {data.mean():.3f}")
        if show_median:
            ax.axvline(np.median(data), color='r', linestyle=':', label=f"Median = {np.median(data):.3f}")

        ax.set_title(str(g))
        ax.set_xlabel("Value")
        ax.set_ylabel("Density") 
        
        # --- Clean axes ---
        ax.legend(frameon=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==============================
# Categorical Distribution by Groups
# ==============================

def categorical_distribution_groups(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    top_k: Optional[int] = None,
    group_titles: Optional[Tuple[str, str]] = None,
    global_normalize: bool = False,
    normalize_within_group : bool= False,
    order_by_group: Optional[str] = None
) -> None:
    """
    Generates side-by-side bar plots for categorical variables, separated by a binary target.
    The X-axis is ordered by the proportion of the specified group (order_by_group).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    categorical_cols : List[str]
        List of categorical columns to plot
    target_col : str
        Binary target column
    top_k : Optional[int], default=None
        Select top K categories per group for plotting; union is taken
    group_titles : Optional[Tuple[str,str]], default=None
        Custom titles for the two target groups. If None, uses target values
    normalize : bool, default=False
        If True, plot proportions instead of counts
    order_by_group : Optional[str], default=None
        Target value used to order categories on X-axis. If None, uses the second value.
    """
    
    # Validate target
    target_values = df[target_col].dropna().unique()
    if len(target_values) != 2:
        raise ValueError("Target column must be binary (exactly 2 unique values).")
    
    group_0_value, group_1_value = target_values
    
    if group_titles is None:
        group_titles = (str(group_0_value), str(group_1_value))
    
    # Determine which group is used for ordering
    if order_by_group is None:
        order_group_value = group_1_value
    else:
        if order_by_group not in target_values:
            raise ValueError(f"order_by_group must be one of the target values: {target_values}")
        order_group_value = order_by_group
    
    y_label = "Proportion" if (global_normalize or normalize_within_group) else "Count"
    num_rows = len(categorical_cols)
    
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=2,
        figsize=(16, 5 * num_rows),
        gridspec_kw={'wspace': 0.4, 'hspace': 0.4}
    )
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_rows == 0:
        return
    
    for i, col in enumerate(categorical_cols):
        df_plot = df.copy()
        x_label_suffix = ""
        top_k_values: List[str] = []
        
        # --- Top K filtering ---
        if top_k is not None and top_k > 0:
            top_k_0 = df_plot[df_plot[target_col] == group_0_value][col].value_counts().nlargest(top_k).index.tolist()
            top_k_1 = df_plot[df_plot[target_col] == group_1_value][col].value_counts().nlargest(top_k).index.tolist()
            top_k_values = list(set(top_k_0) | set(top_k_1))
            df_plot = df_plot[df_plot[col].isin(top_k_values)]
            
            # Remove unused categories if categorical dtype
            if pd.api.types.is_categorical_dtype(df_plot[col]):
                df_plot[col] = df_plot[col].cat.remove_unused_categories()
            
            num_plotted = len(top_k_values)
            x_label_suffix = f"(Top {top_k} per group, {num_plotted} total categories)"
        
        # --- Count and rate calculation ---
        data_counts = df_plot.groupby([col, target_col]).size().reset_index(name='count')
        total_counts = df_plot[col].value_counts().rename('total_count')
        order_group_counts = data_counts[data_counts[target_col] == order_group_value].set_index(col)['count'].rename('order_group_count')
        
        order_df = pd.concat([total_counts, order_group_counts], axis=1).fillna(0)
        order_df['order_group_rate'] = order_df['order_group_count'] / order_df['total_count']
        plot_order = order_df.sort_values('order_group_rate', ascending=False).index.tolist()
        
        # --- Normalization ---
        y_col = 'proportion'
        value_format = ".1%"
        group_totals = len(df_plot)


        if global_normalize:
            data_counts['proportion'] = data_counts['count']/group_totals 

        elif normalize_within_group:
            data_counts['proportion'] = data_counts.groupby(target_col)['count'].transform(lambda x: x / x.sum())
        
        else:
            y_col = 'count'
            value_format = ",.0f"
        
        # Separate data for two groups
        data_0 = data_counts[data_counts[target_col] == group_0_value]
        data_1 = data_counts[data_counts[target_col] == group_1_value]
        
        # --- Helper plotting function ---
        def plot_and_clean_ax(ax, data, group_title, palette):
            if not data.empty:
                sns.barplot(x=col, y=y_col, data=data, ax=ax, hue=None,palette=palette, order=plot_order)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
            
            ax.set_title(group_title, fontsize=14, loc='center', y=0.98)
            ax.set_ylabel(y_label, fontsize=12, loc='center', labelpad=20)
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.grid(False)
            sns.despine(ax=ax, left=True, bottom=False)
            ax.set_xlabel("")
            
            if not data.empty:
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width()/2., height + 0.005*ax.get_ylim()[1],
                            f'{height:{value_format}}', ha='center', va='bottom', fontsize=10)
            
            ax.tick_params(axis='x', rotation=45)
            if global_normalize or normalize_within_group:
                ax.set_ylim(0, 1.05)
        
        # --- Plot both groups ---
        plot_and_clean_ax(axes[i, 0], data_0, group_titles[0], "Blues_d")
        plot_and_clean_ax(axes[i, 1], data_1, group_titles[1], "Reds_d")


       # --- Feature title ---
        main_title = f"CATEGORICAL FEATURE: {col}"

        # Use axes position dynamically to place title above the row
        row_top = axes[i, 0].get_position().y1  # top of the first axis in the row
        y_offset = 0.05 # small offset above the top of the subplot
        fig.text(
            0.5,                     # center horizontally
            row_top + y_offset,      # just above the row
            main_title,
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold'
        )

        # Optional subtitle (x_label_suffix)
        if x_label_suffix:
            fig.text(
                0.5,
                row_top + y_offset - 0.03,  # slightly below the main title
                x_label_suffix,
                ha='center',
                va='bottom',
                fontsize=12,
                color='gray'
    )
            
    plt.subplots_adjust(
    #top=0.9,    
    hspace=1,  
   
)

    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.4)
    plt.show()


# ==============================
#  Business Performance
# ==============================

def simulate_profit_posterior(
    posterior_samples,
    initial_players,
    ltv_per_player=3,
    bins=50,
    figsize=(6,4),
    colors=None,
    ci=0.95,
    show_plot=True,
    title=None
):

    profit_dict = {
        gate: samples * initial_players * ltv_per_player
        for gate, samples in posterior_samples.items()
    }

    # --- Summary table ---
    alpha = (1 - ci) / 2
    summary = {
        gate: {
            'mean': np.mean(samples),
            'std': np.std(samples),
            f'ci_lower': np.percentile(samples, alpha*100),
            f'ci_upper': np.percentile(samples, (1-alpha)*100)
        } for gate, samples in profit_dict.items()
    }
    summary_df = pd.DataFrame(summary)

    # --- Plot ---
    if show_plot:
        n_gates = len(profit_dict)
        if colors is None:
            colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'violet'][:n_gates]

        fig, axes = plt.subplots(1, n_gates, figsize=(figsize[0]*n_gates, figsize[1]), sharey=True)
        if n_gates == 1:
            axes = [axes]

        for ax, (gate, samples), c in zip(axes, profit_dict.items(), colors):
            ax.hist(samples, bins=bins, density=True, alpha=0.7, color=c)
            mean_val = np.mean(samples)
            ci_low = np.percentile(samples, alpha*100)
            ci_high = np.percentile(samples, (1-alpha)*100)
            
            ax.axvline(mean_val, color='k', linestyle='--', label=f"Mean: ${mean_val:,.0f}")
            ax.axvline(ci_low, color='r', linestyle=':', label=f"{int(ci*100)}% CI")
            ax.axvline(ci_high, color='r', linestyle=':')
            ax.set_title(gate)
            ax.set_xlabel("Profit ($)")
            ax.set_yticks([])
            ax.legend(frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return summary_df