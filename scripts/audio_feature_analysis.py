import numpy as np
import scipy.stats
import pandas as pd
import polars as pl

def create_feature_df(analysis_df, feature_groups, feature_group):
    """Create a dataframe with selected features and proper column names."""
    features = feature_groups[feature_group]['features'] + ['is_winner']
    names = feature_groups[feature_group]['names']
    
    df = analysis_df.select(features)
    rename_dict = dict(zip(features, names))
    return df.rename(rename_dict)

def analyze_features(df, features, group_name):
    """Perform statistical analysis on features."""
    print(f"\n=== {group_name} Analysis ===")
    
    results = []
    for feature in features:
        winners = df.filter(pl.col("is_winner") == 1)[feature]
        non_winners = df.filter(pl.col("is_winner") == 0)[feature]
        
        stat, pval = scipy.stats.ttest_ind(
            winners.to_pandas(), 
            non_winners.to_pandas()
        )
        
        # Calculate effect size (Cohen's d)
        n1, n2 = len(winners), len(non_winners)
        var1, var2 = winners.var(), non_winners.var()
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = abs(winners.mean() - non_winners.mean()) / pooled_se
        
        results.append({
            'feature': feature.capitalize(),
            'statistic': stat,
            'p_value': pval,
            'significant': pval < 0.05,
            'effect_size': cohen_d
        })
    
    # Create a formatted table of results
    results_df = pd.DataFrame(results)
    results_df['p_value'] = results_df['p_value'].apply(lambda x: f"{x:.4f}")
    results_df['effect_size'] = results_df['effect_size'].apply(lambda x: f"{x:.4f}")
    return results_df

def analyze_seasonality(df, features):
    import matplotlib.pyplot as plt
    """Analyze seasonal patterns in features."""
    seasonal_patterns = pd.DataFrame()
    
    # Get the same color palette as temporal trends
    from audio_feature_viz import get_feature_colors
    feature_colors = get_feature_colors(len(features))
    winner_styles = ['-', '--']  # solid for winners, dashed for non-winners
    
    for feature in features:
        monthly_avg = df.groupby(['month', 'is_winner'])[feature].mean().unstack()
        seasonal_patterns[f'{feature}_winners'] = monthly_avg[1]
        seasonal_patterns[f'{feature}_non_winners'] = monthly_avg[0]

    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot each feature with consistent colors
    for idx, feature in enumerate(features):
        # Plot winners (solid line)
        plt.plot(range(1, 13), seasonal_patterns[f'{feature}_winners'],
                color=feature_colors[idx], linestyle=winner_styles[0],
                linewidth=2, label=f'{feature.capitalize()}')
        
        # Plot non-winners (dashed line)
        plt.plot(range(1, 13), seasonal_patterns[f'{feature}_non_winners'],
                color=feature_colors[idx], linestyle=winner_styles[1],
                linewidth=2)
        
        # Add direct labels at the end of lines
        last_winner = seasonal_patterns[f'{feature}_winners'].iloc[-1]
        last_non_winner = seasonal_patterns[f'{feature}_non_winners'].iloc[-1]
        
        plt.annotate(feature.capitalize(), 
                    xy=(12, last_winner),
                    xytext=(12.1, last_winner),
                    color=feature_colors[idx],
                    fontweight='bold')
    
    # Customize appearance
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3)
    plt.xlabel('Month')
    plt.ylabel('Average Value')
    
    # Remove legend since we're using direct labels
    plt.xlim(1, 13.5)  # Extend x-axis slightly for labels
    plt.tight_layout()
    
    return seasonal_patterns

def parse_genre(genre_string):
    """Parse genre string into parent and subgenre."""
    parts = genre_string.split('---')
    parent = parts[0].strip()
    subgenre = parts[1].strip() if len(parts) > 1 else None
    return parent, subgenre

def analyze_genre_win_rates(genre_df, top_n=10, genre_col='genre_1st', title='Primary'):
    """Calculate win rates per genre."""
    return (genre_df.group_by(genre_col)
                   .agg([
                       pl.count('is_winner').alias('count'),
                       pl.mean('is_winner').alias('mean')
                   ])
                   .top_k(top_n, by='mean'))