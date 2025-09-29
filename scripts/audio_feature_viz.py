from visualization_utils import (
    plt, sns, mdates, withStroke, np, fig_text, pd
)

def get_feature_colors(n_colors):
    """Get color palette for features."""
    return sns.color_palette("husl", n_colors)

def plot_feature_distributions(df, features, title, ncols=3):
    """Create distribution plots for features."""
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    colors = ["#7570B3", "#1B9E77"]
    sns.set_palette(colors)
    
    for idx, feature in enumerate(features):
        row = idx // ncols
        col = idx % ncols
        
        sns.boxplot(
            data=df.to_pandas(),
            x='is_winner',
            y=feature,
            ax=axes[row, col],
            hue='is_winner'
        )
        axes[row, col].xaxis.label.set_visible(False)
        axes[row, col].yaxis.label.set_visible(False)
        axes[row, col].get_xaxis().set_visible(False)
        axes[row, col].set_title(feature.capitalize())
        axes[row, col].get_legend().set_visible(False)
    
    # Hide empty subplots
    for idx in range(len(features), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row, col])
    
    fig_text(
        s=f'{title}: \n<Non-Winners> vs <Winners>',
        x=0.5, y=.95,
        fontsize=20,
        color='black',
        highlight_textprops=[
            {"color": colors[0], 'fontweight': 'bold'},
            {"color": colors[1], 'fontweight': 'bold'}
        ],
        ha='center'
    )
    
    plt.subplots_adjust(top=0.8) if nrows == 1 else plt.subplots_adjust(top=0.88)
    plt.show()

def plot_temporal_trends(analysis_df, feature_group, feature_groups, title):
    """Plot temporal trends for a feature group."""
    # Convert to pandas for easier time series handling
    from audio_feature_analysis import create_feature_df, analyze_seasonality
    df = create_feature_df(analysis_df, feature_groups, feature_group).to_pandas()
    
    # Add and format date information
    df['date'] = analysis_df.select('date').to_pandas()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    # Set color palette for features
    feature_colors = get_feature_colors(len(feature_groups[feature_group]['names']))
    winner_styles = ['-', '--']  # solid for winners, dashed for non-winners
    
    # Create subplots
    fig, axes = plt.subplots(len(feature_groups[feature_group]['names']), 1, 
                            figsize=(15, 5*len(feature_groups[feature_group]['names'])))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for idx, (feature, color) in enumerate(zip(feature_groups[feature_group]['names'], feature_colors)):
        # Calculate monthly averages
        winners_avg = df[df['is_winner'] == 1].groupby(pd.Grouper(key='date', freq='M'))[feature].mean()
        non_winners_avg = df[df['is_winner'] == 0].groupby(pd.Grouper(key='date', freq='M'))[feature].mean()
        
        # Plot trends with distinct styles
        axes[idx].plot(winners_avg.index, winners_avg.rolling(window=3).mean(),
                      label=f'Winners - {feature}', color=color, 
                      linestyle=winner_styles[0], linewidth=2)
        axes[idx].plot(non_winners_avg.index, non_winners_avg.rolling(window=3).mean(),
                      label=f'Non-Winners - {feature}', color=color, 
                      linestyle=winner_styles[1], linewidth=2)
        
        # Get y-axis limits to place annotations properly
        y_min, y_max = axes[idx].get_ylim()
        annotation_space = (y_max - y_min) * 0.1  # 10% of y-axis range for spacing

        # Add annotations with consistent positioning
        axes[idx].annotate('─── Winners', 
                        xy=(0.02, 0.95), 
                        xycoords='axes fraction',
                        color=color, 
                        fontweight='bold', 
                        fontsize=10,
                        va='center')
        axes[idx].annotate('- - - Non-Winners', 
                        xy=(0.02, 0.89), 
                        xycoords='axes fraction',
                        color=color, 
                        fontweight='bold', 
                        fontsize=10,
                        va='center',
                        path_effects=[withStroke(linewidth=3, foreground='white')])
        
        # Format x-axis
        axes[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        
        # Customize appearance
        axes[idx].set_title(f'{feature.capitalize()}', fontsize=12, pad=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylabel('Value')
    
    # Add overall title
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    return analyze_seasonality(df, feature_groups[feature_group]['names'])

def plot_genre_distributions(genre_df, top_n=10, title="Primary", genre_col="genre_1st"):
    """Plot genre distributions comparing winners and non-winners."""
    top_genres = list(
        genre_df[genre_col]
        .value_counts()
        .top_k(top_n, by='count')[genre_col]
    )

    filtered_df = genre_df.filter(genre_df[genre_col].is_in(top_genres))

    plt.figure(figsize=(12, 6))

    ax = sns.countplot(data=filtered_df,
                  y=genre_col,
                  hue='is_winner',
                  order=top_genres,
                  palette=["#7570B3", "#1B9E77"],
                  legend=False
                  )
    fig_text(
        s=f'{title}: <Non-Winners> vs <Winners>',
        x=0.4, y=.95,
        fontsize=16,
        color='black',
        highlight_textprops=[
            {"color": "#7570B3", 'fontweight': 'bold'},
            {"color": "#1B9E77", 'fontweight': 'bold'}
        ],
        ha='center'
    )

    for genre in top_genres:
        subset = genre_df.filter(genre_df['sorted_combo'] == genre)
        total = subset['sorted_combo'].count()
        y_pos = list(top_genres).index(genre)  
        x_pos = subset['is_winner'].value_counts().max()['count'][0]
        ax.text(x=x_pos + 1.5, y= y_pos, s=f"{subset['is_winner'].sum()/total:.2f}", ha='center', color='gray')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(ylabel=None, xlabel=None)
    
    plt.show()

def plot_correlation_matrix(all_features):
    """Plot correlation matrix for all features."""
    correlation_matrix = all_features.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                square=True,
                mask=mask)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    return correlation_matrix