import polars as pl
import pandas as pd
import numpy as np
from visualization_utils import plt, sns
from radar_factory import radar_factory

def parse_genre(genre_string):
    """Parse genre string into parent and subgenre."""
    parts = genre_string.split('---')
    parent = parts[0].strip()
    subgenre = parts[1].strip() if len(parts) > 1 else None
    return parent, subgenre

def create_genre_df(analysis_df):
    """Create a dataframe with genre information."""
    genre_df = analysis_df[['artist', 'song', 'audio_features__genre__top_1_class', 
                          'audio_features__genre__top_2_class', 'audio_features__genre__top_3_class', 
                          'is_winner']]
    genre_df = genre_df.rename({
        'audio_features__genre__top_1_class': 'genre_1st',
        'audio_features__genre__top_2_class': 'genre_2nd',
        'audio_features__genre__top_3_class': 'genre_3rd'
    })

    genre_df = genre_df.with_columns(
        pl.col('genre_1st').map_elements(parse_genre).alias('genre_1st_split'),
        pl.col('genre_2nd').map_elements(parse_genre).alias('genre_2nd_split')
    ).with_columns(
        pl.col('genre_1st_split').list.get(0).alias('genre_1st_parent'),
        pl.col('genre_1st_split').list.get(1).alias('genre_1st_subgenre'),
        pl.col('genre_2nd_split').list.get(0).alias('genre_2nd_parent'),
        pl.col('genre_2nd_split').list.get(1).alias('genre_2nd_subgenre')
    ).with_columns(
        pl.when(pl.col('genre_1st_subgenre') < pl.col('genre_2nd_subgenre'))
        .then(pl.concat_str([pl.col('genre_1st_subgenre'), pl.col('genre_2nd_subgenre')], separator=' + '))
        .otherwise(pl.concat_str([pl.col('genre_2nd_subgenre'), pl.col('genre_1st_subgenre')], separator=' + '))
        .alias('sorted_combo')
    ).drop(['genre_1st_split', 'genre_2nd_split'])

    return  genre_df

def create_temporal_genre(temporal, audio):
    combo_dates = temporal.join(audio, left_on=["artist", "song"], right_on=["artist", "song_title"], how="inner").with_columns([
        pl.when(pl.col("placement") == 1).then(1).otherwise(0).alias("is_winner")
    ])

    combo_dates = combo_dates[['date', 'artist', 'song', 'audio_features__genre__top_1_class', 'audio_features__genre__top_2_class', 'audio_features__genre__top_3_class', 'is_winner']]
    combo_dates = combo_dates.rename({'audio_features__genre__top_1_class': 'genre_1st', 'audio_features__genre__top_2_class': 'genre_2nd', 'audio_features__genre__top_3_class': 'genre_3rd'})
    combo_dates = combo_dates.with_columns(
        pl.col('genre_1st').map_elements(parse_genre).alias('genre_1st_split'),
        pl.col('genre_2nd').map_elements(parse_genre).alias('genre_2nd_split')
        ).with_columns(
            pl.col('genre_1st_split').list.get(0).alias('genre_1st_parent'),
            pl.col('genre_1st_split').list.get(1).alias('genre_1st_subgenre'),
            pl.col('genre_2nd_split').list.get(0).alias('genre_2nd_parent'),
            pl.col('genre_2nd_split').list.get(1).alias('genre_2nd_subgenre'),
            pl.col('date').str.to_date(format="%Y-%m-%d %H:%M:%S")
        ).with_columns(
            pl.when(pl.col('genre_1st_subgenre') < pl.col('genre_2nd_subgenre'))
            .then(pl.concat_str([pl.col('genre_1st_subgenre'), pl.col('genre_2nd_subgenre')], separator=' + '))
            .otherwise(pl.concat_str([pl.col('genre_2nd_subgenre'), pl.col('genre_1st_subgenre')], separator=' + '))
            .alias('sorted_combo'),
            pl.col('date').dt.year().alias('year')
        ).drop(['genre_1st_split', 'genre_2nd_split'])

    return combo_dates

def analyze_genre_combinations(genre_df):
    """Analyze genre combinations and their win rates."""

    return (genre_df.group_by('sorted_combo')
            .agg([
                pl.col('is_winner').sum().alias('wins'),
                pl.col('is_winner').count().alias('total_appearances'),
                pl.col('is_winner').mean().alias('win_rate')
            ]).with_columns(
                (pl.col('win_rate') * 100).round(2).alias('win_percentage')
            ))

def analyze_temporal_genre_trends(combo_dates):
    """Analyze how genre performance changes over time."""
    temporal_combo = (combo_dates
        .group_by(['sorted_combo', 'year'])  
        .agg([
            pl.struct(['artist', 'song']).n_unique().alias('unique_songs'),
            pl.col('is_winner').sum().alias('wins'),
            pl.col('is_winner').count().alias('total_placements'),
            pl.col('is_winner').mean().alias('win_rate')
        ])
        .with_columns(
            (pl.col('win_rate') * 100).round(2).alias('win_percentage')
        ))
    
    return temporal_combo

def find_flash_in_pan_genres(genre_timeline, combo_dates, min_entries=20, max_years=2):
    """Find genres that had high success but short lifespans."""
    return (genre_timeline
        .filter((pl.col('years_active') <= max_years) & 
                (pl.col('total_entries') >= min_entries))
        .join(
            combo_dates.group_by('sorted_combo')
            .agg(pl.col('is_winner').mean().alias('overall_win_rate')),
            on='sorted_combo'
        )
        .sort('overall_win_rate', descending=True)
    )

def find_golden_periods(temporal_combo, min_placements=5, threshold=15):
    """Find periods where genres significantly outperformed their historical average."""
    return (temporal_combo
        .filter(pl.col('total_placements') >= min_placements)
        .with_columns([
            pl.col('win_percentage').mean().over('sorted_combo')
            .alias('genre_historical_avg')
        ])
        .filter(pl.col('win_percentage') >= pl.col('genre_historical_avg') + threshold)
        .sort(['year', 'win_percentage'])
    )

def plot_genre_temporal_trends(temporal_combo, genre_timeline, min_entries=20):
    """Plot temporal trends for genre combinations with sufficient data."""
    
    # Filter for genres with enough entries
    significant_genres = (genre_timeline
        .filter(pl.col('total_entries') >= min_entries)
        .select('sorted_combo')
    )
    
    # Get trends for significant genres
    trends = temporal_combo.filter(
        pl.col('sorted_combo').is_in(significant_genres['sorted_combo'])
    )
    
    # Create year-based plot
    plt.figure(figsize=(15, 8))
    
    # Plot each genre's trend
    for genre in significant_genres['sorted_combo']:
        genre_data = trends.filter(pl.col('sorted_combo') == genre)
        
        plt.plot(
            genre_data['year'],
            genre_data['win_percentage'],
            label=genre,
            marker='o',
            alpha=0.7
        )
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Win Percentage')
    plt.title('Genre Combination Success Rates Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_genre_win_rates_over_time(data, threshold=0.15, negative_only=False):
    """
    Plot genre win rates over time with enhanced visualization.

    Args:
        data: DataFrame containing genre win rates by year
        threshold: Change threshold to highlight steep trends (default 0.15 or 15%)
    """

    # Ensure data is sorted by year
    data = data.sort_values('year')
    
    # Calculate year-over-year changes
    changes = data.groupby('sorted_combo')['win_rate'].diff()
    
    # Identify genres with significant changes
    if not negative_only:
        significant_genres = data[abs(changes) >= threshold]['sorted_combo'].unique()
    else:
        significant_genres = data[changes < -1 * threshold]['sorted_combo'].unique()

    # Set up the plot with specific dimensions
    plt.figure(figsize=(15, 8))
    
    # Use a visually distinct color palette
    palette = sns.color_palette("deep", n_colors=len(significant_genres))
    color_map = dict(zip(significant_genres, palette))
    
    # Get unique years for x-axis
    years = sorted(data['year'].unique())
    
    # Plot all genres in light gray first
    for genre in data['sorted_combo'].unique():
        genre_data = data[data['sorted_combo'] == genre]
        plt.plot(genre_data['year'], genre_data['win_rate'], 
                color='lightgray', linewidth=1.5, alpha=0.6,
                linestyle='--' if genre not in significant_genres else '-',
                marker='o')
    
    # Plot significant genres with colors and annotations
    for genre in significant_genres:
        genre_data = data[data['sorted_combo'] == genre].copy()
        genre_data = genre_data.sort_values('year')
        
        line = plt.plot(genre_data['year'], genre_data['win_rate'],
                       label=genre, linewidth=2.5,
                       color=color_map[genre],
                       marker='o',
                       markersize=6)
        
        # Add annotations for significant changes
        for idx in range(len(genre_data) - 1):
            current_rate = genre_data['win_rate'].iloc[idx]
            next_rate = genre_data['win_rate'].iloc[idx + 1]
            change = next_rate - current_rate
            
            if abs(change) >= threshold:
                plt.annotate(f'{change:+.0%}',
                           xy=(genre_data['year'].iloc[idx + 1], next_rate),
                           xytext=(10, 10), 
                           textcoords='offset points',
                           fontsize=8,
                           color=color_map[genre],
                           bbox=dict(facecolor='white', 
                                   edgecolor=color_map[genre],
                                   alpha=0.7,
                                   boxstyle='round,pad=0.5'))
    
    # Customize the plot
    # plt.grid(True, alpha=0.3, linestyle='--')
    plt.title('Genre Win Rates Over Time', 
              pad=20, fontsize=14, loc='left')
    plt.ylabel('Win Rate', fontsize=12)
    plt.xticks(years, rotation=45)

    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.ylim(-0.05, 1.05)  # Set y-axis limits with some padding
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1, 1), 
              loc='upper left',
              title='Genres with Notable Changes',
              fontsize=10,
              title_fontsize=11,
              frameon=False
              )
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gca()

def plot_spider_plots(genre_combo:str, cols_to_show:list, top_genre_medians, top_genre_q1, top_genre_q3, figsize=(10, 5)):
    # Testing confidence intervals rather than comparative
    feature_cols = cols_to_show + ['is_winner', 'sorted_combo']
    feature_median_data = top_genre_medians.filter(pl.col('sorted_combo') == genre_combo)[feature_cols]
    feature_q1_data = top_genre_q1.filter(pl.col('sorted_combo') == genre_combo)[feature_cols]
    feature_q3_data = top_genre_q3.filter(pl.col('sorted_combo') == genre_combo)[feature_cols]

    N = len(cols_to_show)
    theta = radar_factory(N, frame='polygon').tolist()
    theta += theta[:1]

    spoke_dict = {
        'global_tempo': 'Tempo',
        'duration_seconds': 'Duration',
        'danceable_probability': 'Danceability',
        'happy_probability': 'Happiness',
        'instrumental_probability': 'Instrumental',
        'bright_probability': 'Brightness',
        'sad_probability': 'Sadness',
        'party_probability': 'Partyness',
        'relaxed_probability': 'Relaxedness',
        'high_engagement_probability': 'Engagement',
        'high_approachability_probability': 'Approachability',
        'valence_normalized': 'Valence',
        'arousal_normalized': 'Arousal'
        }
    
    spoke_labels = [spoke_dict[col] for col in cols_to_show]
    title = f'Characteristics for {feature_median_data[0]["sorted_combo"].item()} songs'

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    fig.suptitle(title, weight='bold', size='medium')

    for i in range(2):
        winner = feature_median_data.row(i)[-2]
        line = ax[i].plot(theta, feature_median_data.row(i)[:-2] + (feature_median_data.row(i)[0],), color="#7570B3" if not winner else "#1B9E77")
        ax[i].fill_between(theta, feature_q1_data.row(i)[:-2] + (feature_q1_data.row(i)[0],), feature_q3_data.row(i)[:-2] + (feature_q3_data.row(i)[0],), alpha=0.25, color="#7570B3" if not winner else "#1B9E77")
        ax[i].set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax[i].set_varlabels(spoke_labels)

    fig.tight_layout()
    plt.show()

def plot_duration_tempo(genre_combo, top_genre_songs):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    filtered_genre = top_genre_songs.filter(pl.col('sorted_combo') == genre_combo)

    winner_palette = {
        0: "#7570B3",
        1: "#1B9E77"
    }

    duration_plot = sns.boxplot(data=filtered_genre, x='is_winner', y='duration_seconds', ax=ax1, palette=winner_palette, hue='is_winner', legend=False)
    tempo_plot = sns.boxplot(data=filtered_genre, x='is_winner', y='global_tempo', ax=ax2, palette=winner_palette, hue='is_winner', legend=False)

    ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax1.set_xticks([])
    ax1.set_xticklabels('')
    ax1.set_title('Duration (Secs)', fontsize=14, fontweight='semibold')

    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_xticks([])
    ax2.set_xticklabels('')
    ax2.set_title('Tempo (BPM)', fontsize=14, fontweight='semibold')

    plt.tight_layout()
    plt.show()