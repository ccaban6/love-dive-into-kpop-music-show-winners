import polars as pl
import pandas as pd
import numpy as np
from visualization_utils import plt, sns

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
    return genre_df.rename({
        'audio_features__genre__top_1_class': 'genre_1st',
        'audio_features__genre__top_2_class': 'genre_2nd',
        'audio_features__genre__top_3_class': 'genre_3rd'
    })

def analyze_genre_combinations(genre_df):
    """Analyze genre combinations and their win rates."""
    genre_combo = genre_df.with_columns(
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
    
    return (genre_combo.group_by('sorted_combo')
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