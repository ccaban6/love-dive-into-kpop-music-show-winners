import math
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sqlite3

def plot_debut_bin_heatmap_grid(run_sql, show_names, debut_bin_order=None, n_cols=2):
    """
    Plot a grid of heatmaps showing win count by debut bin and year, one for each show.
    
    Parameters:
    - run_sql: function to execute SQL and return a DataFrame
    - show_names: list of show names
    - debut_bin_order: optional list defining the order of debut bins
    - n_cols: number of columns in the grid layout
    """

    if debut_bin_order is None:
        debut_bin_order = ['0', '1-2', '3-4', '5-6', '7-8', '9-10', '11+']
    
    n_shows = len(show_names)
    n_rows = math.ceil(n_shows / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, show in enumerate(show_names):
        query = f"""
        WITH debut_bins AS (
            SELECT 
                STRFTIME('%Y', a.date) AS year,
                a.show,
                CASE 
                    WHEN CAST((JULIANDAY(a.date) - JULIANDAY(b."Debut Date"))/365.25 AS INT) = 0 THEN '0'
                    WHEN CAST((JULIANDAY(a.date) - JULIANDAY(b."Debut Date"))/365.25 AS INT) BETWEEN 1 AND 2 THEN '1-2'
                    WHEN CAST((JULIANDAY(a.date) - JULIANDAY(b."Debut Date"))/365.25 AS INT) BETWEEN 3 AND 4 THEN '3-4'
                    WHEN CAST((JULIANDAY(a.date) - JULIANDAY(b."Debut Date"))/365.25 AS INT) BETWEEN 5 AND 6 THEN '5-6'
                    WHEN CAST((JULIANDAY(a.date) - JULIANDAY(b."Debut Date"))/365.25 AS INT) BETWEEN 7 AND 8 THEN '7-8'
                    WHEN CAST((JULIANDAY(a.date) - JULIANDAY(b."Debut Date"))/365.25 AS INT) BETWEEN 9 AND 10 THEN '9-10'
                    ELSE '11+'
                END AS debut_bin
            FROM all_awards a
            JOIN artist_metadata b ON a.artist = b.artist
            WHERE a.placement = 1 AND a.show = '{show}'
        )
        SELECT year, debut_bin, COUNT(*) AS win_count
        FROM debut_bins
        GROUP BY year, debut_bin
        ORDER BY year, debut_bin;
        """

        df = run_sql(query)
        ax = axes[i]

        if df.empty:
            ax.set_visible(False)
            continue

        df['debut_bin'] = pd.Categorical(df['debut_bin'], categories=debut_bin_order, ordered=True)
        heatmap_data = df.pivot(columns='debut_bin', index='year', values='win_count').fillna(0)

        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='g', 
            cmap='YlGnBu', 
            ax=ax, 
            cbar_kws={'shrink': 0.75} 
        )

        ax.set_title(show)

        # Remove x-axis labels and ticks unless bottom row
        if i // n_cols != n_rows - 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('Years Since Debut')

        # Remove y-axis labels and ticks unless leftmost column
        if i % n_cols != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.set_ylabel('Year')

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_matchup_bar_chart(df):
    # Filter out same-type matchups (diagonal elements)
    different_types = df[df["group_comp_a"] != df["group_comp_b"]].copy()
    
    # Calculate win share for group_comp_a
    different_types["win_share_a"] = different_types["group_comp_a_wins"] / different_types["total_matchups"]
    
    # Create matchup labels
    different_types["matchup"] = different_types["group_comp_a"] + " vs " + different_types["group_comp_b"]
    
    # Filter for top matchups by total count (minimum 10 matchups for reliability)
    reliable_matchups = different_types[different_types["total_matchups"] >= 10]
    
    # Sort by total matchups and take top 8-10 most frequent
    top_matchups = reliable_matchups.nlargest(10, "total_matchups")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    bars = ax.barh(range(len(top_matchups)), top_matchups["win_share_a"], 
                   color=['#d62728' if x > 0.6 else '#2ca02c' if x < 0.4 else '#1f77b4' 
                          for x in top_matchups["win_share_a"]])
    
    # Customize the plot
    ax.set_yticks(range(len(top_matchups)))
    ax.set_yticklabels(top_matchups["matchup"], fontsize=10)
    ax.set_xlabel("Win Rate for First Artist Type", fontsize=12)
    ax.set_title("Artist Type Head-to-Head Win Rates\n(Most Frequent Matchups)", 
                 fontsize=14, fontweight='bold')
    
    # Add percentage labels on bars
    for i, (bar, win_rate, total) in enumerate(zip(bars, top_matchups["win_share_a"], top_matchups["total_matchups"])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{win_rate:.1%} (n={total})', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add reference line at 50%
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(0.5, len(top_matchups), '50%', ha='center', va='bottom', 
            color='gray', fontsize=9)
    
    # Format x-axis as percentages
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '50%', '60%', '80%', '100%'])
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    
    # Invert y-axis so highest frequency is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

def plot_gender_matchup_chart(df):
    # Filter out same-type matchups
    different_types = df[df["group_comp_a"] != df["group_comp_b"]].copy()
    
    # Calculate win share
    different_types["win_share_a"] = different_types["group_comp_a_wins"] / different_types["total_matchups"]
    
    # Create simplified categories for cleaner visualization
    def simplify_category(cat):
        if 'female' in cat and 'group' in cat:
            return 'Female Groups'
        elif 'male' in cat and 'group' in cat:
            return 'Male Groups'
        elif 'female' in cat and 'solo' in cat:
            return 'Female Solos'
        elif 'male' in cat and 'solo' in cat:
            return 'Male Solos'
        elif 'coed' in cat:
            return 'Co-ed Groups'
        return cat
    
    different_types["simple_a"] = different_types["group_comp_a"].apply(simplify_category)
    different_types["simple_b"] = different_types["group_comp_b"].apply(simplify_category)
    different_types["simple_matchup"] = different_types["simple_a"] + " vs " + different_types["simple_b"]
    
    # Filter for meaningful sample sizes and sort
    reliable = different_types[different_types["total_matchups"] >= 5]
    top_simple = reliable.nlargest(8, "total_matchups")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(top_simple)), top_simple["win_share_a"],
                  color=['#e74c3c' if x > 0.6 else '#27ae60' if x < 0.4 else '#3498db' 
                         for x in top_simple["win_share_a"]])
    
    # Customize
    ax.set_xticks(range(len(top_simple)))
    ax.set_xticklabels(top_simple["simple_matchup"], rotation=45, ha='right')
    ax.set_ylabel("Win Rate for First Group", fontsize=12)
    ax.set_title("Artist Type Competitive Dynamics", fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, win_rate, total in zip(bars, top_simple["win_share_a"], top_simple["total_matchups"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{win_rate:.1%}\n(n={total})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Format y-axis
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '50%', '60%', '80%', '100%'])
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_sql(query, db_path="../data/sql/clean.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df