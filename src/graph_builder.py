# File: graph_builder.py
# Description: Builds graphs to visualize saved review data
#
# Copyright (c) 2025 Michael Powers
#
# Usage:
#   N/A
# 
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#########################################################################
# ▗▄▖  ▗▄▄▖ ▗▄▄▖▗▄▄▖ ▗▄▄▄▖ ▗▄▄▖ ▗▄▖▗▄▄▄▖▗▄▄▄▖
#▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌ █  ▐▌   
#▐▛▀▜▌▐▌▝▜▌▐▌▝▜▌▐▛▀▚▖▐▛▀▀▘▐▌▝▜▌▐▛▀▜▌ █  ▐▛▀▀▘
#▐▌ ▐▌▝▚▄▞▘▝▚▄▞▘▐▌ ▐▌▐▙▄▄▖▝▚▄▞▘▐▌ ▐▌ █  ▐▙▄▄▖
#########################################################################
                                           

def plot_overall_sentiment(df, start_date=None, end_date=None):
    """Generates a bar chart for overall sentiment distribution."""

    filtered_df = df.copy()
    filtered_df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        except Exception as e:
            print(f"Warning: Could not parse start_date '{start_date}'. Skipping start date filter. Error: {e}")
    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        except Exception as e:
            print(f"Warning: Could not parse end_date '{end_date}'. Skipping start date filter. Error: {e}")

    sentiment_counts = filtered_df['overall_sentiment'].value_counts().reindex(['positive', 'mixed', 'negative']).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6)) # Create a new figure and axes
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, hue=sentiment_counts.index, palette='viridis', ax=ax, legend=False)
    ax.set_title('Overall Sentiment Distribution' + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Reviews')
    plt.close(fig) # Close the figure to prevent it from displaying automatically if not using st.pyplot
    return fig



def plot_recommendation(df, start_date=None, end_date=None):
    """Generates a bar chart for recommendation vs. anti-recommendation."""
    
    filtered_df = df.copy()
    filtered_df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        except Exception as e:
            print(f"Warning: Could not parse start_date '{start_date}'. Skipping start date filter. Error: {e}")
    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        except Exception as e:
            print(f"Warning: Could not parse end_date '{end_date}'. Skipping start date filter. Error: {e}")

    recommendation_data = {
        'Recommended': filtered_df['recommendation'].sum(),
        'Anti-Recommended': filtered_df['warning_anti_recommendation'].sum()
    }
    recommendation_df = pd.Series(recommendation_data).reindex(['Recommended', 'Anti-Recommended']).fillna(0).infer_objects(copy=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=recommendation_df.index, y=recommendation_df.values, hue=recommendation_df.index, palette='plasma', ax=ax, legend=False)
    ax.set_xlabel('Type')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Recommendation vs. Anti-Recommendation' + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))
    plt.tight_layout()
    plt.close(fig)
    return fig



def plot_negative_tracker(df, start_date=None, end_date=None):
    """Generates a bar chart for negative review categories."""
    filtered_df = df.copy()
    filtered_df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        except Exception as e:
            print(f"Warning: Could not parse start_date '{start_date}'. Skipping start date filter. Error: {e}")
    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        except Exception as e:
            print(f"Warning: Could not parse end_date '{end_date}'. Skipping start date filter. Error: {e}")

    negative_tracker_columns = [
        "ad_game_mismatch", "game_cheating_manipulating",
        "bugs_crashes_performance", "monetization", "live_ops_events"
    ]
    negative_tracker_counts = filtered_df[negative_tracker_columns].sum()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=negative_tracker_counts.index, y=negative_tracker_counts.values, hue=negative_tracker_counts.index, palette='rocket', ax=ax, legend=False)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Reviews Flagged')
    ax.tick_params(axis='x', rotation=45) #, ha='right'
    ax.set_title('Negative Review Categories' + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_top_keywords(df, keyword_type='positive', top_n=5, start_date=None, end_date=None):
    """Generates a bar chart for top N positive or negative keywords."""
    filtered_df = df.copy()
    filtered_df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        except Exception as e:
            print(f"Warning: Could not parse start_date '{start_date}'. Skipping start date filter. Error: {e}")
    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        except Exception as e:
            print(f"Warning: Could not parse end_date '{end_date}'. Skipping start date filter. Error: {e}")


    if keyword_type == 'positive':
        all_keywords = filtered_df['positive_keywords'].explode().dropna()
        title = f'Top {top_n} Positive Keywords'
        palette = 'Greens_d'
    elif keyword_type == 'negative':
        all_keywords = filtered_df['negative_keywords'].explode().dropna()
        title = f'Top {top_n} Negative Keywords'
        palette = 'Reds_d'
    else:
        raise ValueError("keyword_type must be 'positive' or 'negative'")

    if not all_keywords.empty:
        top_keywords = all_keywords.value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_keywords.index, y=top_keywords.values, hue=top_keywords.index, palette=palette, ax=ax, legend=False)
        ax.set_xlabel('Keyword')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45) #, ha='right'
        ax.set_title(title + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))
   
        fig.tight_layout()
        plt.close(fig)
        return fig
    else:
        print(f"No {keyword_type} keywords found to plot.")
        return None # Return None if no plot can be generated


######################################################################### 
#▗▄▄▄  ▗▄▖▗▄▄▄▖▗▄▄▄▖
#▐▌  █▐▌ ▐▌ █  ▐▌   
#▐▌  █▐▛▀▜▌ █  ▐▛▀▀▘
#▐▙▄▄▀▐▌ ▐▌ █  ▐▙▄▄▖
#########################################################################
    
def plot_overall_sentiment_trend(df, freq='D', start_date=None, end_date=None):
    
    working_df = df.copy()

    
    working_df['date'] = pd.to_datetime(working_df['timestamp'], errors='coerce').dt.date

    if start_date:
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        working_df = working_df[working_df['date'] >= start_date]
    if end_date:
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
        working_df = working_df[working_df['date'] <= end_date]

    sentiment_trends = working_df.groupby(['date', 'overall_sentiment']).size().unstack(fill_value=0)

    if freq not in ['D', 'W', 'ME']:
        freq = 'D'

    all_dates = pd.date_range(
        start=sentiment_trends.index.min() if not sentiment_trends.empty else (start_date or date.today()),
        end=sentiment_trends.index.max() if not sentiment_trends.empty else (end_date or date.today()),
        freq=freq # Daily frequency
    ).date 

    full_sentiment_trends = pd.DataFrame(index=all_dates, columns=['positive', 'mixed', 'negative']).fillna(0)

    for sentiment_type in ['positive', 'mixed', 'negative']:
        if sentiment_type in sentiment_trends.columns:
            full_sentiment_trends[sentiment_type] = sentiment_trends[sentiment_type].reindex(full_sentiment_trends.index).fillna(0).infer_objects(copy=False)


    full_sentiment_trends.index = pd.to_datetime(full_sentiment_trends.index)

    full_sentiment_trends['total_daily_sentiment'] = full_sentiment_trends['positive'] + full_sentiment_trends['mixed'] + full_sentiment_trends['negative']
    full_sentiment_trends['positive'] = (full_sentiment_trends['positive'] / full_sentiment_trends['total_daily_sentiment'].replace(0, 1)) * 100
    full_sentiment_trends['mixed'] = (full_sentiment_trends['mixed'] / full_sentiment_trends['total_daily_sentiment'].replace(0, 1)) * 100
    full_sentiment_trends['negative'] = (full_sentiment_trends['negative'] / full_sentiment_trends['total_daily_sentiment'].replace(0, 1)) * 100
    full_sentiment_trends = full_sentiment_trends.drop(columns=['total_daily_sentiment'])

    # Melt the DataFrame for Seaborn lineplot
    melted_trends = full_sentiment_trends.reset_index().melt(
        id_vars='index',
        var_name='Sentiment Type',
        value_name='Number of Reviews'
    ).rename(columns={'index': 'Date'})

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7)) # Increased size for better line chart visibility

    # Use Seaborn's lineplot to visualize trends
    sns.lineplot(
        data=melted_trends,
        x='Date',
        y='Number of Reviews',
        hue='Sentiment Type',
        palette={'positive': 'green', 'mixed': 'orange', 'negative': 'red'},
        marker='o', # Add markers for data points
        ax=ax
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Percentage of Reviews', fontsize=12)
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability

    ax.set_title('Sentiment Trend' + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))

    ax.legend(title='Sentiment')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() 

    plt.close(fig) 
    return fig


def plot_recommendation_trend(df, freq='D', start_date=None, end_date=None):
   
    working_df = df.copy()

  
    working_df['date'] = pd.to_datetime(working_df['timestamp'], errors='coerce').dt.date

   
    if start_date:
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        working_df = working_df[working_df['date'] >= start_date]

   
    if end_date:
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
        working_df = working_df[working_df['date'] <= end_date]

  
    working_df['is_recommended'] = working_df['recommendation'].astype(int) # 1 if recommended, 0 otherwise
    working_df['is_anti_recommended'] = working_df['warning_anti_recommendation'].astype(int) # 1 if anti-recommended, 0 otherwise
 

    recommendation_trends = working_df.groupby('date').agg(
        Recommended=('is_recommended', 'sum'),
        Anti_Recommended=('is_anti_recommended', 'sum'),
        Total_Data_Points=('date', 'count') 
    )

    if freq not in ['D', 'W', 'ME']:
        freq = 'D'

  
    all_dates = pd.date_range(
        start=recommendation_trends.index.min() if not recommendation_trends.empty else (start_date or date.today()),
        end=recommendation_trends.index.max() if not recommendation_trends.empty else (end_date or date.today()),
        freq=freq
    ).date

    full_recommendation_trends = pd.DataFrame(index=all_dates, columns=['Recommended', 'Anti_Recommended', 'Total_Data_Points']).fillna(0).infer_objects(copy=False)

    for rec_type in ['Recommended', 'Anti_Recommended', 'Total_Data_Points']:
        if rec_type in recommendation_trends.columns:
            full_recommendation_trends[rec_type] = recommendation_trends[rec_type].reindex(full_recommendation_trends.index).fillna(0).infer_objects(copy=False)


    full_recommendation_trends['Recommended_Percentage'] = (
        full_recommendation_trends['Recommended'] / full_recommendation_trends['Total_Data_Points'].replace(0, 1)
    ) * 100
    full_recommendation_trends['Anti_Recommended_Percentage'] = (
        full_recommendation_trends['Anti_Recommended'] / full_recommendation_trends['Total_Data_Points'].replace(0, 1)
    ) * 100

    # Drop the raw counts and total data points columns for plotting if only percentages are desired
    full_recommendation_trends = full_recommendation_trends.drop(columns=['Recommended', 'Anti_Recommended', 'Total_Data_Points'])
    
    # Rename columns for plotting clarity
    full_recommendation_trends = full_recommendation_trends.rename(columns={
        'Recommended_Percentage': 'Recommended',
        'Anti_Recommended_Percentage': 'Anti-Recommended'
    })


    melted_trends = full_recommendation_trends.reset_index().melt(
        id_vars='index',
        var_name='Recommendation Type',
        value_name='Percentage of Reviews'
    ).rename(columns={'index': 'Date'})

    # Create the plot figure and axes
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=melted_trends,
        x='Date',
        y='Percentage of Reviews',
        hue='Recommendation Type',
        palette={'Recommended': 'green', 'Anti-Recommended': 'red'}, # Custom palette for clarity
        marker='o', # Add markers for data points
        ax=ax
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Percentage of Reviews', fontsize=12)
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability
    ax.set_title('Sentiment Trend' + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))

    ax.legend(title='Recommendation Type')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.close(fig)

    return fig

def plot_negative_tracker_trend(df, freq='D', start_date=None, end_date=None):
   
    working_df = df.copy()
    working_df['date'] = pd.to_datetime(working_df['timestamp'], errors='coerce').dt.date

    if start_date:
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        working_df = working_df[working_df['date'] >= start_date]

    if end_date:
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
        working_df = working_df[working_df['date'] <= end_date]

    negative_tracker_columns = [
        "ad_game_mismatch", "game_cheating_manipulating",
        "bugs_crashes_performance", "monetization", "live_ops_events"
    ]


    for col in negative_tracker_columns:
        if col not in working_df.columns:
            working_df[col] = 0 # Add column if it doesn't exist

    negative_trends = working_df.groupby('date')[negative_tracker_columns].sum()

    if freq not in ['D', 'W', 'ME']:
        freq = 'D'

    min_date = negative_trends.index.min() if not negative_trends.empty else (start_date or date.today())
    max_date = negative_trends.index.max() if not negative_trends.empty else (end_date or date.today())

    if isinstance(min_date, pd.Timestamp):
        min_date = min_date.date()
    if isinstance(max_date, pd.Timestamp):
        max_date = max_date.date()

    all_dates = pd.date_range(
        start=min_date,
        end=max_date,
        freq=freq
    ).date

    full_negative_trends = pd.DataFrame(index=all_dates, columns=negative_tracker_columns).fillna(0).infer_objects(copy=False)

    for col in negative_tracker_columns:
        if col in negative_trends.columns:
            full_negative_trends[col] = negative_trends[col].reindex(full_negative_trends.index).fillna(0).infer_objects(copy=False)

    full_negative_trends.index = pd.to_datetime(full_negative_trends.index)

    full_negative_trends['total_flags_daily'] = full_negative_trends[negative_tracker_columns].sum(axis=1)
    for col in negative_tracker_columns:
        full_negative_trends[col] = full_negative_trends[col] / full_negative_trends['total_flags_daily'].replace(0, 1) * 100
    full_negative_trends = full_negative_trends.drop(columns=['total_flags_daily'])



    melted_trends = full_negative_trends.reset_index().melt(
        id_vars='index',
        var_name='Category',
        value_name='Number of Reviews Flagged' 
    ).rename(columns={'index': 'Date'})

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=melted_trends,
        x='Date',
        y='Number of Reviews Flagged',
        hue='Category',
        palette='husl', 
        marker='o', # Add markers for data points
        ax=ax
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Percent of Reviews Flagged', fontsize=12)
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability

    ax.set_title('Negative Tracker Trend' + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))
    
    ax.legend(title='Category')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_top_keywords_trend(df, keyword_type='positive', top_n=5, freq='D', start_date=None, end_date=None):
    working_df = df.copy()
    working_df['date'] = pd.to_datetime(working_df['timestamp'], errors='coerce').dt.date

    if start_date:
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        working_df = working_df[working_df['date'] >= start_date]
    if end_date:
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
        working_df = working_df[working_df['date'] <= end_date]

    if freq not in ['D', 'W', 'ME']:
        freq = 'D'

    if keyword_type == 'positive':
        keyword_column = 'positive_keywords'
        title = f'Top {top_n} Positive Keywords Trend'
        palette = 'husl'
    elif keyword_type == 'negative':
        keyword_column = 'negative_keywords'
        title = f'Top {top_n} Negative Keywords Trend'
        palette = 'husl'
    else:
        raise ValueError("keyword_type must be 'positive' or 'negative'")

    # Prepare data for trending the top keywords
    # First, find the overall top N keywords within the filtered date range
    all_keywords_flat = working_df[keyword_column].explode().dropna()
    if all_keywords_flat.empty:
        print(f"No {keyword_type} keywords found to plot.")
        return None

    overall_top_n_keywords = all_keywords_flat.value_counts().head(top_n).index.tolist()

    # If no keywords, return None
    if not overall_top_n_keywords:
        print(f"No {keyword_type} keywords found to plot within the top {top_n}.")
        return None


    daily_keyword_counts_dict = {}
    for current_date, group in working_df.groupby('date'):
        exploded_keywords = group[keyword_column].explode().dropna()
        if not exploded_keywords.empty:
            # Filter for only the overall_top_n_keywords
            filtered_keywords = exploded_keywords[exploded_keywords.isin(overall_top_n_keywords)]
            if not filtered_keywords.empty:
                daily_keyword_counts_dict[current_date] = filtered_keywords.value_counts()
    
    # Convert the dictionary of Series to a DataFrame
    keyword_trends = pd.DataFrame(daily_keyword_counts_dict).T.fillna(0) # Transpose to get dates as index
    keyword_trends.index = pd.to_datetime(keyword_trends.index) # Ensure index is datetime

    # -----
    # Calculate the total frequency for each date
    keyword_trends['total_frequency'] = keyword_trends[overall_top_n_keywords].sum(axis=1)
    # Convert counts to percentages
    for col in overall_top_n_keywords:
        # Avoid division by zero for dates with no keywords
        keyword_trends[col] = keyword_trends[col] / keyword_trends['total_frequency'].replace(0, 1) * 100
    # Drop the temporary total_frequency column if not needed later
    keyword_trends = keyword_trends.drop(columns=['total_frequency'])

    #------

    # Ensure all overall_top_n_keywords are columns, filling missing with 0
    for kw in overall_top_n_keywords:
        if kw not in keyword_trends.columns:
            keyword_trends[kw] = 0

    # Reorder columns to match overall_top_n_keywords order for consistency
    keyword_trends = keyword_trends[overall_top_n_keywords]




    # Generate a full range of dates
    min_date = keyword_trends.index.min() if not keyword_trends.empty else (start_date or date.today())
    max_date = keyword_trends.index.max() if not keyword_trends.empty else (end_date or date.today())

    if isinstance(min_date, pd.Timestamp):
        min_date = min_date.date()
    if isinstance(max_date, pd.Timestamp):
        max_date = max_date.date()

    all_dates = pd.date_range(
        start=min_date,
        end=max_date,
        freq=freq
    ).date

    full_keyword_trends = pd.DataFrame(index=all_dates, columns=overall_top_n_keywords).fillna(0).infer_objects(copy=False)

    for col in overall_top_n_keywords:
        if col in keyword_trends.columns:
            full_keyword_trends[col] = keyword_trends[col].reindex(full_keyword_trends.index).fillna(0).infer_objects(copy=False)

    full_keyword_trends.index = pd.to_datetime(full_keyword_trends.index)

    # Melt the DataFrame for Seaborn lineplot
    melted_trends = full_keyword_trends.reset_index().melt(
        id_vars='index',
        var_name='Keyword',
        value_name='Frequency'
    ).rename(columns={'index': 'Date'})

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=melted_trends,
        x='Date',
        y='Frequency',
        hue='Keyword',
        palette=palette,
        marker='o',
        ax=ax
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Percent', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    ax.set_title(title + (f' ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})' if start_date and end_date else ''))

    ax.legend(title='Keyword')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.close(fig)

    return fig







