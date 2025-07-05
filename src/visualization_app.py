# File: visualization_app.py
# Description: Streamlit app to view extracted game review data as graphs
#
# Copyright (c) 2025 Michael Powers
#
# Usage:
#   streamlit run visualization_app.py
# 
#
import streamlit as st
import io
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import os
from core import load_data
from graph_builder import plot_overall_sentiment, plot_recommendation, plot_negative_tracker, plot_top_keywords 
from graph_builder import plot_overall_sentiment_trend, plot_recommendation_trend, plot_negative_tracker_trend, plot_top_keywords_trend


#########################################################################
#▗▖ ▗▖▗▖  ▗▖▗▖ ▗▖ ▗▄▄▖▗▄▄▄▖▗▄▄▄ 
#▐▌ ▐▌▐▛▚▖▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌  █
#▐▌ ▐▌▐▌ ▝▜▌▐▌ ▐▌ ▝▀▚▖▐▛▀▀▘▐▌  █
#▝▚▄▞▘▐▌  ▐▌▝▚▄▞▘▗▄▄▞▘▐▙▄▄▖▐▙▄▄▀
#########################################################################

# 
#
# live ops: 5 days, +&-"update", +"some new feature or map", -"lag, freezing" - 
# sample data should be able to take the date (instead of default to datetime.now)???
#
def add_sample_data(num_days=365, min_reviews_per_day=3, max_reviews_per_day=20, file_path='../data/review_data/MyExampleGame.parquet'):
    from core import _add_data_point
    import random

    overall_sentiments = ['positive', 'negative', 'mixed']
    sentiment_weights = [5, 7, 3]
    positive_keywords_list = [
        "new map", "new map", "new map", "engaging", "addictive", "great graphics", "smooth gameplay", "refreshing", "awesome", "free", "good", "great", "nice","relaxing","fresh","cool",
        "challenging", "innovative", "rewarding", "friendly community", "good story", "love", "recommend"
    ]
    negative_keywords_list = [
        "update", "freezing", "freezing", "freezing", "crashes", "buggy", "pay to win", "boring", "pop ups", "rigged", "difficulty", "impossible", "uninstall",
        "repetitive", "bad controls", "poor support", "monetization", "unfair", "ads", "hate", "annoying", "frustrating", "dumb"
    ]
    #keyword_weights = [36, 18, 80, 89, 46, 59, 5, 16, 48, 32, 85, 8, 66, 23, 24, 57, 12, 55, 78, 28]
    keyword_weights = [81, 60, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bool_weights = [3, 2]
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=num_days - 1)
    print(f"Generating sample data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    current_date = start_date
    day = 1
    while current_date <= end_date:
        # Random number of data points for the current day
        num_data_points_today = random.randint(min_reviews_per_day, max_reviews_per_day)
        
        for _ in range(num_data_points_today):
            # Generate a random timestamp within the current day
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            timestamp = datetime(
                current_date.year, current_date.month, current_date.day,
                hour, minute, second
            )
            overall_sentiment = random.choices(overall_sentiments, weights=sentiment_weights)[0]
            recommendation = random.choices([True, False], weights=bool_weights)[0]
            warning_anti_recommendation = random.choices([True, False], weights=bool_weights)[0]

            num_pos_keywords = random.randint(2, 3)
            positive_keywords = random.choices(positive_keywords_list, weights=keyword_weights, k=num_pos_keywords)
            #random.sample(positive_keywords_list, min(num_pos_keywords, len(positive_keywords_list)))
            num_neg_keywords = random.randint(2, 3)
            negative_keywords = random.choices(negative_keywords_list, weights=keyword_weights, k=num_neg_keywords)
            #random.sample(negative_keywords_list, min(num_neg_keywords, len(negative_keywords_list)))
            #ad_game_mismatch = random.choices([True, False], weights=[random.randint(1,10), random.randint(9,10)])[0]
            ad_game_mismatch = False
            game_cheating_manipulating = random.choices([True, False], weights=[random.randint(1,10), random.randint(6,10)] )[0]
            bugs_crashes_performance = random.choices([True, False], weights=[random.randint(10,10), random.randint(1,10)])[0]
            monetization = random.choices([True, False], weights=[random.randint(3,10), random.randint(4,10)])[0]
            live_ops_events = random.choices([True, False], weights=[random.randint(1,10), random.randint(8,10)])[0]

            _add_data_point(file_path, 1, overall_sentiment, recommendation, warning_anti_recommendation, positive_keywords, negative_keywords, ad_game_mismatch, game_cheating_manipulating, bugs_crashes_performance, monetization, live_ops_events, timestamp)
        print(f'day {day} done')
        current_date += timedelta(days=1)
        day += 1

    print("--------Done generating data.")

#add_sample_data()
#add_sample_data(num_days=8, min_reviews_per_day=50, max_reviews_per_day=100)


#########################################################################
#▗▄▄▄  ▗▄▖▗▄▄▄▖▗▄▖      ▗▄▄▖▗▄▄▄▖▗▖   ▗▄▄▄▖ ▗▄▄▖▗▄▄▄▖
#▐▌  █▐▌ ▐▌ █ ▐▌ ▐▌    ▐▌   ▐▌   ▐▌   ▐▌   ▐▌     █  
#▐▌  █▐▛▀▜▌ █ ▐▛▀▜▌     ▝▀▚▖▐▛▀▀▘▐▌   ▐▛▀▀▘▐▌     █  
#▐▙▄▄▀▐▌ ▐▌ █ ▐▌ ▐▌    ▗▄▄▞▘▐▙▄▄▖▐▙▄▄▖▐▙▄▄▖▝▚▄▄▖  █  
#########################################################################

@st.cache_data
def get_data(file_path):
    df = load_data(file_path)
    if df.empty:
        st.error(f"No data found at {file_path}. Please ensure the file exists and contains data.")
    return df    

def set_data(filepath):
    st.session_state.project_name = filepath.removesuffix('.parquet')
    filepath = f'../data/review_data/{filepath}'
    print(f'Data filepath set: {filepath}')
    st.session_state.datafile = filepath
    return filepath

def get_available_datafiles(data_dir='../data/review_data'):
    if not os.path.exists(data_dir):
        st.warning(f"Directory {data_dir} does not exist.")
        mylogs.log(f"data_select_interface: {data_dir} directory does not exist.")
        return None

    data_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    if not data_list:
        st.warning(f"No files found in the '{data_list}' directory.")
        mylogs.log(f"data_select_interface: no subdirs found in {data_list} directory.")
        return []

    return data_list

def data_select_interface():
    with st.container(border=True):
        st.subheader("Select a datafile to load")
        
        data_list = get_available_datafiles()

        if data_list is None:
            return None
        if not data_list:
            return None

        data_list.insert(0, "Please select a data file")
        selected_file = st.selectbox("Choose a data file:", data_list, on_change=lambda: set_data(st.session_state.data_select), key="data_select")
        
        if selected_file != "Please select a data file":
            set_data(selected_file)
            return selected_file


        if st.button("Cancel"):
            return None

        return None




#########################################################################
#▗▖ ▗▖▗▖  ▗▖▗▖ ▗▖ ▗▄▄▖▗▄▄▄▖▗▄▄▄ 
#▐▌ ▐▌▐▛▚▖▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌  █
#▐▌ ▐▌▐▌ ▝▜▌▐▌ ▐▌ ▝▀▚▖▐▛▀▀▘▐▌  █
#▝▚▄▞▘▐▌  ▐▌▝▚▄▞▘▗▄▄▞▘▐▙▄▄▖▐▙▄▄▀
#########################################################################
                                    
# --- BUILD PDF REPORT ---

def build_report(data_file_path, target_file_path, title):

    df = get_data(data_file_path)

    #load the graphs
    sentiment_fig = plot_overall_sentiment(df)
    reco_fig = plot_recommendation(df)
    neg_tracker_fig = plot_negative_tracker(df)
    pos_kw_fig = plot_top_keywords(df, keyword_type='positive', top_n=5)
    neg_kw_fig = plot_top_keywords(df, keyword_type='negative', top_n=5)

    #build the pdf
    with PdfPages(target_file_path) as pdf:
        fig_title = plt.figure(figsize=(8.5,11))
        fig_title.text(0.5,0.95, title, fontsize=22, fontweight='bold', ha='center', va='top')
        pdf.savefig(fig_title)
        pdf.savefig(sentiment_fig, bbox_inches='tight')
        pdf.savefig(reco_fig, bbox_inches='tight')
        pdf.savefig(neg_tracker_fig, bbox_inches='tight')
        pdf.savefig(pos_kw_fig, bbox_inches='tight')
        pdf.savefig(neg_kw_fig, bbox_inches='tight')


#########################################################################
# ▗▄▄▖▗▄▄▄▖▗▄▖ ▗▄▄▖▗▄▄▄▖
#▐▌     █ ▐▌ ▐▌▐▌ ▐▌ █  
# ▝▀▚▖  █ ▐▛▀▜▌▐▛▀▚▖ █  
#▗▄▄▞▘  █ ▐▌ ▐▌▐▌ ▐▌ █  
#########################################################################      

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Game Review Dashboard")

st.title("🎮 Game Review Analytics Dashboard")

if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.today().date() - timedelta(days=90 - 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.today().date()

if 'graph_mode' not in st.session_state:
    st.session_state.graph_mode = 'Overall'

if 'date_type' not in st.session_state:
    st.session_state.date_type = 'D'

if 'num_keywords' not in st.session_state:
    st.session_state.num_keywords = 5

#################################
# CSS -- Larger font for data preview
#################################
custom_css = """
<style>
/* Target the paragraph element inside the expander header button */
div[data-testid="stExpander"] p {
    font-size: 20px; /* Adjust this value as needed (e.g., 24px, 1.5em) */
    font-weight: bold; /* Optional: make it bold */
    color: #4CAF50; /* Optional: change color */
}


</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


#########################################################################
#▗▖ ▗▖▗▄▄▄▖ ▗▄▖ ▗▄▄▄ ▗▄▄▄▖▗▄▄▖ 
#▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌  █▐▌   ▐▌ ▐▌
#▐▛▀▜▌▐▛▀▀▘▐▛▀▜▌▐▌  █▐▛▀▀▘▐▛▀▚▖
#▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▐▙▄▄▀▐▙▄▄▖▐▌ ▐▌
#########################################################################
                            
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(r"Load Data"):
            data_select_interface()
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.session_state.graph_mode = st.radio("Select Graph Mode:", ['Overall', 'Trend'])
        with col_b:
            if st.session_state.graph_mode == 'Trend':
                st.session_state.date_type = st.radio("Select Trend Granularity", ['D', 'W', 'ME'], captions=["Daily", "Weekly", "Monthly"], index=['D', 'W', 'ME'].index(st.session_state.date_type))
    with col3:
        st.session_state.num_keywords = st.slider("Number of Keywords", min_value=1, max_value=20, value=5)

    with col4:
        st.session_state.start_date = st.date_input('Start Date', st.session_state.start_date)
        st.session_state.end_date = st.date_input('End Date', st.session_state.end_date)


#################################
# DATA PREVIEW
#################################

df = None

if 'datafile' in st.session_state:
    print(f'get data in : {st.session_state.datafile}')
    df = get_data(st.session_state.datafile)

if not df is None:
    st.success(f"Successfully loaded {len(df)} reviews.")
    #st.subheader("Data Preview:")
    with st.expander("Data Preview (click arrow to expand):", expanded=False):
        st.dataframe(df.head())

    st.markdown("---")

#########################################################################
# ▗▄▄▖▗▄▄▖  ▗▄▖ ▗▄▄▖ ▗▖ ▗▖ ▗▄▄▖
#▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   
#▐▌▝▜▌▐▛▀▚▖▐▛▀▜▌▐▛▀▘ ▐▛▀▜▌ ▝▀▚▖
#▝▚▄▞▘▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌ ▐▌▗▄▄▞▘      
#########################################################################
  

    # --- Plot Generation and Display ---
    project_name = st.session_state.project_name if 'project_name' in st.session_state.keys() else ""
    display_graph_mode = "Overall Visualizations" if st.session_state.graph_mode == "Overall" else "Trend Visualizations"
    display_graph_mode += f' for {project_name}'
    display_graph_mode += (f' ({st.session_state.start_date.strftime("%Y-%m-%d")} to {st.session_state.end_date.strftime("%Y-%m-%d")})' if st.session_state.start_date and st.session_state.end_date else '')
    st.header(display_graph_mode)
   
    # --- Select correct graphs based on graph mode ---
    if st.session_state.graph_mode == 'Overall':
        sentiment_fig = plot_overall_sentiment(df, st.session_state.start_date, st.session_state.end_date)  
        reco_fig = plot_recommendation(df, st.session_state.start_date, st.session_state.end_date)
        neg_tracker_fig = plot_negative_tracker(df, st.session_state.start_date, st.session_state.end_date)
        pos_kw_fig = plot_top_keywords(df, keyword_type='positive', top_n=st.session_state.num_keywords, start_date=st.session_state.start_date, end_date=st.session_state.end_date)
        neg_kw_fig = plot_top_keywords(df, keyword_type='negative', top_n=st.session_state.num_keywords, start_date=st.session_state.start_date, end_date=st.session_state.end_date)
    else:
        sentiment_fig = plot_overall_sentiment_trend(df, st.session_state.date_type, st.session_state.start_date, st.session_state.end_date)  
        reco_fig = plot_recommendation_trend(df, st.session_state.date_type, st.session_state.start_date, st.session_state.end_date)  
        neg_tracker_fig = plot_negative_tracker_trend(df, st.session_state.date_type, st.session_state.start_date, st.session_state.end_date)  
        pos_kw_fig = plot_top_keywords_trend(df, keyword_type='positive', top_n=st.session_state.num_keywords, freq=st.session_state.date_type, start_date=st.session_state.start_date, end_date=st.session_state.end_date)
        neg_kw_fig = plot_top_keywords_trend(df, keyword_type='negative', top_n=st.session_state.num_keywords, freq=st.session_state.date_type, start_date=st.session_state.start_date, end_date=st.session_state.end_date)

    # --- SENTIMENT SECTION ---
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            # Overall Sentiment
            st.write('<center><h2>Overall Sentiment</h2></center>', unsafe_allow_html=True)
            if sentiment_fig and not isinstance(sentiment_fig, str):
                st.pyplot(sentiment_fig)
        with col2:
            # Recommendation
            st.write('<center><h2>Recommendation Status</h2></center>', unsafe_allow_html=True)
            if reco_fig and not isinstance(reco_fig, str):
                st.pyplot(reco_fig)
       
    # --- NEGATIVE TRACKER SECTION ---
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
       
        with col2:
            # Negative Tracker
            st.write('<center><h2>Negative Review Categories</h2></center>', unsafe_allow_html=True)
            if neg_tracker_fig and not isinstance(neg_tracker_fig, str):
                st.pyplot(neg_tracker_fig)
        

    # --- KEYWORD SECTION ---
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            # Overall Sentiment
            st.write('<center><h2>Top Positive Keywords</h2></center>', unsafe_allow_html=True)
            if pos_kw_fig and not isinstance(pos_kw_fig, str):
                st.pyplot(pos_kw_fig)
        with col2:
            # Recommendation
            st.write('<center><h2>Top Negative Keywords</h2></center>', unsafe_allow_html=True)
            if neg_kw_fig and not isinstance(neg_kw_fig, str):
                st.pyplot(neg_kw_fig)

    st.markdown("---")


else:
    st.info("Please load some data.")
