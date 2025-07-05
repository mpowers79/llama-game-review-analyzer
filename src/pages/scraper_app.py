# File: scraper_app.py
# Description: Streamlit app to scraper Google Play reviews for future analysis
#   This is a manual process for testing or when you don't want to run an automated scraper
#
# Copyright (c) 2025 Michael Powers
#
# Usage:
#   streamlit run scraper_app.py
# 
#

import streamlit as st
import datetime
import os
import json
from google_play_scraper import Sort, reviews



DATA_FILE = "../data/scraper_entries.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                for name, details in data.items():
                    if "Last_date" in details and isinstance(details["Last_date"], str):
                        details["Last_date"] = datetime.datetime.strptime(details["Last_date"], "%Y-%m-%d %H:%M:%S").date()
                return data
        except json.JSONDecodeError:
            st.error("Error decoding JSON from file. Starting with empty data.")
            return {}
        except Exception as e:
            st.error(f"An unexpected error occurred while loading data: {e}. Starting with empty data.")
            return {}
    return {} # Return empty dictionary if file doesn't exist

def save_data(data):
    serializable_data = {}
    for name, details in data.items():
        serializable_details = details.copy()
        if "Last_date" in serializable_details and isinstance(serializable_details["Last_date"], datetime.date):
            serializable_details["Last_date"] = serializable_details["Last_date"].strftime("%Y-%m-%d %H:%M:%S")
        serializable_data[name] = serializable_details
    
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(serializable_data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving data to file: {e}")

def update_last_date(com_name, new_date):

    if 'name' in st.session_state.game_data:
        current_date = st.session_state.game_data[name]["Last_date"]
        if new_date > current_date:
            st.session_state.game_data[name]["Last_date"] = new_date
            save_data(st.session_state.game_data) 
            return True
        else:
            return False # New date is not later
    return False # name not found

def add_new_game_interface():
    with st.form("add_person_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        com_name = ""
        game_name = ""

        with col1:
            com_name = st.text_input("Com Name", key="com_name_input")
        with col2:
            game_name = st.text_input("Game Name", key="game_name_input")
        date = datetime.date(2000,1,1)

    
        submitted = st.form_submit_button("➕ Add Game")

        if (submitted):
            print("button pressed")
            if com_name and game_name:
                if game_name not in st.session_state.game_data:
                    st.session_state.game_data[game_name] = {
                        "com_name": com_name,
                        "Last_date": date
                    }
                    save_data(st.session_state.game_data) # Save after adding
                    st.success(f"Added {com_name} to the dictionary!")
                    st.rerun() # Rerun to update the displayed table
                    st.session_state.show_add_game_form = False
                else:
                    st.warning(f"'{com_name}' already exists. Use the 'Edit/Delete' section to update.")
                    print(f"{com_name} already exists.")
            else:
                st.error("Please enter all the info.")
                print(f"Please enter all the info to add a new game.")

    if st.button("Cancel"):
        st.session_state.show_add_game_form = False
        return



def run_scraper(com_name="com.scopely.monopolygo", project_name="MonopolyGo", num_reviews=200, last_date = None, dir="../data/incoming_reviews"):

    full_dir_path = os.path.join(dir, project_name)

    try:
        os.makedirs(full_dir_path, exist_ok=True)
    except OSError as e:
        st.error(f"Error creating directory {full_dir_path}: {e}")
        print(f"Error creating directory {full_dir_path}: {e}")
        return # Exit the function if directory creation fails

    progress_text = st.empty() 
    my_bar = st.progress(0, text="Pulling reviews. Please wait.")

    review_count = 0
    
    results, continuation_token = reviews(
        app_id=com_name,
        lang='en', # defaults to 'en'
        country='us', # defaults to 'us'
        sort=Sort.NEWEST, # defaults to Sort.NEWEST
        count=int(num_reviews), # defaults to 100
        filter_score_with=None # defaults to None(means all score)
    )

    for result in results:
        
        review = result.get("content")
        new_date = result.get("at")
        new_date_string = new_date.strftime("%Y-%m-%d %H:%M:%S")
        review_count += 1
        percent_complete = float(review_count / int(num_reviews))
        my_bar.progress(percent_complete, text=f"Getting review {review_count} out of {int(num_reviews)}")

        if 'enforce_date' in st.session_state.keys() and st.session_state.enforce_date and last_date is not None and last_date > new_date:
            pass
        else:
    
            review_entry = {"review": review, "date":new_date_string}
            update_last_date(com_name, new_date_string)
            filename = f"{dir}/{project_name}/review{new_date_string}.json"
            try:
                with open(filename, "w") as f:
                    json.dump(review_entry, f, indent=4)
            except Exception as e:
                st.error(f"Error saving data to file: {e}")
                print(f"Error saving scraped review to file: {e}")

    my_bar.empty() # Clear the progress bar after completion
    progress_text.write("Review pulling complete!")
    print("Review pulling complete!")


        

def get_available_games():
    if 'game_data' not in st.session_state.keys():
        return ["None"]
    game_list =  list(st.session_state.game_data.keys())
    game_list.insert(0, "Please select a game")
    return game_list
    


#########################################################################
# ▗▄▄▖▗▄▄▄▖▗▄▖ ▗▄▄▖▗▄▄▄▖
#▐▌     █ ▐▌ ▐▌▐▌ ▐▌ █  
# ▝▀▚▖  █ ▐▛▀▜▌▐▛▀▚▖ █  
#▗▄▄▞▘  █ ▐▌ ▐▌▐▌ ▐▌ █  
#########################################################################      
if 'game_data' not in st.session_state:
    st.session_state.game_data = load_data()
   
    if not st.session_state.game_data:
        st.session_state.game_data = {
            "MonopolyGo": {"com_name": "com.scopely.monopolygo", "Last_date": datetime.date(1990, 1, 15)}
        }
        save_data(st.session_state.game_data) # Save initial data to file

if 'scrape_num' not in st.session_state.keys():
    st.session_state.scrape_num = 200

if 'enforce_date' not in st.session_state.keys():
    st.session_state.enforce_date = True

if 'show_add_game_form' not in st.session_state:
    st.session_state.show_add_game_form = False

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Game Review Scraper")

st.title("🎮 Game Review Scraper")


if st.button(r"Add New Game to List"):
    st.session_state.show_add_game_form = True

if st.session_state.show_add_game_form:
    add_new_game_interface()

# ------ game scraper interface --------
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.selected_game = st.selectbox("Select a game:", get_available_games())
    with col2:
        st.session_state.scrape_num = st.text_input("Number Reviews", value = "200")
    with col3:
        st.session_state.enforce_date = st.checkbox("Only New Reviews?", value = st.session_state.enforce_date)
    if st.button("Run"):
        if st.session_state.selected_game == "Please select a game":
            st.write("Please select a game first!")
        else:
            com_name = st.session_state.game_data[st.session_state.selected_game].get("com_name")
            project_name = st.session_state.selected_game
            date_val =  st.session_state.game_data[st.session_state.selected_game].get("Last_date")
            run_scraper(com_name, project_name, st.session_state.scrape_num, date_val  )
            #st.write(f'{st.session_state.game_data[st.session_state.selected_game].get("com_name")} {st.session_state.scrape_num} {st.session_state.enforce_date}')










