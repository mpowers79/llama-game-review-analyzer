# File: process_reviews.py
# Description: Runs LLM inference on reviews in INPUT directory, saves data to Parquet file,
#      saves original review in ARCHIVE directory. Subdirectory name determines Parquet file used
#      Input reviews (e.g. from web scraper) should be stored in JSON under "review"
#      Input reviews (e.g. manual testing) can be stored as .txt, multiple reviews in single file, seperated by END_SNIPPET_TOKEN 
#
# Copyright (c) 2025 Michael Powers
#
# Usage:
#   python3 process_reviews.py
# 
#
import os
import shutil
from core import parse_review_and_save 
import json
from datetime import datetime

MAX_REVIEWS_TO_PROCESS = 20

INPUT_BASE_DIR = '../data/incoming_reviews/'
ARCHIVE_BASE_DIR = '../data/archived_reviews/'



def log_process(text):
    with open("process.log", "w") as f:
        f.write(text)
        f.flush() # Ensure it's written to disk immediately

def log_message(text):
    print(text)
    with open("status.log", "a") as f:
        f.write(f"{text}\n")
        f.flush() # Ensure it's written to disk immediately

def log_progress(value):
    with open("progress.log", "w") as f:
        f.write(f"{value}")
        f.flush()

def delete_log():
    os.remove("status.log")
    os.remove("progress.log")
    log_process(f"Stopped. Last run ended at: {datetime.now().strftime('[%a %m/%d %H:%M]')}")


#########################################################################
#▗▄▄▖ ▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖ ▗▄▄▖
#▐▌ ▐▌▐▌   ▐▌  ▐▌  █  ▐▌   ▐▌ ▐▌▐▌   
#▐▛▀▚▖▐▛▀▀▘▐▌  ▐▌  █  ▐▛▀▀▘▐▌ ▐▌ ▝▀▚▖
#▐▌ ▐▌▐▙▄▄▖ ▝▚▞▘ ▗▄█▄▖▐▙▄▄▖▐▙█▟▌▗▄▄▞▘
#########################################################################


## ---- THIS IS JUST FOR MANUAL TESTING ------
def extract_reviews_from_text(input_filepath):
    END_SNIPPET_TOKEN = "---END_SNIPPET---"
    reviews = []
    current_review_lines = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip() # Remove leading/trailing whitespace

                if line_stripped == END_SNIPPET_TOKEN:
                    if current_review_lines: # If we have collected lines for a review
                        reviews.append("\n".join(current_review_lines).strip())
                        current_review_lines = [] # Reset for the next review
                else:
                    current_review_lines.append(line) # Keep the line as part of the current review

            if current_review_lines:
                final_snippet = "\n".join(current_review_lines).strip()
                if final_snippet: # Only add if it's not just empty lines
                    reviews.append(final_snippet)

    except FileNotFoundError:
        print(f"  Input file not found: {input_filepath}")
        return []
    except Exception as e:
        print(f"  Error reading input file {input_filepath}: {e}")
        return []

    directory_path = os.path.dirname(input_filepath)
    proj_name = directory_path.removeprefix(INPUT_BASE_DIR)
    parquet_file = f'../data/review_data/{proj_name}.parquet'
    

    for review in reviews:
        if review is not None and review != "":
            print("PROECESSING REVIEW")
            parse_review_and_save(review, parquet_file)
        else:
            print("SKIPPING EMPTY REVIEW")

    filename = os.path.basename(input_filepath)
    archive_dir = os.path.join(ARCHIVE_BASE_DIR, proj_name)
    output_filepath = os.path.join(archive_dir, filename)
    try:
        os.makedirs(archive_dir, exist_ok=True) 
        shutil.move(input_filepath, output_filepath)
        print(f'Archived file: {input_filepath} to {output_filepath}')
    except Exception as e:
        print(f"Error moving file: {e}")


def extract_review_from_json(input_filepath):
    data = None
    try:
        with open(input_filepath, 'r') as file:
            data = json.load(file)
    except Exception as e:
        log_message(f"Error loading json review: {e}")
        return

    if data is None:
        log_message(f"Error -- no json loaded in : {input_filepath}")

    else:
        directory_path = os.path.dirname(input_filepath)
        proj_name = directory_path.removeprefix(INPUT_BASE_DIR)
        parquet_file = f'../data/review_data/{proj_name}.parquet'
        filename = os.path.basename(input_filepath)
        archive_dir = os.path.join(ARCHIVE_BASE_DIR, proj_name)
        output_filepath = os.path.join(archive_dir, filename)

        review = data.get('review')
        date = data.get('date')
        if review is None:
            log_message(f'Error - no review found in json: {input_filepath}')
        else:
            log_message("----------------\nPROCESSING REVIEW")
            parse_review_and_save(review, parquet_file, date)

    try:
        os.makedirs(archive_dir, exist_ok=True)
        shutil.move(input_filepath, output_filepath)
        #log_message(f'Archived file: {input_filepath} to {output_filepath}')
        log_message(f'Done: review processed and archived.')
    except Exception as e:
        log_message(f"Error moving file: {e}")




#########################################################################
#▗▖    ▗▄▖  ▗▄▖ ▗▄▄▖ 
#▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌
#▐▌   ▐▌ ▐▌▐▌ ▐▌▐▛▀▘ 
#▐▙▄▄▖▝▚▄▞▘▝▚▄▞▘▐▌                     
#########################################################################

def process_reviews(max_reviews=MAX_REVIEWS_TO_PROCESS):
    products = [] 
    try:
       products = [d for d in os.listdir(INPUT_BASE_DIR) if os.path.isdir(os.path.join(INPUT_BASE_DIR, d))]
    except FileNotFoundError:
        log_message(f"Error: INPUT_BASE_DIR '{INPUT_BASE_DIR}' not found. Please ensure the directory exists. EXITING")
        delete_log()
        return 
    reviews_processed_count = 0

    os.makedirs('../data', exist_ok=True)

    log_progress(0.0)
    log_process("Running")
    for prod_subdir in products:
        if reviews_processed_count >= max_reviews:
            log_message(f"Reached maximum reviews to process ({max_reviews}). Stopping.")
            delete_log()
            return #this is for manual testing and only expect one file
        prod_dir_path = os.path.join(INPUT_BASE_DIR, prod_subdir)
        log_message(f"\nProcessing product directory: {prod_dir_path}")

        # Loop through each file in the product subdirectory
        for filename in os.listdir(prod_dir_path):
            if reviews_processed_count >= max_reviews:
                log_message(f"Reached maximum reviews to process ({max_reviews}). Stopping.")
                delete_log()
                return

            filepath = os.path.join(prod_dir_path, filename)
            if not os.path.isfile(filepath):
                continue # Skip directories or other non-file entries
            if filename.lower().endswith('.txt'):
                extract_reviews_from_text(filepath)
                log_message("Extracted text reviews, Stopping")
                delete_log()
                return
            elif filename.lower().endswith('.json'):
                extract_review_from_json(filepath)
                reviews_processed_count += 1
                log_progress(reviews_processed_count/max_reviews)
            else:
                log_message(f" SKIPPING UNKNOWN FILETYPE: {filepath}")
    log_message(f'FINISHED processing reviews: Total procsssed: {reviews_processed_count}')
    delete_log()


if __name__ == "__main__":
    process_reviews()








