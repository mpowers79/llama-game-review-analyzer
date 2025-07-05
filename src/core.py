# File: core.py
# Description: Functions for Saving/Loading data & calling LLM
#  uses llama index and ollama
#
# Copyright (c) 2025 Michael Powers
#
# Usage:
#   N/A
# 
#
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from datetime import datetime
import pandas as pd
import os
import json
import time
from filelock import FileLock, Timeout

DEBUG_OUTPUT = False
MODEL_NAME = "hf.co/MrMike42/GameReview-llama3.1-8b-v4-Q4_K_M-GGUF:latest"
LOCK_FILE = "LLM_RUNNING.LOCK"


#########################################################################
#▗▖ ▗▖▗▄▄▄▖▗▖   ▗▄▄▖ ▗▄▄▄▖▗▄▄▖ 
#▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌ ▐▌
#▐▛▀▜▌▐▛▀▀▘▐▌   ▐▛▀▘ ▐▛▀▀▘▐▛▀▚▖
#▐▌ ▐▌▐▙▄▄▖▐▙▄▄▖▐▌   ▐▙▄▄▖▐▌ ▐▌
#########################################################################

def unwrap_json(response):
    clean_response = response.strip()
    if clean_response.startswith("```json") and clean_response.endswith("```"):
        clean_response = clean_response[len("```json"): -len("```")].strip()
        print(f"json needed cleaning.")
    elif clean_response.startswith("```") and clean_response.endswith("```"):
        clean_response = clean_response[len("```"): -len("```")].strip()
        print(f"json needed cleaning.")
    return clean_response


def normalize_bool(value):
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        lower_val = value.strip().lower()
        if lower_val == 'true':
            return True
        if lower_val == 'false':
            return False
    return None

#########################################################################
#▗▄▄▄  ▗▄▖▗▄▄▄▖▗▄▖ 
#▐▌  █▐▌ ▐▌ █ ▐▌ ▐▌
#▐▌  █▐▛▀▜▌ █ ▐▛▀▜▌
#▐▙▄▄▀▐▌ ▐▌ █ ▐▌ ▐▌
#########################################################################

#columns - "timestamp", "completion_time", "overall_sentiment", "recommendation", "warning_anti_recommendation", "positive_keywords", "negative_keywords", "ad_game_mismatch", "game_cheating_manipulating", "bugs_crashes_performance", "monetization", "live_ops_events"

def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        #return pd.DataFrame(columns=["timestamp", "completion_time", "overall_sentiment", "recommendation", "warning_anti_recommendation", "positive_keywords", "negative_keywords", "ad_game_mismatch", "game_cheating_manipulating", "bugs_crashes_performance", "monetization", "live_ops_events"])
        return pd.DataFrame(columns=[
            "timestamp", "completion_time", "overall_sentiment", "recommendation",
            "warning_anti_recommendation", "positive_keywords", "negative_keywords",
            "ad_game_mismatch", "game_cheating_manipulating", "bugs_crashes_performance",
            "monetization", "live_ops_events"
        ]).astype({
            "timestamp": 'datetime64[ns]',
            "completion_time": float,
            "overall_sentiment": str,
            "recommendation": bool,
            "warning_anti_recommendation": bool,
            "positive_keywords": object, # Use 'object' for columns that will hold lists/sets
            "negative_keywords": object, # Use 'object' for columns that will hold lists/sets
            "ad_game_mismatch": bool,
            "game_cheating_manipulating": bool,
            "bugs_crashes_performance": bool,
            "monetization": bool,
            "live_ops_events": bool
        })

def _save_data(df, file_path):
    #df.to_csv(CSV_FILE, index=False)
    df.to_parquet(file_path, index=False)

def _add_data_point(file_path, completion_time, overall_sentiment, recommendation, warning_anti_recommendation, positive_keywords, negative_keywords, ad_game_mismatch, game_cheating_manipulating, bugs_crashes_performance, monetization, live_ops_events, timestamp = None):
    
    if timestamp == None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str): #convert from string
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    new_row = pd.DataFrame({"timestamp": [timestamp], "completion_time":[completion_time], "overall_sentiment":[overall_sentiment], "recommendation":[recommendation], "warning_anti_recommendation":[warning_anti_recommendation], "positive_keywords":[positive_keywords], "negative_keywords":[negative_keywords], "ad_game_mismatch":[ad_game_mismatch], "game_cheating_manipulating":[game_cheating_manipulating], "bugs_crashes_performance":[bugs_crashes_performance], "monetization":[monetization], "live_ops_events":[live_ops_events]})
    df = load_data(file_path)
    df = pd.concat([df, new_row], ignore_index=True)
    _save_data(df, file_path)

def save_review_data(json_response, completion_time, file_path, date):

    parsed_output = None
    try:
        parsed_output = json.loads(unwrap_json(json_response))
    except json.JSONDecodeError as e:
        print(f"Invalid JSON output: {e}\nDATA NOT SAVED")
        return

    #Get and clean values from json
    
    overall_sentiment = parsed_output.get("sentiment", {}).get("overall")
    overall_sentiment = overall_sentiment.strip().lower() if isinstance(overall_sentiment, str) else None #normalize to lower case
    recommendation = normalize_bool(parsed_output.get("sentiment", {}).get("recommendation"))
    warning_anti_recommendation = normalize_bool(parsed_output.get("sentiment", {}).get("warning_anti_recommendation"))

    raw_positive_keywords = parsed_output.get("specifics", {}).get("positive_keywords", [])
    positive_keywords = sorted(list({str(keyword).strip().lower() for keyword in raw_positive_keywords if isinstance(keyword, str) and keyword.strip()}))

    raw_negative_keywords = parsed_output.get("specifics", {}).get("negative_keywords", [])
    negative_keywords = sorted(list({str(keyword).strip().lower() for keyword in raw_negative_keywords if isinstance(keyword, str) and keyword.strip()}))

    ad_game_mismatch = normalize_bool(parsed_output.get("negative_tracker", {}).get("ad_game_mismatch", False))
    game_cheating_manipulating = normalize_bool(parsed_output.get("negative_tracker", {}).get("game_cheating_manipulating", False))
    bugs_crashes_performance = normalize_bool(parsed_output.get("negative_tracker", {}).get("bugs_crashes_performance", False))
    monetization = normalize_bool(parsed_output.get("negative_tracker", {}).get("monetization", False))
    live_ops_events = normalize_bool(parsed_output.get("negative_tracker", {}).get("live_ops_events", False))

    #TESTING
    if DEBUG_OUTPUT:
        print("-----DEBUG: is our data in correct format?")
        print(f"overall_sentiment: {overall_sentiment} : EXPECT: positive, negative, mixed")
        print(f"recommendation: {recommendation}       : EXPECT: True/False")
        print(f"anti recommendation: {warning_anti_recommendation}       : EXPECT: True/False")
        print(f"positive_keywords: {positive_keywords} : EXPECT: set of lowercase keywords or empty")
        print(f"negative_keywords: {negative_keywords} : EXPECT: set of lowercase keywords or empty")
        print(f"ad_game_mismatch: {ad_game_mismatch}       : EXPECT: True/False")
        print(f"game_cheating_manipulating: {game_cheating_manipulating}       : EXPECT: True/False")
        print(f"bugs_crashes_performance: {bugs_crashes_performance}       : EXPECT: True/False")
        print(f"monetization: {monetization}       : EXPECT: True/False")
        print(f"live_ops_events: {live_ops_events}       : EXPECT: True/False")
    
    _add_data_point(file_path, completion_time, overall_sentiment, recommendation, warning_anti_recommendation, positive_keywords, negative_keywords, ad_game_mismatch, game_cheating_manipulating, bugs_crashes_performance, monetization, live_ops_events, date)


#########################################################################
#▗▖   ▗▖   ▗▖  ▗▖
#▐▌   ▐▌   ▐▛▚▞▜▌
#▐▌   ▐▌   ▐▌  ▐▌
#▐▙▄▄▖▐▙▄▄▖▐▌  ▐▌
#########################################################################


def get_system_prompt():
    prompt = """\
You are an expert game review analyzer. Your task is to extract structured information from game reviews, outputting a precise JSON object with sentiment, specific keywords, and negative flags. Ensure the output is valid JSON, following this schema: {'sentiment': {'overall': 'positive|negative|neutral|mixed', 'recommendation': true|false, 'warning_anti_recommendation': true|false}, 'specifics': {'positive_keywords': ['list', 'of', 'phrases'], 'negative_keywords': ['list', 'of', 'phrases']}, 'negative_tracker': {'ad_game_mismatch': true|false, 'game_cheating_manipulating': true|false, 'bugs_crashes_performance': true|false, 'monetization': true|false, 'live_ops_events': true|false}}
    """
    return prompt


def ask_llama3_json(review, system_prompt, model_name):
    Settings.llm = Ollama(model=model_name, request_timeout=10000.0, temperature = 0.0, json_mode=True)
    wrapped_system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> {system_prompt} <|eot_id|>"
    wrapped_prompt = f"<|start_header_id|>user<|end_header_id|>{review}<|eot_id|>"

    if DEBUG_OUTPUT:
        print(f'---System Prompt: {wrapped_system_prompt}')
        print(f'---prompt: {wrapped_prompt}')
    
    chat_engine = SimpleChatEngine.from_defaults(system_prompt = wrapped_system_prompt)
    response = chat_engine.chat(wrapped_prompt)
    return response.response


def parse_review_and_save(review, file_path, date=None, debug=True):
    lock = FileLock(LOCK_FILE, timeout=0)

    try:
        with lock:
            start_time = time.perf_counter()
            response = ask_llama3_json(review, get_system_prompt(), MODEL_NAME)
            end_time = time.perf_counter()
            duration = end_time - start_time
            save_review_data(response, duration, file_path, date)

            if debug:
                print(f"Review: {review}\n\n--Response: {response}")

    except Timeout:
        print(f"SKIPPING LLM CALL: Already running")
    except Exception as e:
        print(f"error: skipping llm call: {e}")



