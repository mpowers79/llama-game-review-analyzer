# Game Review Sentiment Analysis with Fine-Tuned Llama 3.1
*Leveraging Fine-Tuned Llama 3.1 to transform unstructured game reviews into actionable product insights*

## The problem
Game teams struggle to extract actionable insights from the **immense volume of unstructured game review feedback, particularly within game reviews**, often only grasping fragmented, anecdotal bits rather than the comprehensive whole.

## The solution
**AI-Powered Review Analysis:** this project automates the extraction of sentiment and key themes from game reviews using a fine-tuned LLM, providing structured data for data-driven decisions.

<video src="https://github.com/user-attachments/assets/31316858-e336-438b-9fc7-c24fbd04415f" controls loop muted playsinline width="320">
    Your browser does not support the video tag.
</video>

# Key Features and Functionality
* **LLM Fine-Tuning:** Fine-tuned Llama3.1 model that approches the accuracy of sentiment and keyword extraction provided by commercial models like Gemini
* **Automated Data Processing:** Pipeline to ingest game review data, process it through the LLM, and store structured JSON output in Parquet files.
* **Flexible Data Ingestion:** Designed to integrate with custom web scrapers for automated review collection
* **Interactive Streamlit (web) Applications:**
  * **Google Play Store Scraper:** user-friendly interface to directly pull reviews from the Google Play store
  * **LLM Triggering & Data Processing:** Initiate the LLM processing pipeline directly from the web interface, and view processing status.
  * **Data Visualization:** Visualize sentiment trends, negative topics, top positive and negative keywords

## Stability and automation ready
Designed for easy integration into larger data pipelines / BI systems and scheduled automation

# Finetune
**URL:** https://huggingface.co/MrMike42/GameReview-llama3.1-8b-v9-Q4_K_M-GGUF

**Download with Ollama:** 
```
ollama run hf.co/MrMike42/GameReview-llama3.1-8b-v9-Q4_K_M-GGUF
```
**Model Performance Results (custom F1 scoring metrics):**

| Model Version | Overall Score | Sentiment(F1) | Neg Tracker(F1) | Keywords(F1) |
|:-------------:|:-------------:|:-------------:|:---------------:|:------------:|
| Llama Baseline|         0.810 |         0.877 |            0.95 |        0.580 |
| Fine-Tune | 0.854 | 0.911 | 0.96 | 0.674 |
| Gemini 2 flash*| 0.855 | 0.916 | 0.95 | 0.677 |

*Gemini 2 flash was used as a baseline comparision vs commercial models*

# LLM Output Schema
```
{
  'sentiment': {
    'overall': 'positive|negative|neutral|mixed', 
    'recommendation': true|false, 
    'warning_anti_recommendation': true|false
  }, 
  'specifics': {
    'positive_keywords': ['list', 'of', 'phrases'], 
    'negative_keywords': ['list', 'of', 'phrases']
  }, 
  'negative_tracker': {
    'ad_game_mismatch': true|false, 
    'game_cheating_manipulating': true|false, 
    'bugs_crashes_performance': true|false, 
    'monetization': true|false, 
    'live_ops_events': true|false
  }
}
```


# Tech Stack
- Python
- Jupyter Notebooks
- Streamlit
- Pandas
- PyArrow / Parquet
- Unsloth
- google-play-scraper
- Llama 3.1 8b (fine-tuning)
- Llama-index
- Ollama

# How to Run
Project is set up to run the LLM via Ollama. You might need to configure in the *'./src/core.py'* file, and download the fine-tune in Ollama.

**Run LLM to process reviews:**
```
cd src
python process_reviews
```

**Run the streamlit applications:**  
```
cd src
streamlit run visualization_app.py
```

# Contact me
**Connect with me on Linkedin:** https://www.linkedin.com/in/michaelspowers/

**Email:** michael.sean.powers@gmail.com

*For professional inquiries, including job opportunities, please connect with me on Linkedin or send an email.*


