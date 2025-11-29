# NLP Disaster Tweets Classification

Binary text classification project for the Kaggle competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview)

## Project Overview

Twitter has become an important communication channel during emergencies. This project develops models to automatically identify tweets about real disasters, which has real-world applications for:
- Disaster relief organizations monitoring social media
- News agencies filtering actionable emergency information
- Emergency response systems

## Dataset

The dataset includes:
- **Training set**: 7,613 tweets with labels
- **Test set**: 3,263 tweets for prediction
- **Features**:
  - `id`: Unique identifier
  - `keyword`: Pre-selected disaster-related term (optional)
  - `location`: User-provided location (optional)
  - `text`: Full tweet text
  - `target`: Label (1 = disaster, 0 = not disaster)

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-disaster-tweets
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

## Usage

### Run the Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook disaster_tweets_analysis.ipynb
```

The notebook includes:
1. **Problem Description & Data Overview**
2. **Exploratory Data Analysis (EDA)**
   - Missing value analysis
   - Class distribution visualization
   - Text length analysis
   - Keyword and word frequency analysis
3. **Data Preprocessing & Feature Engineering**
   - Text cleaning pipeline
   - Tokenization and lemmatization
   - TF-IDF vectorization
4. **Model Building & Training**
   - Traditional ML models (Logistic Regression, Naive Bayes, Linear SVM)
   - Transformer-based model (DistilBERT)
5. **Results & Model Comparison**
6. **Discussion & Conclusion**

### Generate Submission

The notebook automatically generates `submission.csv` in the project root, which can be submitted to the Kaggle competition.

### Key Preprocessing Steps

- URL and mention removal
- Hashtag text preservation
- Lemmatization
- TF-IDF vectorization (for traditional ML models)
- Context-preserving tokenization (for DistilBERT)

## Project Structure

```
nlp-disaster-tweets/
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── sample_submission.csv  # Submission format example
├── disaster_tweets_analysis.ipynb  # Main analysis notebook
├── submission.csv             # Generated predictions
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Locked dependencies
└── README.md                 # This file
```

## Dependencies

Core dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Traditional ML models
- `nltk` - NLP preprocessing
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers (DistilBERT)
- `datasets` - Dataset handling
- `jupyter` - Notebook environment

## Key Findings

1. **Class Balance**: Dataset is slightly imbalanced (57% non-disaster, 43% disaster)
2. **Text Patterns**: Disaster tweets tend to be longer and contain news-like language
3. **Keywords**: Some keywords are strong predictors (e.g., "derailment", "wreckage")
4. **Model Performance**: Transformer models outperform traditional ML by ~3% F1 score
5. **Challenges**: Detecting metaphorical usage and sarcasm remains difficult

## AI Citation

- "Fix any grammatical issues with the following assignment." prompt. ChatGPT, GPT 5.1, OpenAI, November 29th, 2025 chat.openai.com/chat
