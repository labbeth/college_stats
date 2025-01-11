from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords

# nltk.download('stopwords')  # run only once
# stop_words = set(stopwords.words('french'))  # Add multiple languages if needed


# Homogenize classes
def clean_parent_columns(df, columns):
    """
    Cleans specific columns by applying the following rules:
    - Change 'Oui' to 'Oui, plutôt'
    - Change 'Non' to 'Plutôt non'
    - Change 'Oui,tout à fait' to 'Oui, tout à fait'
    """

    for col in columns:
        df[col] = df[col].str.strip()  # Remove leading/trailing spaces
        df[col] = df[col].replace({
            'Oui': 'Oui, tout à fait',
            'Non': 'Absolument pas',
            'Oui,tout à fait': 'Oui, tout à fait'
        }, regex=False)

    return df


# Define the cleaning function
def clean_classe_column(df):
    """
    Cleans the 'Classe de votre enfant' column by extracting the 'Niveau'
    and adding a constant 'College' column.
    """
    # Extract the numerical part with 'e' from 'Classe de votre enfant'
    df['Niveau'] = df['Classe de votre enfant'].str.extract(r'(\d+[eè])')
    # Add a constant column 'College'
    df['College'] = 'college'

    return df


# Helper function to detect free text columns
# def detect_free_text(df, threshold_unique=0.9, min_avg_length=12):
#     free_text_cols = []
#     for col in df.select_dtypes(include=['object', 'category']).columns:
#         unique_ratio = df[col].nunique() / len(df)
#         avg_length = df[col].astype(str).apply(len).mean()
#         if unique_ratio > threshold_unique or avg_length > min_avg_length:
#             free_text_cols.append(col)
#     return free_text_cols

def detect_free_text(
    df,
    unique_threshold=0.9,
    min_unique_values=10,
    min_avg_length=20,
    std_length_threshold=10
):
    """
    Detects free-text columns using three heuristics:
    1. Ratio of unique values and minimum unique count.
    2. Average text length.
    3. Standard deviation of text lengths.
    """
    free_text_cols = []

    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        unique_count = df[col].nunique()
        avg_length = df[col].astype(str).apply(len).mean()
        std_length = df[col].astype(str).apply(len).std()

        # Apply heuristics
        if (
            (unique_ratio > unique_threshold and unique_count > min_unique_values) or
            avg_length > min_avg_length or
            std_length > std_length_threshold
        ):
            free_text_cols.append(col)

    return free_text_cols


def format_title(title, max_words_per_line=12):
    """
    Formats a title by adding line breaks (<br>) to ensure it fits within the plot width.

    Args:
        title (str): The title to format.
        max_words_per_line (int): Maximum number of words per line.

    Returns:
        str: Formatted title with line breaks.
    """
    words = title.split()
    lines = [" ".join(words[i:i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
    return "<br>".join(lines)


def generate_ngram_wordcloud(text_data, n_range=(1, 3), stop_words=None):
    """
    Generate a word cloud based on n-grams (unigrams, bigrams, trigrams).

    Args:
        text_data (list): List of text strings.
        n_range (tuple): Range of n-grams to consider (e.g., (1, 3) for unigrams, bigrams, and trigrams).
        stop_words (set or list): Stop words to exclude from the word cloud.

    Returns:
        WordCloud: Generated word cloud.
    """
    # Ensure stop_words is a list
    stop_words = list(stop_words) if stop_words else None

    # Vectorize the text data into n-grams
    vectorizer = CountVectorizer(ngram_range=n_range, stop_words=stop_words)
    X = vectorizer.fit_transform(text_data)

    # Get the n-gram counts
    ngram_counts = X.sum(axis=0).A1
    ngram_features = vectorizer.get_feature_names_out()
    ngram_frequencies = dict(zip(ngram_features, ngram_counts))

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_frequencies)
    return wordcloud
