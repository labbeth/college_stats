from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def detect_free_text(
    df,
    unique_threshold=0.9,
    min_unique_values=10,
    min_avg_length=20,
    std_length_threshold=10,
    max_choices_for_categorical=12,
):
    """
    Detect free-text columns while keeping long-but-structured survey answers
    (for example smartphone categories or time-range answers) as categorical.
    """
    free_text_cols = []

    for col in df.select_dtypes(include=["object", "category"]).columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)
        avg_length = series.map(len).mean()
        std_length = series.map(len).std()

        looks_like_categorical = unique_count <= max_choices_for_categorical

        if looks_like_categorical:
            continue

        if (
            (unique_ratio > unique_threshold and unique_count > min_unique_values)
            or (avg_length > min_avg_length and unique_count > min_unique_values)
            or (std_length > std_length_threshold and unique_count > min_unique_values)
        ):
            free_text_cols.append(col)

    return free_text_cols


def format_title(title, max_chars_per_line=55):
    words = title.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        projected = current_len + len(word) + (1 if current_line else 0)
        if projected <= max_chars_per_line:
            current_line.append(word)
            current_len = projected
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)


def generate_ngram_wordcloud(text_data, n_range=(1, 3), stop_words=None):
    stop_words = list(stop_words) if stop_words else None
    vectorizer = CountVectorizer(ngram_range=n_range, stop_words=stop_words)
    X = vectorizer.fit_transform(text_data)
    ngram_counts = X.sum(axis=0).A1
    ngram_features = vectorizer.get_feature_names_out()
    ngram_frequencies = dict(zip(ngram_features, ngram_counts))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_frequencies)
    return wordcloud
