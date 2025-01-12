# Streamlit App: Data Analysis and Visualization

This Streamlit app allows users to upload CSV or Excel files and generate insightful visualizations, including:
- Bar plots for categorical variables.
- Scatter plot for numerical variables.
- Word clouds for free-text columns with support for n-grams (unigrams, bigrams, and trigrams).

The app dynamically handles free-text columns, links them to related categorical plots, and ensures clear and interactive visualizations.

---

## Features

### 📊 Data Visualization
1. **Categorical Bar Plots**:
   - Visualize the distribution of categorical variables by a target variable.
   - Supports percentage-based y-axis with counts displayed in hover boxes.
   - Dynamically adjusts the legend order and bar stacking for better readability.

2. **Free Text Handling**:
   - Detects free-text columns automatically.
   - Displays associated free-text entries in a table below the relevant categorical plot.

3. **Word Cloud with N-Grams**:
   - Generates word clouds using unigrams, bigrams, and trigrams.
   - Customizable stop words for text preprocessing.
   - Supports multilingual text, e.g., French stop words.

### ⚙️ Dynamic Adjustments
- Automatically adjusts plot titles to wrap across multiple lines if they're too long.
- Handles free-text tables dynamically, ensuring they are displayed only once and in the correct order.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Recommended: A virtual environment to manage dependencies.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
