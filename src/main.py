import streamlit as st
import pandas as pd
import re
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.corpus import stopwords
import nltk
from utils import *

# Download stop words if not already available
# nltk.download('stopwords')  # run only once
# stop_words = set(stopwords.words('french'))  # Add multiple languages if needed

# set page layout
# st.set_page_config(layout="wide")

# App title
st.title("Analyse statistique 2ème semestre")

# File upload
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # Apply the functions to the dataframe
    # df = clean_classe_column(df_raw)

    df = add_concatenated_column(df_raw, "Niveau", "Classe", "Niveau_Classe")

    # Reorder the columns to place 'Niveau' and 'College' after 'Classe de votre enfant'
    # cols = list(df.columns)
    # index = cols.index('Classe de votre enfant')

    # Insert the new columns at the correct position
    # cols.insert(index + 1, cols.pop(cols.index('Niveau')))
    # cols.insert(index + 2, cols.pop(cols.index('College')))
    # df = df_raw[cols]

    # Homogenize classes
    # parent_columns = [col for col in df.columns if col.startswith('En tant que parent,')]
    # df = clean_parent_columns(df, parent_columns)

    df = df_raw

    '''Streamlit App'''

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Detect variable types
    numerical_vars = df.select_dtypes(include=['number']).columns.tolist()
    datetime_vars = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
    free_text_vars = detect_free_text(df)
    ip_vars = []
    email_vars = []

    for col in df.select_dtypes(include=["object"]):
        try:
            sample = df[col].dropna().astype(str).iloc[0]
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", sample):
                ip_vars.append(col)
            elif re.match(r".+@.+\..+", sample):
                email_vars.append(col)
        except Exception:
            continue

    # Exclude free text from categorical variables
    categorical_vars = [col for col in categorical_vars if col not in free_text_vars]

    st.write("### Variables identifiées")
    st.write("Variables numériques:", numerical_vars)
    st.write("Variables catégorielles:", categorical_vars)
    st.write("Variables texte libre:", free_text_vars)

    # Step 1: Allow the user to select multiple target variables
    st.write("### Sélectionnez les variables cibles")
    target_variables = st.multiselect(
        "Selectionnez les variables cibles sur lesquelles effectuer le groupement",
        options=df.select_dtypes(include=['object', 'category']).columns.tolist(),
        default=[],
        help="These variables will be used for grouping and will not be available for analysis."
    )

    # Step 2: Automatically exclude target variables from analysis column selection
    st.write("### Selectionnez les colonnes à analyser")

    # Regroupement des colonnes à exclure
    excluded_vars = set(target_variables + numerical_vars + datetime_vars + ip_vars + email_vars)

    analysis_columns = []
    for col in df.columns:
        default_checked = col not in excluded_vars
        if st.checkbox(f"{col}", value=default_checked):
            analysis_columns.append(col)

    # Step 3: Dropdown to select the active target variable for plots
    if target_variables:
        target_var = st.selectbox(
            "Switch target variable for plots",
            target_variables,
            help="Select the target variable to group the data."
        )

    # Étape optionnelle : filtrer une valeur spécifique du groupe cible
    if target_var:
        st.write(f"### Filtrer les données sur une valeur de '{target_var}'")
        filter_value = st.selectbox(
            f"Choisissez une valeur de {target_var} à analyser (ou laissez vide pour tout analyser)",
            options=["-- Toutes les valeurs --"] + sorted(df[target_var].dropna().unique().tolist())
        )

        if filter_value != "-- Toutes les valeurs --":
            df = df[df[target_var] == filter_value]

    # Step 4: Button to trigger statistics computation
    if st.button("Lancer l'analyse"):
        if target_variables and analysis_columns:
            # Overall Statistics
            st.write("### Statistiques globales")
            overall_stats = df[analysis_columns].describe(include='all').transpose()
            st.dataframe(overall_stats)

            # Grouped Statistics
            st.write(f"### Statistiques groupées")
            grouped_stats = df.groupby(target_var)[analysis_columns].describe()
            grouped_stats.columns = ['_'.join(col).strip() for col in grouped_stats.columns.values]
            st.dataframe(grouped_stats)

            # Visualizations
            # Define a custom color scale
            custom_colors = {
                "Absolument pas": "#d73027",  # Red tone for negative class
                "Plutôt non": "#fc8d59",  # Lighter red
                "Non, assez peu": "#fc8d59",
                "Assez mal": "#fc8d59",
                "Non, plutôt pas": "#fc8d59",
                "Oui, un peu": "#fc8d59",
                "Non": "#d73027",  # Lighter red (similar tone for general "Non")
                "Non pas du tout": "#d73027",
                "Non, pas du tout": "#d73027",
                "Très mal": "#d73027",
                "Oui, souvent": "#d73027",
                "Négativement": "#d73027",
                "Peu satisfaisante": "#d73027",
                "Plutôt difficile": "#d73027",
                "Oui, plutôt": "#91bfdb",  # Light blue
                "Plutôt oui": "#91bfdb",
                "Assez bien": "#91bfdb",
                "Oui, partiellement": "#91bfdb",
                "Non, pas vraiment": "#91bfdb",
                "Oui": "#4575b4",  # Light blue (similar tone for general "Oui")
                "Oui, beaucoup": "#4575b4",
                "Positivement": "#4575b4",
                "Oui tout à fait": "#4575b4",
                "Assez satisfaisante": "#91bfdb",
                "Plutôt bien": "#91bfdb",
                "Oui, tout à fait": "#4575b4",  # Blue tone for positive class
                "Très satisfaisante": "#4575b4",
                "Très bien": "#4575b4"
            }

            default_color = "#999999"  # Gray for unhandled categories

            # Keep track of displayed free text columns
            displayed_free_text_columns = set()

            st.write("### Visualisations")
            # Initialize session state for user notes
            if "user_notes" not in st.session_state:
                st.session_state.user_notes = {}

            for i, var in enumerate(analysis_columns):
                try:
                    if var in categorical_vars:
                        # st.write(f"Distribution of '{var}' by '{target_var}'")

                        # Grouped data
                        grouped_data = df.groupby(target_var)[var].value_counts(normalize=False).unstack().fillna(0)

                        # Convert counts to percentages for plotting
                        percentage_data = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

                        # Get unique categories for this variable
                        unique_classes = grouped_data.columns.tolist()

                        # Dynamically order categories (negative to positive where applicable)
                        class_order = [cls for cls in custom_colors if cls in unique_classes] + \
                                      [cls for cls in unique_classes if cls not in custom_colors]

                        # Dynamically map colors for present classes
                        dynamic_colors = {cls: custom_colors.get(cls, default_color) for cls in class_order}

                        # Prepare data for Plotly
                        plot_data = percentage_data.stack().reset_index()
                        plot_data.columns = [target_var, var, "Percentage"]
                        plot_data["Count"] = grouped_data.stack().values

                        # Ensure the correct order of categories
                        plot_data[var] = pd.Categorical(plot_data[var], categories=class_order, ordered=True)

                        # Plot using Plotly
                        fig = px.bar(
                            plot_data,
                            x=target_var,
                            y="Percentage",
                            color=var,
                            text="Count",
                            hover_data={"Count": True, "Percentage": True, target_var: False},
                            barmode="stack",
                            labels={"Percentage": "Percentage (%)", "Count": "Count"},
                            color_discrete_map=dynamic_colors,
                            category_orders={var: class_order},  # Enforce category order
                        )

                        # Format the plot title with line breaks
                        formatted_title = format_title(f"{var}")

                        # Update layout for better visuals
                        fig.update_layout(
                            title=formatted_title,
                            xaxis_title=target_var,
                            yaxis_title="Percentage (%)",
                            legend_title="",
                            legend=dict(x=1.05, y=1, orientation="v"),
                        )
                        # fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")  # To display values in bars
                        fig.update_traces(texttemplate="%{text}", textposition="inside")

                        # Display the plot in Streamlit
                        st.plotly_chart(fig, use_container_width=True)

                        # Check if the next column is a free text column
                        col_index = df.columns.get_loc(var)
                        if col_index + 1 < len(df.columns):
                            next_col = df.columns[col_index + 1]
                            if next_col in free_text_vars:
                                st.write(f"{next_col}")

                                # Extract and display the free text table
                                free_text_table = df[[target_var, next_col]].dropna().reset_index(drop=True)
                                st.dataframe(free_text_table)

                                # Add a form for notes input
                                with st.form(key=f"form_{next_col}"):
                                    if next_col not in st.session_state.user_notes:
                                        st.session_state.user_notes[next_col] = ""  # Initialize note in session state

                                    user_note = st.text_area(
                                        f"Notes d'analyse :",
                                        value=st.session_state.user_notes[next_col],
                                        placeholder="Entrez vos observations et analyse ici..."
                                    )

                                    # Add a hidden submit button to satisfy Streamlit's form requirement
                                    st.form_submit_button(label="Submit", disabled=True)

                                    # Save the note to session state
                                    st.session_state.user_notes[next_col] = user_note

                except ValueError as e:
                    st.warning(f"Could not generate visualization for '{var}': {e}")

            # # Numerical variable
            # st.write("### Variables Numériques")
            # for num_var in numerical_vars:
            #     st.write(f"#### Scatter Plot: {num_var} vs. {target_var}")
            #     fig = px.scatter(
            #         df,
            #         x=target_var,
            #         y=num_var,
            #         color=target_var,
            #         labels={target_var: target_var, num_var: num_var},
            #         title=f"{num_var} vs. {target_var}",
            #         hover_data=df.columns,
            #     )
            #     fig.update_layout(
            #         xaxis_title=target_var,
            #         yaxis_title=num_var,
            #         legend_title=target_var,
            #     )
            #     st.plotly_chart(fig, use_container_width=True)

            # # Free Text Analysis with Word Cloud
            # if free_text_vars:
            #     st.write("### Word Clouds")
            #     for col in free_text_vars:
            #         st.write(f"#### {col}")

            #         # Combine all text in the column
            #         text_data = ' '.join(df[col].dropna().astype(str).tolist())

            #         # Remove stop words
            #         text_data = ' '.join(word for word in text_data.split() if word.lower() not in stop_words)

            #         # Generate word cloud
            #         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

            #         # Display the word cloud
            #         plt.figure(figsize=(10, 5))
            #         plt.imshow(wordcloud, interpolation='bilinear')
            #         plt.axis('off')
            #         st.pyplot(plt)

            # # Free Text Word Cloud with N-Grams
            # if free_text_vars:
            #     st.write("### Word Cloud")
            #     for col in free_text_vars:
            #         st.write(f"#### {col}")
            #
            #         # Combine all text in the column
            #         text_data = df[col].dropna().astype(str).tolist()
            #
            #         # Generate the word cloud
            #         wordcloud = generate_ngram_wordcloud(text_data, n_range=(1, 3), stop_words=stop_words)
            #
            #         # Display the word cloud
            #         plt.figure(figsize=(10, 5))
            #         plt.imshow(wordcloud, interpolation='bilinear')
            #         plt.axis('off')
            #         st.pyplot(plt)
