import io
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import detect_free_text, format_title

st.title("Analyse statistique 2ème semestre")

uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

NIVEAU_ORDER = ["6ème", "5ème", "4ème", "3ème"]

DISPLAY_RENAMES = {
    "Comment votre enfant a-t-il trouvé cette journée ?": (
        "Comment votre enfant a-t-il trouvé cette journée d’intégration ?"
    ),
    "Votre enfant a-t-il des problèmes spécifiques liés à son emploi du temps ?": (
        "Votre enfant a-t-il des problèmes spécifiques liés à son emploi du temps\u00A0?"
    ),
}

EXCLUDED_COLUMNS = {
    "Comment s’est déroulée la journée d’intégration ?",
}

COMMENT_PATTERNS = (
    "commentaire",
    "commentaires",
    "notes d'analyse",
)

INVERT_RED_BLUE_FOR = {
    "Vous a-t-il manqué des éléments permettant une meilleure intégration de votre enfant en 6e ?",
    "Votre enfant a-t-il des problèmes spécifiques liés à son emploi du temps ?",
    "Votre enfant a-t-il des problèmes spécifiques liés à son emploi du temps\u00A0?",
}

POSITIVE_COLOR = "#4575b4"
NEGATIVE_COLOR = "#d73027"
LIGHT_POSITIVE = "#91bfdb"
LIGHT_NEGATIVE = "#fc8d59"
NEUTRAL_COLOR = "#999999"

BASE_COLORS = {
    "Absolument pas": NEGATIVE_COLOR,
    "Plutôt non": LIGHT_NEGATIVE,
    "Non, assez peu": LIGHT_NEGATIVE,
    "Assez mal": LIGHT_NEGATIVE,
    "Non, plutôt pas": LIGHT_NEGATIVE,
    "Non, pas vraiment": LIGHT_NEGATIVE,
    "Oui, un peu": LIGHT_NEGATIVE,
    "Non": NEGATIVE_COLOR,
    "Non pas du tout": NEGATIVE_COLOR,
    "Non, pas du tout": NEGATIVE_COLOR,
    "Très mal": NEGATIVE_COLOR,
    "Oui, souvent": NEGATIVE_COLOR,
    "Négativement": NEGATIVE_COLOR,
    "Peu satisfaisante": NEGATIVE_COLOR,
    "Non satisfaisante": NEGATIVE_COLOR,
    "Plutôt difficile": NEGATIVE_COLOR,
    "Oui, plutôt": LIGHT_POSITIVE,
    "Plutôt oui": LIGHT_POSITIVE,
    "Assez bien": LIGHT_POSITIVE,
    "Oui, partiellement": LIGHT_POSITIVE,
    #"Non, pas vraiment": LIGHT_POSITIVE,
    "Oui": POSITIVE_COLOR,
    "Oui, beaucoup": POSITIVE_COLOR,
    "Positivement": POSITIVE_COLOR,
    "Oui tout à fait": POSITIVE_COLOR,
    "Assez satisfaisante": LIGHT_POSITIVE,
    "Plutôt bien": LIGHT_POSITIVE,
    "Oui, tout à fait": POSITIVE_COLOR,
    "Très satisfaisante": POSITIVE_COLOR,
    "Très bien": POSITIVE_COLOR,
    "Non concerné": NEUTRAL_COLOR,
}


def is_comment_column(col_name: str) -> bool:
    lower = col_name.lower()
    return any(pattern in lower for pattern in COMMENT_PATTERNS)


def normalize_display_name(col_name: str) -> str:
    return DISPLAY_RENAMES.get(col_name, col_name)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    seen = {}
    for col in df.columns:
        name = "" if pd.isna(col) else str(col)
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            name = "Colonne_sans_nom"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    return df


def load_data(file_obj):
    file_obj.seek(0)
    if file_obj.name.endswith(".csv"):
        df = pd.read_csv(file_obj, header=1)
    else:
        df = pd.read_excel(file_obj, header=1)

    df = clean_columns(df)

    # Remove rows that are entirely empty after taking the second row as header.
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def ordered_group_values(series: pd.Series):
    values = [v for v in series.dropna().unique().tolist()]
    if set(values).issubset(set(NIVEAU_ORDER)):
        return [v for v in NIVEAU_ORDER if v in values]
    return sorted(values)


def get_color_map(question_label: str, present_classes: list[str]) -> dict[str, str]:
    colors = BASE_COLORS.copy()
    q = str(question_label).strip().lower()

    invert_needed = (
        q.startswith("vous a-t-il manqué des éléments")
        or "emploi du temps" in q
    )

    if invert_needed:
        swap_pairs = [
            ("Non", "Oui"),
            ("Absolument pas", "Oui, tout à fait"),
            ("Plutôt non", "Oui, plutôt"),
            ("Non, pas du tout", "Oui, tout à fait"),
            ("Non, pas vraiment", "Oui, plutôt"),
        ]
        for left, right in swap_pairs:
            if left in colors and right in colors:
                colors[left], colors[right] = colors[right], colors[left]

    mapped = {cls: colors[cls] for cls in present_classes if cls in colors}
    missing = [cls for cls in present_classes if cls not in colors]

    if missing:
        fallback = px.colors.qualitative.Plotly
        for i, cls in enumerate(missing):
            mapped[cls] = fallback[i % len(fallback)]

    return mapped


def split_report_columns(df: pd.DataFrame, target_variables, numerical_vars, datetime_vars, ip_vars, email_vars):
    excluded_vars = set(target_variables + numerical_vars + datetime_vars + ip_vars + email_vars)
    report_columns = []
    comment_columns = []

    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if col in excluded_vars:
            continue
        if is_comment_column(col):
            comment_columns.append(col)
        else:
            report_columns.append(col)

    return report_columns, comment_columns


def build_comments_workbook(df: pd.DataFrame, target_var: str, columns: list[str]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for col in columns:
            subset_cols = [c for c in [target_var, col] if c in df.columns]
            comments_df = df[subset_cols].dropna(how="all")
            if comments_df.empty:
                continue
            sheet_name = re.sub(r"[\\/*?:\[\]]", "_", normalize_display_name(col))[:31]
            comments_df = comments_df.rename(columns={col: normalize_display_name(col)})
            comments_df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


if uploaded_file:
    df = load_data(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    numerical_vars = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_vars = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
    categorical_vars = df.select_dtypes(include=["object", "category"]).columns.tolist()
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

    categorical_vars = [col for col in categorical_vars if col not in free_text_vars]

    st.write("### Variables identifiées")
    st.write("Variables numériques:", numerical_vars)
    st.write("Variables catégorielles:", [normalize_display_name(c) for c in categorical_vars if c not in EXCLUDED_COLUMNS])
    st.write("Variables texte libre:", [normalize_display_name(c) for c in free_text_vars])

    st.write("### Sélectionnez les variables cibles")
    target_variables = st.multiselect(
        "Selectionnez les variables cibles sur lesquelles effectuer le groupement",
        options=df.select_dtypes(include=["object", "category"]).columns.tolist(),
        default=["Niveau"] if "Niveau" in df.columns else [],
        help="These variables will be used for grouping and will not be available for analysis.",
    )

    report_columns, comment_columns = split_report_columns(
        df, target_variables, numerical_vars, datetime_vars, ip_vars, email_vars
    )

    st.write("### Colonnes incluses dans le rapport principal")
    st.write([normalize_display_name(c) for c in report_columns])

    st.write("### Colonnes exportées dans le fichier commentaires")
    st.write([normalize_display_name(c) for c in comment_columns])

    target_var = None
    if target_variables:
        target_var = st.selectbox(
            "Switch target variable for plots",
            target_variables,
            help="Select the target variable to group the data.",
        )

    if target_var:
        st.write(f"### Filtrer les données sur une valeur de '{target_var}'")
        filter_options = ["-- Toutes les valeurs --"] + ordered_group_values(df[target_var])
        filter_value = st.selectbox(
            f"Choisissez une valeur de {target_var} à analyser (ou laissez vide pour tout analyser)",
            options=filter_options,
        )
        if filter_value != "-- Toutes les valeurs --":
            df = df[df[target_var] == filter_value]

    if st.button("Lancer l'analyse"):
        if target_var and report_columns:
            if comment_columns:
                comments_xlsx = build_comments_workbook(df, target_var, comment_columns)
                st.download_button(
                    "Télécharger le fichier des commentaires (.xlsx)",
                    data=comments_xlsx,
                    file_name="commentaires_separes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.write("### Statistiques globales")
            overall_stats = df[report_columns].describe(include="all").transpose()
            overall_stats.index = [normalize_display_name(idx) for idx in overall_stats.index]
            st.dataframe(overall_stats)

            st.write("### Statistiques groupées")
            grouped_stats = df.groupby(target_var, sort=False)[report_columns].describe()
            grouped_stats.columns = ["_".join(col).strip() for col in grouped_stats.columns.values]
            st.dataframe(grouped_stats)

            st.write("### Visualisations")
            ordered_target_values = ordered_group_values(df[target_var])

            sorted_report_columns = report_columns
            rentree_col = "Comment s’est déroulée la rentrée scolaire ?"
            if rentree_col in report_columns:
                start_idx = report_columns.index(rentree_col)
                sorted_report_columns = report_columns[start_idx:] + report_columns[:start_idx]

            for var in sorted_report_columns:
                if var not in categorical_vars:
                    continue

                try:
                    grouped_data = (
                        df.groupby(target_var, sort=False)[var]
                        .value_counts(normalize=False)
                        .unstack()
                        .fillna(0)
                    )

                    grouped_data = grouped_data.reindex(ordered_target_values)
                    percentage_data = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

                    unique_classes = grouped_data.columns.tolist()
                    class_order = [cls for cls in BASE_COLORS if cls in unique_classes] + [
                        cls for cls in unique_classes if cls not in BASE_COLORS
                    ]
                    dynamic_colors = get_color_map(normalize_display_name(var), class_order)

                    plot_data = percentage_data.stack().reset_index()
                    plot_data.columns = [target_var, var, "Percentage"]
                    plot_data["Count"] = grouped_data.stack().values
                    plot_data[var] = pd.Categorical(plot_data[var], categories=class_order, ordered=True)
                    plot_data[target_var] = pd.Categorical(
                        plot_data[target_var], categories=ordered_target_values, ordered=True
                    )

                    fig = px.bar(
                        plot_data,
                        x=target_var,
                        y="Percentage",
                        color=var,
                        text="Count",
                        hover_data={"Count": True, "Percentage": True, target_var: False},
                        barmode="stack",
                        labels={"Percentage": "Percentage (%)", "Count": "Count", target_var: target_var},
                        color_discrete_map=dynamic_colors,
                        category_orders={var: class_order, target_var: ordered_target_values},
                    )

                    formatted_title = format_title(normalize_display_name(var), max_chars_per_line=55)
                    fig.update_layout(
                        title={"text": formatted_title, "x": 0.0, "xanchor": "left"},
                        xaxis_title=target_var,
                        yaxis_title="Percentage (%)",
                        legend_title="",
                        legend=dict(x=1.02, y=1, orientation="v"),
                        margin=dict(t=140, r=40, l=40, b=40),
                        height=650,
                    )
                    fig.update_traces(texttemplate="%{text}", textposition="inside", cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)

                except ValueError as e:
                    st.warning(f"Could not generate visualization for '{var}': {e}")
        else:
            st.warning("Veuillez sélectionner au moins une variable cible et des colonnes à analyser.")
