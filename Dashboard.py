import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.compose import make_column_selector as selector

sns.set(style="whitegrid")

st.set_page_config(page_title="Airline Delay Cause Dashboard", layout="wide")
st.title("Airline Delay Cause Dashboard")

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "EDA",
        "Pre-processing",
        "Modeling",
        "Model Comparison",
        "Predictions",
        "Insights"
    ]
)

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1NRc9MdpYlry15fWcIEX_sRV4mTZFORrd"
    return pd.read_csv(url)

df = load_data()

DISPLAY_NAMES = {
    "year": "Year",
    "month": "Month",
    "carrier": "Carrier Code",
    "carrier_name": "Airline",
    "airport": "Airport Code",
    "airport_name": "Airport Name",
    "arr_flights": "Arrival Flights",
    "arr_del15": "Delayed Flights (15+ min)",
    "arr_cancelled": "Cancelled Flights",
    "arr_diverted": "Diverted Flights",
    "arr_delay": "Arrival Delay",
    "carrier_ct": "Carrier Delay Count",
    "weather_ct": "Weather Delay Count",
    "nas_ct": "NAS Delay Count",
    "security_ct": "Security Delay Count",
    "late_aircraft_ct": "Late Aircraft Delay Count",
    "carrier_delay": "Carrier Delay",
    "weather_delay": "Weather Delay",
    "nas_delay": "NAS Delay",
    "security_delay": "Security Delay",
    "late_aircraft_delay": "Late Aircraft Delay",
}

WEEK4_RESULTS = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Ridge Regression",
            "Gradient Boosting Regressor",
        ],
        "R²": [
            0.9606818516871168,
            0.9498284150095941,
            0.9724110584146967,
            0.9606818607060222,
            0.9713055738219620,
        ],
        "MSE": [
            5992047.916632083,
            7646101.208112732,
            4204528.113398814,
            5992046.542159593,
            4373002.900106750,
        ],
        "MAE": [
            885.3203435985833,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    }
)

TOP_POSITIVE_COEFS = pd.DataFrame(
    {
        "Feature": [
            "airport_ORD",
            "airport_SFB",
            "airport_EWR",
            "airport_JFK",
            "airport_IPT",
            "airport_SFO",
            "carrier_ct",
            "late_aircraft_ct",
            "airport_ISP",
            "airport_PGD",
        ],
        "Coefficient": [
            3803.372259,
            2805.726466,
            2318.430910,
            1393.910582,
            1253.710424,
            1239.396080,
            1227.423599,
            1156.701735,
            1061.553477,
            971.577855,
        ],
    }
)

MODEL_NOTES = {
    "Linear Regression": "A strong baseline and easy to interpret. It helped show which airports and delay causes had the largest positive and negative effects.",
    "Decision Tree Regressor": "Good for capturing nonlinearity and easy to visualize conceptually, but it had lower performance than the ensemble models.",
    "Random Forest Regressor": "The best overall model. It achieved the highest R² and the lowest MSE, so it was chosen as the final model.",
    "Ridge Regression": "Very close to Linear Regression, but more stable when there are many encoded features.",
    "Gradient Boosting Regressor": "Also performed very well and was close to Random Forest, but Random Forest still had the best overall numbers."
}

@st.cache_data
def get_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_", regex=False)
    )
    return df

def pretty_df(df: pd.DataFrame):
    renamed = {col: DISPLAY_NAMES.get(col, col.replace("_", " ").title()) for col in df.columns}
    return df.rename(columns=renamed)

@st.cache_data
def get_sampled_data(df: pd.DataFrame, n: int = 6000):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42)

@st.cache_data
def get_missing_table(df: pd.DataFrame):
    miss = df.isna().sum().reset_index()
    miss.columns = ["column", "missing_values"]
    miss["percent_missing"] = (miss["missing_values"] / len(df) * 100).round(2)
    return miss.sort_values(["missing_values", "column"], ascending=[False, True])

@st.cache_data
def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

@st.cache_data
def get_categorical_columns(df: pd.DataFrame):
    cols = df.select_dtypes(include=["object"]).columns.tolist()
    for fallback in ["year", "month"]:
        if fallback in df.columns and fallback not in cols:
            cols.append(fallback)
    return cols

@st.cache_data
def get_cause_totals(df: pd.DataFrame):
    cause_map = {
        "carrier_delay": "Carrier Delay",
        "weather_delay": "Weather Delay",
        "nas_delay": "NAS Delay",
        "security_delay": "Security Delay",
        "late_aircraft_delay": "Late Aircraft Delay",
    }
    rows = []
    for col, label in cause_map.items():
        if col in df.columns:
            rows.append({"Cause": label, "Total Delay": df[col].sum()})
    return pd.DataFrame(rows).sort_values("Total Delay", ascending=False)

@st.cache_data
def get_top_airports(df: pd.DataFrame, top_n: int = 10):
    if "airport" not in df.columns or "arr_delay" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("airport", dropna=True)["arr_delay"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

@st.cache_data
def get_top_carriers(df: pd.DataFrame, top_n: int = 10):
    name_col = "carrier_name" if "carrier_name" in df.columns else "carrier"
    return (
        df.groupby(name_col, dropna=True)["arr_delay"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

@st.cache_data
def get_yearly_delay(df: pd.DataFrame):
    return (
        df.groupby("year", dropna=True)["arr_delay"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

@st.cache_data
def get_monthly_delay(df: pd.DataFrame):
    return (
        df.groupby("month", dropna=True)["arr_delay"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

def home_page(df: pd.DataFrame):
    st.header("Home")

    total_flights = df["arr_flights"].sum()
    total_delay = df["arr_delay"].sum()
    total_delayed_flights = df["arr_del15"].sum()

    delay_rate = (total_delayed_flights / total_flights) * 100 if total_flights > 0 else 0

    # Top airport by delay
    top_airport = (
        df.groupby("airport")["arr_delay"]
        .sum()
        .sort_values(ascending=False)
        .idxmax()
    )

    # Top delay cause
    cause_cols = [
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay"
    ]

    cause_totals = df[cause_cols].sum()
    top_cause = cause_totals.idxmax().replace("_", " ").title()

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.metric("Years Covered", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c2.metric("Chosen Model", "Random Forest")
    c3.metric("Delay Rate", f"{delay_rate:.1f}%")
    c4.metric("Top Delay Cause", top_cause)

    st.markdown(
        '''
        ### Project Overview

        This project analyzes airline delay data across the United States to better understand the factors that contribute to arrival delays. The goal is to identify patterns in how delays vary by airport, airline, and time, and to determine which types of delays have the greatest impact.

        The dataset was explored through visualizations and statistical analysis to uncover trends and relationships between variables. Based on these insights, predictive models were developed to estimate arrival delays using relevant features. The results were then compared to determine which approach performed best.

        This dashboard presents the findings through interactive visualizations, model comparisons, and a prediction tool that estimates expected delays and their most likely causes based on historical patterns.
        '''
    )

    st.subheader("Dataset Preview")
    st.dataframe(pretty_df(df.head()), use_container_width=True, hide_index=True)

def eda_page(df: pd.DataFrame):
    st.header("EDA")
    st.markdown("Interactive visualizations based on the airline delay dataset.")

    # Year filter
    year_options = ["All Years"] + sorted(df["year"].dropna().astype(int).unique().tolist())
    selected_year = st.selectbox("Filter by Year", year_options)

    if selected_year == "All Years":
        eda_df = df.copy()
    else:
        eda_df = df[df["year"] == selected_year].copy()

    # Units toggle
    show_hours = st.checkbox("Display delay values in hours", value=False)
    unit_label = "Hours" if show_hours else "Minutes"
    conversion = 1 / 60 if show_hours else 1

    # Apply conversion to delay columns
    delay_cols = [
        "arr_delay",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay"
    ]

    eda_df = eda_df.copy()
    for col in delay_cols:
        if col in eda_df.columns:
            eda_df[col] = eda_df[col] * conversion

    numeric_columns = get_numeric_columns(eda_df)
    categorical_columns = get_categorical_columns(eda_df)
    plot_df = get_sampled_data(eda_df, 6000)

    st.markdown(
        f"**Current selection:** {'2003–2022' if selected_year == 'All Years' else selected_year} ({unit_label})"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Scatter Plot",
            "Bar Chart Comparison",
            "Line Trends",
            "Delay Cause Breakdown",
            "Heatmap"
        ]
    )

    # ---------------- TAB 1: SCATTER ----------------
    with tab1:
        st.subheader("Interactive Scatter Plot")

        x_axis = st.selectbox(
            "Select X-axis",
            numeric_columns,
            format_func=lambda c: DISPLAY_NAMES.get(c, c),
            key="scatter_x"
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_columns,
            format_func=lambda c: DISPLAY_NAMES.get(c, c),
            key="scatter_y"
        )

        color_col = st.selectbox(
            "Select color grouping",
            categorical_columns,
            format_func=lambda c: DISPLAY_NAMES.get(c, c),
            key="scatter_color"
        )

        fig_scatter = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color_col,
            title=f"{DISPLAY_NAMES.get(x_axis, x_axis)} vs {DISPLAY_NAMES.get(y_axis, y_axis)}",
            labels={
                x_axis: DISPLAY_NAMES.get(x_axis, x_axis),
                y_axis: DISPLAY_NAMES.get(y_axis, y_axis),
                color_col: DISPLAY_NAMES.get(color_col, color_col),
            },
            template="plotly_dark",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------------- TAB 2: BAR CHART ----------------
    with tab2:
        st.subheader("Bar Chart Comparison")

        compare_type = st.selectbox(
            "Compare by",
            ["Airport", "Carrier"],
            key="bar_compare"
        )

        top_n = st.slider("Top N", 5, 20, 10, key="top_n_bar")

        if compare_type == "Airport":
            bar_df = get_top_airports(eda_df, top_n).copy()

            fig_bar = px.bar(
                bar_df,
                x="airport",
                y="arr_delay",
                title=f"Top {top_n} Airports by Total Arrival Delay ({unit_label})",
                labels={
                    "airport": DISPLAY_NAMES.get("airport", "Airport"),
                    "arr_delay": f"Arrival Delay ({unit_label})"
                },
                template="plotly_dark"
            )
        else:
            bar_df = get_top_carriers(eda_df, top_n).copy()
            name_col = "carrier_name" if "carrier_name" in bar_df.columns else "carrier"

            fig_bar = px.bar(
                bar_df,
                x=name_col,
                y="arr_delay",
                title=f"Top {top_n} Carriers by Total Arrival Delay ({unit_label})",
                labels={
                    name_col: DISPLAY_NAMES.get(name_col, "Carrier"),
                    "arr_delay": f"Arrival Delay ({unit_label})"
                },
                template="plotly_dark"
            )

        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- TAB 3: LINE TRENDS ----------------
    with tab3:
        st.subheader("Line Trend Analysis")

        trend_type = st.selectbox(
            "View trend by",
            ["Month", "Year"],
            key="trend_type"
        )

        if trend_type == "Month":
            trend_df = get_monthly_delay(eda_df).copy()
            x_col = "month"
            title = f"Average Arrival Delay by Month ({unit_label})"
        else:
            trend_df = get_yearly_delay(eda_df).copy()
            x_col = "year"
            title = f"Total Arrival Delay by Year ({unit_label})"

        fig_line = px.line(
            trend_df,
            x=x_col,
            y="arr_delay",
            markers=True,
            title=title,
            labels={
                x_col: DISPLAY_NAMES.get(x_col, x_col),
                "arr_delay": f"Arrival Delay ({unit_label})"
            },
            template="plotly_dark"
        )

        st.plotly_chart(fig_line, use_container_width=True)

    # ---------------- TAB 4: DELAY CAUSE BREAKDOWN ----------------
    with tab4:
        st.subheader("Delay Cause Breakdown")

        cause_totals = get_cause_totals(eda_df).copy()

        fig_pie = px.pie(
            cause_totals,
            names="Cause",
            values="Total Delay",
            title=f"Share of Total Delay by Cause ({unit_label})",
            template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_cause_bar = px.bar(
            cause_totals,
            x="Cause",
            y="Total Delay",
            title=f"Total Delay by Cause ({unit_label})",
            labels={"Total Delay": f"Total Delay ({unit_label})"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_cause_bar, use_container_width=True)

    # ---------------- TAB 5: HEATMAP ----------------
    with tab5:
        st.subheader("Delay Heatmap")

        heatmap_df = eda_df.pivot_table(
            values="arr_delay",
            index="year",
            columns="month",
            aggfunc="mean"
        )

        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            origin="lower",
            title=f"Average Arrival Delay by Year and Month ({unit_label})",
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

def preprocessing_page(df: pd.DataFrame):
    st.header("Pre-processing")

    st.markdown(
        """
        This section explains how the dataset was cleaned and prepared before analysis and modeling.

        The goal of preprocessing was to make the data more consistent, easier to work with, and more reliable for visualization and machine learning.
        """
    )

    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Missing Data", "Cleaned Dataset Summary"]
    )

    with tab1:
        st.subheader("What Was Done to Clean the Data")

        st.markdown(
            """
            ### 1. Standardized column names
            The original column names were cleaned by:
            - removing extra spaces
            - converting all text to lowercase
            - replacing spaces and dashes with underscores

            This made the dataset easier to work with in Python and kept naming consistent across the project.

            ### 2. Checked for missing values
            The dataset was reviewed to identify columns with missing values so they could be handled appropriately before analysis.

            ### 3. Removed rows with missing target values
            Rows with missing `arr_delay` values were removed before modeling, since the target variable must be present in order to train and evaluate a model.

            ### 4. Filled missing numeric values
            Missing values in numeric columns were filled using the median.
            The median was chosen because it is less affected by extreme values and outliers than the mean.

            ### 5. Filled missing categorical values
            Missing values in categorical columns were filled with `"Unknown"`.
            This allowed the rows to remain in the dataset instead of being dropped.

            ### 6. Preserved the original delay units
            The delay values were kept in the same units as they appear in the dataset.
            No additional unit conversion was applied in this dashboard.

            ### 7. Prepared features for modeling
            Before modeling, categorical variables such as airports and airlines were encoded into a numeric form,
            and numeric features were scaled where needed so the models could process them correctly.
            """
        )

    with tab2:
        st.subheader("Missing Data Before Cleaning")

        missing_df = get_missing_table(df).copy()
        missing_df["column"] = missing_df["column"].map(lambda c: DISPLAY_NAMES.get(c, c))

        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        fig_missing = px.bar(
            missing_df,
            x="column",
            y="missing_values",
            title="Missing Values by Column",
            labels={"column": "Column", "missing_values": "Missing Values"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_missing, use_container_width=True)

        st.markdown(
            """
            This chart shows how many missing values appeared in each column before cleaning.
            """
        )

    with tab3:
        st.subheader("Dataset Summary After Cleaning")

        cleaned_df = df.copy()

        if "arr_delay" in cleaned_df.columns:
            cleaned_df = cleaned_df.dropna(subset=["arr_delay"])

        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cat_cols = cleaned_df.select_dtypes(include=["object"]).columns

        cleaned_df[num_cols] = cleaned_df[num_cols].fillna(cleaned_df[num_cols].median())
        cleaned_df[cat_cols] = cleaned_df[cat_cols].fillna("Unknown")

        summary_df = pd.DataFrame(
            {
                "Stage": ["Original Dataset", "After Cleaning"],
                "Rows": [len(df), len(cleaned_df)],
                "Columns": [df.shape[1], cleaned_df.shape[1]],
                "Total Missing Values": [
                    int(df.isna().sum().sum()),
                    int(cleaned_df.isna().sum().sum())
                ],
            }
        )

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.subheader("Preview of Cleaned Data")
        start_row = st.number_input("Start Row", value=7145)
        num_rows = 25

        st.dataframe(
        pretty_df(cleaned_df.iloc[start_row:start_row + num_rows]),
        use_container_width=True,
        hide_index=True
        )

        st.markdown(
            """
            After cleaning, the dataset was more consistent and complete, making it better suited for both exploratory analysis and model development.
            This preprocessing step was important because clean input data leads to more reliable visualizations, more stable modeling, and more trustworthy results.
            """
        )

def modeling_page():
    st.header("Modeling")

    st.markdown(
    """
    The modeling stage focuses on a regression problem, because the target variable, `arr_delay`, is continuous.

    ### What is a target variable?
    In machine learning, the target is the value we are trying to predict. In this project, the target is `arr_delay`, which represents the total arrival delay. All other variables (such as airport, carrier, and delay causes) are used as input features to help estimate this value.

    Having a clearly defined target is important because it determines:
    - what the model is trying to learn
    - which type of model should be used
    - how performance is evaluated

    Without a target, the model would not know what to predict.

    ### What is a regression problem?
    A regression problem is used when the target variable is **continuous**, meaning it can take on a wide range of numeric values (for example, delay measured in hours).

    This is different from classification, where the goal is to predict categories (such as "delayed" vs "not delayed").

    In this project:
    - `arr_delay` is a numeric value
    - delays can vary widely in size
    - we want to estimate how large the delay is

    Because of this, we use regression models.

    ### Why regression is used here
    Airline delays are not just yes/no outcomes. A flight could be delayed by a small amount or by several hours. Regression allows the model to capture this variation and provide more detailed insights into delay behavior.

    The regression models used in this project learn how different factors contribute to the overall arrival delay and help quantify the impact of those factors.
    """
)

    st.subheader("Model Results")
    st.dataframe(WEEK4_RESULTS, use_container_width=True, hide_index=True)

    st.subheader("Model Explanations")
    for model, note in MODEL_NOTES.items():
        st.markdown(f"**{model}:** {note}")

    st.subheader("Linear Regression Coefficient Highlights")
    fig_coef = px.bar(
        TOP_POSITIVE_COEFS,
        x="Feature",
        y="Coefficient",
        title="Top Positive Linear Regression Coefficients",
        template="plotly_dark",
    )
    
    st.plotly_chart(fig_coef, use_container_width=True)
    st.markdown(
    """
    ### Interpreting the Bar Graph

    This graph shows the features that have the strongest impact on increasing arrival delay according to the Linear Regression model.

    Each bar represents a feature, and the height of the bar shows how much that feature pushes the predicted delay upward. The taller the bar, the stronger its impact.

    The coefficient values come directly from the model. A coefficient tells us:
    How much the predicted delay changes when that feature increases by 1, while all other features stay the same.

    For example:
    - Large coefficients for airports like ORD or JFK mean that flights arriving into these airports tend to have significantly higher delays.
    - Features like late aircraft delay and carrier-related delays also have strong positive coefficients, showing that operational issues are major contributors to delay.

    These values were learned during training, where the model adjusted each coefficient to make its predictions as close as possible to the actual delays in the data.

    Overall, this graph highlights the main drivers of delay, showing that delays are largely influenced by specific airports and operational factors rather than random events.
    """
)

def comparison_page():
    st.header("Model Comparison")

    st.dataframe(WEEK4_RESULTS, use_container_width=True, hide_index=True)

    tab1, tab2 = st.tabs(["R² Comparison", "MSE Comparison"])

    with tab1:
        fig_r2 = px.bar(
            WEEK4_RESULTS,
            x="Model",
            y="R²",
            title="Model Comparison by R²",
            template="plotly_dark",
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with tab2:
        fig_mse = px.bar(
            WEEK4_RESULTS,
            x="Model",
            y="MSE",
            title="Model Comparison by MSE",
            template="plotly_dark",
        )
        st.plotly_chart(fig_mse, use_container_width=True, hide_index=True)

    st.subheader("Why I Picked Random Forest")
    st.markdown(
        '''
        I picked **Random Forest Regressor** because it gave the **highest R²** and the **lowest MSE**
        out of the five models tested.

        - Random Forest R²: **0.9724**
        - Random Forest MSE: **4,204,528.11**

        It also makes sense for this dataset because airline delays are influenced by many interacting factors,
        and Random Forest can capture nonlinear relationships better than a basic linear model.
        '''
    )

def predictions_page(df: pd.DataFrame):
    st.header("Predictions")

    st.markdown(
        """
        This section uses historical patterns in the dataset to estimate how delay-prone
        a flight may be based on **airport, airline, and month**.

        It focuses on:
        - how likely a delay is
        - how severe delays tend to be
        - which type of delay is most likely
        """
    )

    carrier_col = "carrier_name" if "carrier_name" in df.columns else "carrier"

    airports = sorted(df["airport"].dropna().astype(str).unique().tolist())
    carriers = sorted(df[carrier_col].dropna().astype(str).unique().tolist())
    months = sorted(df["month"].dropna().astype(int).unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_airport = st.selectbox("Airport", airports)
    with col2:
        selected_month = st.selectbox("Month", months)
    with col3:
        selected_carrier = st.selectbox("Carrier", carriers)

    filtered = df.copy()
    filtered = filtered[filtered["airport"].astype(str) == selected_airport]
    filtered = filtered[filtered["month"] == selected_month]
    filtered = filtered[filtered[carrier_col].astype(str) == selected_carrier]

    if filtered.empty:
        st.warning("No matching historical flights were found for that combination.")
        return

    filtered = filtered.copy()

    needed_cols = ["arr_flights", "arr_del15", "arr_delay"]
    if not all(col in filtered.columns for col in needed_cols):
        st.warning("The dataset is missing columns needed for the prediction summary.")
        return

    filtered = filtered[filtered["arr_flights"] > 0]

    if filtered.empty:
        st.warning("No usable historical rows were found after removing rows with 0 flights.")
        return

    total_flights = filtered["arr_flights"].sum()
    total_delayed_flights = filtered["arr_del15"].sum()
    total_arrival_delay = filtered["arr_delay"].sum()

    delay_rate = (total_delayed_flights / total_flights) * 100 if total_flights > 0 else 0

    # Severity based on delay rate + historical delay totals
    if delay_rate < 15:
        severity = "Low"
    elif delay_rate < 35:
        severity = "Moderate"
    else:
        severity = "High"

    delay_cols = {
        "carrier_delay": "Carrier Delay",
        "weather_delay": "Weather Delay",
        "nas_delay": "NAS Delay",
        "security_delay": "Security Delay",
        "late_aircraft_delay": "Late Aircraft Delay"
    }

    cause_values = []
    for col, label in delay_cols.items():
        if col in filtered.columns:
            cause_values.append({
                "Delay Type": label,
                "Total Delay": filtered[col].sum()
            })

    cause_df = pd.DataFrame(cause_values).sort_values("Total Delay", ascending=False)

    total_cause_delay = cause_df["Total Delay"].sum() if not cause_df.empty else 0
    if not cause_df.empty and total_cause_delay > 0:
        cause_df["Percent of Total"] = ((cause_df["Total Delay"] / total_cause_delay) * 100).round(2)
        top_cause = cause_df.iloc[0]["Delay Type"]
    else:
        cause_df["Percent of Total"] = 0
        top_cause = "N/A"

    matched_rows = len(filtered)

    st.subheader("Prediction Summary")

    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of Delay", f"{delay_rate:.1f}%")
    m2.metric("Delay Severity", severity)
    m3.metric("Most Likely Cause", top_cause)

    st.markdown(
        f"""
        Based on historical flights for **{selected_carrier}** going into **{selected_airport}**
        during **month {selected_month}**:

        - about **{delay_rate:.1f}%** of flights were delayed
        - the overall delay risk for this combination is **{severity}**
        - the most likely source of delay is **{top_cause}**
        """
    )

    tab1, tab2, tab3 = st.tabs(
        ["Delay Cause Breakdown", "Seasonal Context", "Matching Rows"]
    )

    with tab1:
        if not cause_df.empty:
            fig_bar = px.bar(
                cause_df,
                x="Delay Type",
                y="Total Delay",
                title="Delay Causes for Selected Flight Conditions",
                labels={"Delay Type": "Delay Type", "Total Delay": "Total Delay"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                cause_df,
                names="Delay Type",
                values="Percent of Total",
                title="Share of Delay Types",
                template="plotly_dark"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(cause_df, use_container_width=True, hide_index=True)

    with tab2:
        seasonal_df = df.copy()
        seasonal_df = seasonal_df[
            (seasonal_df["airport"].astype(str) == selected_airport) &
            (seasonal_df[carrier_col].astype(str) == selected_carrier)
        ]

        if not seasonal_df.empty:
            monthly_summary = (
                seasonal_df.groupby("month", as_index=False)
                .agg(
                    total_flights=("arr_flights", "sum"),
                    delayed_flights=("arr_del15", "sum"),
                )
            )

            monthly_summary = monthly_summary[monthly_summary["total_flights"] > 0]
            monthly_summary["delay_rate"] = (
                monthly_summary["delayed_flights"] / monthly_summary["total_flights"] * 100
            )

            fig_line = px.line(
                monthly_summary,
                x="month",
                y="delay_rate",
                markers=True,
                title=f"Seasonal Delay Rate for {selected_carrier} at {selected_airport}",
                labels={"month": "Month", "delay_rate": "Delay Rate (%)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        preview_cols = [c for c in [
            "year", "month", "carrier", carrier_col, "airport",
            "arr_flights", "arr_del15", "arr_delay",
            "carrier_delay", "weather_delay", "nas_delay",
            "security_delay", "late_aircraft_delay"
        ] if c in filtered.columns]

        st.markdown(f"**Matching historical rows:** {matched_rows:,}")
        st.dataframe(pretty_df(filtered[preview_cols].head(25)), use_container_width=True, hide_index=True)

def insights_page(df: pd.DataFrame):
    st.header("Insights")

    cause_totals = get_cause_totals(df)
    yearly = get_yearly_delay(df)
    monthly = get_monthly_delay(df)

    st.markdown(
        '''
        ### Final Summary

        This project showed that airline delays are not caused by just one factor.
        The biggest contributors tend to be **late aircraft delays**, **carrier delays**, and **NAS delays**,
        while **security delays** are much smaller.

        The EDA helped reveal which airports, carriers, and time periods were associated with larger total delay values.
        The modeling stage then confirmed that the dataset could explain arrival delay very well, with
        **Random Forest** performing the best overall.
        '''
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_year = px.line(
            yearly,
            x="year",
            y="arr_delay",
            markers=True,
            title="Total Arrival Delay by Year",
            labels={"year": DISPLAY_NAMES.get("year", "year"), "arr_delay": DISPLAY_NAMES.get("arr_delay", "arr_delay")},
            template="plotly_dark",
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with col2:
        fig_month = px.line(
            monthly,
            x="month",
            y="arr_delay",
            markers=True,
            title="Average Arrival Delay by Month",
            labels={"month": DISPLAY_NAMES.get("month", "month"), "arr_delay": DISPLAY_NAMES.get("arr_delay", "arr_delay")},
            template="plotly_dark",
        )
        st.plotly_chart(fig_month, use_container_width=True)

    fig_pie = px.pie(
        cause_totals,
        names="Cause",
        values="Total Delay",
        title="Overall Share of Delay Causes",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        '''
        ### Main Takeaways
        - Delay causes connected to late aircraf, carrier issues, and NAS matter the most
        - Some airports and carriers consistently contribute more total delay than others
        - Regression was the right approach because `arr_delay` is continuous
        - Random Forest gave the best overall performance
        - Historical filtering can help estimate which delay types a traveler is most likely to encounter
        '''
    )

st.sidebar.markdown("---")


try:

    if app_mode == "Home":
        home_page(df)
    elif app_mode == "EDA":
        eda_page(df)
    elif app_mode == "Pre-processing":
        preprocessing_page(df)
    elif app_mode == "Modeling":
        modeling_page()
    elif app_mode == "Model Comparison":
        comparison_page()
    elif app_mode == "Predictions":
        predictions_page(df)
    elif app_mode == "Insights":
        insights_page(df)

except FileNotFoundError:
    st.error(f"Could not find the CSV file: {csv_path}")
    st.info("Put Airline_Delay_Cause.csv in the same folder as this Python file, or change the path in the sidebar.")
except Exception as e:
    st.error("The dashboard ran into an error.")
    st.exception(e)
