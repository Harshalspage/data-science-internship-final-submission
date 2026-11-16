
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go



# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------


# --------------------------------------------------
# PROFESSIONAL DASHBOARD THEME
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0B1220;
        color: #E5EDF7;
    }

    [data-testid="stSidebar"] {
        background-color: #111D30;
        border-right: 1px solid #24364D;
    }

    [data-testid="stSidebar"] * {
        color: #E5EDF7;
    }

    h1, h2, h3 {
        color: #F1F5F9;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #17283D, #1B3048);
        border: 1px solid #29445E;
        padding: 20px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #A9BCD0;
    }

    [data-testid="stMetricValue"] {
        color: #54C6B5;
        font-weight: 800;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #29445E;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .dashboard-subtitle {
        color: #9BAFC4;
        font-size: 15px;
        margin-top: -10px;
    }

    .section-label {
        color: #54C6B5;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    hr {
        border-color: #263A50;
    }
</style>
""", unsafe_allow_html=True)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "evaluation"
FIGURES = ROOT / "reports" / "figures"


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_json(filename):
    path = DATA / filename

    if not path.exists():
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    return pd.DataFrame(records)


@st.cache_data
def load_csv(filename):
    path = REPORTS / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


cleaned_df = load_json("cleaned_posts.json")
sentiment_df = load_json("sentiment_results_unified.json")
topics_df = load_json("topic_model_results_unified.json")
misinfo_df = load_json("misinformation_social_posts.json")

centrality_df = load_csv("user_centrality.csv")
interactions_df = load_csv("user_interactions.csv")
overall_sentiment_df = load_csv("overall_sentiment.csv")
platform_sentiment_df = load_csv("platform_sentiment.csv")
daily_sentiment_df = load_csv("daily_sentiment_trends.csv")
topic_keywords_df = load_csv("unified_topic_keywords.csv")
platform_topics_df = load_csv("platform_topic_distribution.csv")
misinfo_predictions_df = load_csv("misinformation_test_predictions.csv")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_percentage(value, total):
    if total == 0:
        return 0
    return round((value / total) * 100, 2)


def filter_platform(df, selected_platform):
    if df.empty or selected_platform == "All":
        return df.copy()

    if "platform" not in df.columns:
        return df.copy()

    return df[df["platform"] == selected_platform].copy()


def show_empty_message(message):
    st.info(message)


# --------------------------------------------------
# VALIDATE DATA
# --------------------------------------------------

if cleaned_df.empty:
    st.error(
        "The cleaned dataset could not be loaded. "
        "Check data/processed/cleaned_posts.json."
    )
    st.stop()

if "platform" not in cleaned_df.columns:
    st.error("The cleaned dataset does not contain a platform column.")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 Social Media Intelligence")
st.sidebar.caption("Multi-platform analytics dashboard")

platforms = ["All"] + sorted(
    cleaned_df["platform"].dropna().unique().tolist()
)

selected_platform = st.sidebar.selectbox(
    "Filter by platform",
    platforms,
)

selected_origin = st.sidebar.selectbox(
    "Filter by data origin",
    ["All"] + sorted(
        cleaned_df["data_origin"].dropna().unique().tolist()
    ) if "data_origin" in cleaned_df.columns else ["All"],
)

st.sidebar.divider()
st.sidebar.markdown("### Project Information")
st.sidebar.write("**Project:** Social Media Intelligence Platform")
st.sidebar.write("**Data sources:** YouTube, Hacker News, Reddit samples")
st.sidebar.write("**Analysis:** Sentiment, topics, misinformation and network")
st.sidebar.caption("Academic capstone project | 2026")


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

filtered_cleaned = filter_platform(cleaned_df, selected_platform)
filtered_sentiment = filter_platform(sentiment_df, selected_platform)
filtered_topics = filter_platform(topics_df, selected_platform)
filtered_misinfo = filter_platform(misinfo_df, selected_platform)

if selected_origin != "All":
    for name, df in [
        ("cleaned", filtered_cleaned),
        ("sentiment", filtered_sentiment),
        ("topics", filtered_topics),
        ("misinfo", filtered_misinfo),
    ]:
        if "data_origin" in df.columns:
            df = df[df["data_origin"] == selected_origin].copy()

        if name == "cleaned":
            filtered_cleaned = df
        elif name == "sentiment":
            filtered_sentiment = df
        elif name == "topics":
            filtered_topics = df
        else:
            filtered_misinfo = df


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌐 Social Media Intelligence Platform")

st.markdown(
    """
    An integrated analytics platform for understanding social media
    discussions through sentiment analysis, topic discovery,
    misinformation classification and interaction network analysis.
    """
)

st.caption(
    f"Current filter: {selected_platform} | "
    f"Data origin: {selected_origin}"
)

st.divider()


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

overview_tab, sentiment_tab, topics_tab, misinfo_tab, network_tab = (
    st.tabs([
        "🏠 Overview",
        "😊 Sentiment Analysis",
        "🔎 Topic Modelling",
        "⚠️ Misinformation",
        "🕸️ Network Analytics",
    ])
)


# ==================================================
# 1. OVERVIEW
# ==================================================

with overview_tab:

    st.header("Platform Overview")

    total_posts = len(filtered_cleaned)

    platform_count = (
        filtered_cleaned["platform"].nunique()
        if not filtered_cleaned.empty else 0
    )

    api_count = (
        (filtered_cleaned["data_origin"] == "api").sum()
        if "data_origin" in filtered_cleaned.columns else 0
    )

    sample_count = (
        (filtered_cleaned["data_origin"] == "sample").sum()
        if "data_origin" in filtered_cleaned.columns else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Posts", f"{total_posts:,}")
    col2.metric("Platforms", platform_count)
    col3.metric("API Records", f"{api_count:,}")
    col4.metric("Sample Records", f"{sample_count:,}")

    st.divider()


    # --------------------------------------------------
    # PROFESSIONAL OVERVIEW CHARTS
    # --------------------------------------------------

    left, right = st.columns([1.2, 1], gap="large")

    chart_layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, sans-serif",
            "color": "#DCE6F1",
            "size": 12
        },
        "margin": dict(l=15, r=15, t=35, b=15),
        "height": 350
    }

    with left:
        st.subheader("Platform Distribution")

        if not filtered_cleaned.empty:
            platform_data = (
                filtered_cleaned["platform"]
                .value_counts()
                .rename_axis("Platform")
                .reset_index(name="Posts")
                .sort_values("Posts", ascending=True)
            )

            fig_platform = px.bar(
                platform_data,
                x="Posts",
                y="Platform",
                orientation="h",
                text="Posts",
                color="Platform",
                color_discrete_map={
                    "hacker_news": "#43B5D1",
                    "youtube": "#31B5A3",
                    "reddit": "#E8A45C"
                },
                hover_data={"Posts": True}
            )

            fig_platform.update_traces(
                textposition="outside",
                marker_line_width=0,
                cliponaxis=False
            )

            fig_platform.update_layout(
                **chart_layout,
                xaxis_title="Number of Posts",
                yaxis_title="",
                xaxis=dict(showgrid=True, gridcolor="#263A50"),
                yaxis=dict(showgrid=False),
                showlegend=False
            )

            st.plotly_chart(
                fig_platform,
                use_container_width=True
            )
        else:
            show_empty_message("No platform records available.")

    with right:
        st.subheader("Data Origin Distribution")

        if (
            not filtered_cleaned.empty
            and "data_origin" in filtered_cleaned.columns
        ):
            origin_data = (
                filtered_cleaned["data_origin"]
                .value_counts()
                .rename_axis("Origin")
                .reset_index(name="Records")
            )

            fig_origin = px.pie(
                origin_data,
                names="Origin",
                values="Records",
                hole=0.68,
                color="Origin",
                color_discrete_map={
                    "api": "#31B5A3",
                    "sample": "#E8A45C"
                }
            )

            fig_origin.update_traces(
                textinfo="percent",
                textposition="inside",
                marker=dict(
                    line=dict(color="#0B1220", width=3)
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Records: %{value}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )

            fig_origin.update_layout(
                **chart_layout,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    y=-0.12,
                    x=0.5,
                    xanchor="center"
                ),
                annotations=[
                    dict(
                        text=f"<b>{len(filtered_cleaned)}</b><br>Records",
                        x=0.5,
                        y=0.5,
                        font=dict(size=19, color="#FFFFFF"),
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(
                fig_origin,
                use_container_width=True
            )
        else:
            show_empty_message("No data-origin information available.")

    # --------------------------------------------------
    # PLATFORM INTELLIGENCE SUMMARY
    # --------------------------------------------------

    st.divider()
    st.subheader("Platform Intelligence Summary")

    if not filtered_cleaned.empty:

        platform_summary = (
            filtered_cleaned["platform"]
            .value_counts()
            .rename_axis("Platform")
            .reset_index(name="Posts")
        )

        total_records = len(filtered_cleaned)

        platform_summary["Share (%)"] = (
            platform_summary["Posts"] / total_records * 100
        ).round(2)

        platform_summary = platform_summary.sort_values(
            "Posts", ascending=False
        )

        most_represented = platform_summary.iloc[0]

        summary1, summary2, summary3 = st.columns(3)

        summary1.metric(
            "Leading Platform",
            most_represented["Platform"]
        )

        summary2.metric(
            "Leading Platform Share",
            f"{most_represented['Share (%)']:.1f}%"
        )

        api_coverage = (
            filtered_cleaned["data_origin"].eq("api").sum()
            if "data_origin" in filtered_cleaned.columns
            else 0
        )

        summary3.metric(
            "API Collection Coverage",
            f"{api_coverage / total_records * 100:.1f}%"
        )

        st.dataframe(
            platform_summary,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No records available for the selected filters.")


    st.subheader("Recent Dataset Records")

    display_columns = [
        column for column in [
            "post_id",
            "platform",
            "author",
            "community",
            "created_at",
            "data_origin",
        ]
        if column in filtered_cleaned.columns
    ]

    st.dataframe(
        filtered_cleaned[display_columns].head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The dataset is a collected snapshot, not continuous real-time monitoring."
    )


# ==================================================
# 2. SENTIMENT ANALYSIS
# ==================================================

with sentiment_tab:

    st.header("Sentiment Analysis")

    if filtered_sentiment.empty:
        show_empty_message("No sentiment records available for this filter.")

    else:
        sentiment_counts = (
            filtered_sentiment["sentiment_label"]
            .value_counts()
            .reindex(["positive", "neutral", "negative"], fill_value=0)
        )

        total = len(filtered_sentiment)

        col1, col2, col3,col4 = st.columns(4)

        for col, label in zip(
            [col1, col2, col3],
            ["positive", "neutral", "negative"],
        ):
            count = int(sentiment_counts[label])
            col.metric(
                label.title(),
                count,
                f"{safe_percentage(count, total)}%",
            )
        col4.metric("Total Analysed", f"{total:,}")

        st.divider()

        left, right = st.columns(2)


        with left:
            st.subheader("Overall Sentiment Distribution")

            sentiment_chart = (
                sentiment_counts
                .rename_axis("Sentiment")
                .reset_index(name="Posts")
            )

            sentiment_chart = sentiment_chart[
                sentiment_chart["Posts"] > 0
            ]

            fig_sentiment = px.pie(
                sentiment_chart,
                names="Sentiment",
                values="Posts",
                hole=0.68,
                color="Sentiment",
                color_discrete_map={
                    "positive": "#31B5A3",
                    "neutral": "#8B9BB4",
                    "negative": "#F07878"
                }
            )

            fig_sentiment.update_traces(
                textinfo="percent",
                textposition="inside",
                marker=dict(
                    line=dict(color="#0B1220", width=3)
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Posts: %{value}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )

            fig_sentiment.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=15, r=15, t=35, b=20),
                legend=dict(
                    orientation="h",
                    y=-0.12,
                    x=0.5,
                    xanchor="center"
                ),
                annotations=[
                    dict(
                        text=f"<b>{total}</b><br>Posts",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20, color="#FFFFFF"),
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(
                fig_sentiment,
                use_container_width=True
            )


        with right:
            st.subheader("Sentiment by Platform")

            platform_sentiment = (
                filtered_sentiment.groupby(
                    ["platform", "sentiment_label"]
                )
                .size()
                .unstack(fill_value=0)
            )

            st.bar_chart(platform_sentiment)

        st.divider()

        st.subheader("Sentiment Records")

        sentiment_columns = [
            column for column in [
                "platform",
                "author",
                "text",
                "sentiment_label",
                "sentiment_score",
                "created_at",
            ]
            if column in filtered_sentiment.columns
        ]

        st.dataframe(
            filtered_sentiment[sentiment_columns],
            use_container_width=True,
            hide_index=True,
        )


        # ------------------------------------------
        # DAILY SENTIMENT ACTIVITY
        # ------------------------------------------

        st.divider()
        st.subheader("Daily Sentiment Activity")

        if "created_at" in filtered_sentiment.columns:

            trend_data = filtered_sentiment.copy()

            trend_data["created_at"] = pd.to_datetime(
                trend_data["created_at"],
                errors="coerce",
                utc=True
            )

            trend_data = trend_data.dropna(
                subset=["created_at"]
            )

            if not trend_data.empty:

                trend_data["date"] = (
                    trend_data["created_at"].dt.strftime("%Y-%m-%d")
                )

                daily_data = (
                    trend_data.groupby(
                        ["date", "sentiment_label"]
                    )
                    .size()
                    .reset_index(name="Posts")
                )

                daily_pivot = (
                    daily_data.pivot(
                        index="date",
                        columns="sentiment_label",
                        values="Posts"
                    )
                    .fillna(0)
                    .reindex(
                        columns=["positive", "neutral", "negative"],
                        fill_value=0
                    )
                    .reset_index()
                    .sort_values("date")
                )

                # ------------------------------------------
                # DAILY ACTIVITY KPI CARDS
                # ------------------------------------------

                daily_pivot["total_posts"] = (
                    daily_pivot[
                        ["positive", "neutral", "negative"]
                    ].sum(axis=1)
                )

                busiest_row = daily_pivot.loc[
                    daily_pivot["total_posts"].idxmax()
                ]

                busiest_date = busiest_row["date"]
                peak_posts = int(busiest_row["total_posts"])
                observed_dates = daily_pivot["date"].nunique()

                st.subheader("Daily Activity Insights")

                kpi1, kpi2, kpi3 = st.columns(3)

                kpi1.metric(
                    "Busiest Publication Date",
                    busiest_date
                )

                kpi2.metric(
                    "Peak Daily Posts",
                    f"{peak_posts:,}"
                )

                kpi3.metric(
                    "Observed Publication Dates",
                    f"{observed_dates:,}"
                )


                # ------------------------------------------
                # DAILY ACTIVITY KPI CARDS
                # ------------------------------------------

                daily_pivot["total_posts"] = (
                    daily_pivot[
                        ["positive", "neutral", "negative"]
                    ].sum(axis=1)
                )

                busiest_row = daily_pivot.loc[
                    daily_pivot["total_posts"].idxmax()
                ]

                busiest_date = busiest_row["date"]
                peak_posts = int(busiest_row["total_posts"])
                observed_dates = daily_pivot["date"].nunique()

                st.subheader("Daily Activity Insights")

                kpi1, kpi2, kpi3 = st.columns(3)

                kpi1.metric(
                    "Busiest Publication Date",
                    busiest_date
                )

                kpi2.metric(
                    "Peak Daily Posts",
                    f"{peak_posts:,}"
                )

                kpi3.metric(
                    "Observed Publication Dates",
                    f"{observed_dates:,}"
                )

                fig_daily = px.bar(
                    daily_pivot,
                    x="date",
                    y=["positive", "neutral", "negative"],
                    barmode="stack",
                    color_discrete_map={
                        "positive": "#31B5A3",
                        "neutral": "#8B9BB4",
                        "negative": "#F07878"
                    },
                    labels={
                        "date": "Publication Date",
                        "value": "Number of Posts",
                        "variable": "Sentiment"
                    }
                )

                fig_daily.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=420,
                    margin=dict(l=15, r=15, t=35, b=90),
                    xaxis=dict(
                        title="Observed Publication Dates",
                        type="category",
                        tickangle=-45,
                        tickmode="auto",
                        nticks=12,
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="Number of Posts",
                        rangemode="tozero",
                        showgrid=True,
                        gridcolor="#263A50"
                    ),
                    legend_title="Sentiment",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_daily,
                    use_container_width=True
                )

                st.caption(
                    f"Showing {len(daily_pivot)} observed publication dates. "
                    "Missing dates are not interpreted as zero activity."
                )

            else:
                st.info("No valid publication dates available.")

        else:
            st.info("Publication date information is unavailable.")

        # ------------------------------------------
        # MODEL CONFIDENCE ANALYSIS
        # ------------------------------------------

        st.divider()
        st.subheader("Model Confidence Analysis")

        confidence_data = filtered_sentiment.copy()

        confidence_data["sentiment_score"] = pd.to_numeric(
            confidence_data["sentiment_score"],
            errors="coerce"
        )

        confidence_data = confidence_data.dropna(
            subset=["sentiment_score"]
        )

        if not confidence_data.empty:

            average_confidence = (
                confidence_data["sentiment_score"].mean()
            )

            median_confidence = (
                confidence_data["sentiment_score"].median()
            )

            high_confidence_count = (
                confidence_data["sentiment_score"] >= 0.80
            ).sum()

            confidence1, confidence2, confidence3 = st.columns(3)

            confidence1.metric(
                "Average Model Confidence",
                f"{average_confidence:.1%}"
            )

            confidence2.metric(
                "Median Model Confidence",
                f"{median_confidence:.1%}"
            )

            confidence3.metric(
                "Predictions at 80%+ Confidence",
                f"{high_confidence_count:,}"
            )

            st.divider()

            chart_left, chart_right = st.columns(2, gap="large")

            # CONFIDENCE DISTRIBUTION

            with chart_left:

                st.markdown("**Confidence Score Distribution**")

                fig_confidence = px.histogram(
                    confidence_data,
                    x="sentiment_score",
                    nbins=15,
                    color_discrete_sequence=["#43B5D1"],
                    labels={
                        "sentiment_score": "Confidence Score",
                        "count": "Number of Predictions"
                    }
                )

                fig_confidence.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=15, r=15, t=35, b=35),
                    xaxis=dict(
                        title="Model Confidence",
                        range=[0, 1],
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="Number of Predictions",
                        showgrid=True,
                        gridcolor="#263A50"
                    ),
                    showlegend=False
                )

                fig_confidence.update_traces(
                    marker_line_color="#0B1220",
                    marker_line_width=1,
                    hovertemplate=(
                        "Confidence range: %{x}<br>"
                        "Predictions: %{y}"
                        "<extra></extra>"
                    )
                )

                st.plotly_chart(
                    fig_confidence,
                    use_container_width=True
                )

            # AVERAGE CONFIDENCE BY SENTIMENT

            with chart_right:

                st.markdown("**Average Confidence by Sentiment**")

                confidence_by_sentiment = (
                    confidence_data.groupby("sentiment_label")
                    ["sentiment_score"]
                    .mean()
                    .reset_index()
                )

                fig_confidence_sentiment = px.bar(
                    confidence_by_sentiment,
                    x="sentiment_label",
                    y="sentiment_score",
                    color="sentiment_label",
                    text=confidence_by_sentiment[
                        "sentiment_score"
                    ].map(lambda x: f"{x:.1%}"),
                    color_discrete_map={
                        "positive": "#31B5A3",
                        "neutral": "#8B9BB4",
                        "negative": "#F07878"
                    },
                    labels={
                        "sentiment_label": "Predicted Sentiment",
                        "sentiment_score": "Average Confidence"
                    }
                )

                fig_confidence_sentiment.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=15, r=15, t=35, b=35),
                    xaxis_title="Predicted Sentiment",
                    yaxis=dict(
                        title="Average Confidence",
                        tickformat=".0%",
                        range=[0, 1],
                        showgrid=True,
                        gridcolor="#263A50"
                    ),
                    showlegend=False
                )

                fig_confidence_sentiment.update_traces(
                    textposition="outside",
                    cliponaxis=False
                )

                st.plotly_chart(
                    fig_confidence_sentiment,
                    use_container_width=True
                )

            st.caption(
                "Confidence scores represent the model's confidence "
                "in its predicted class. They do not measure sentiment "
                "intensity or guarantee prediction correctness."
            )

        else:
            st.info("No valid confidence scores are available.")

        # ------------------------------------------
        # AUTOMATED SENTIMENT INSIGHTS
        # ------------------------------------------

        st.divider()
        st.subheader("Automated Sentiment Insights")

        insight_data = filtered_sentiment.copy()

        insight_data["sentiment_label"] = (
            insight_data["sentiment_label"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        if not insight_data.empty:

            total_insights = len(insight_data)

            overall_counts = (
                insight_data["sentiment_label"]
                .value_counts()
            )

            dominant_sentiment = overall_counts.idxmax()

            dominant_count = int(
                overall_counts.max()
            )

            dominant_percentage = (
                dominant_count / total_insights * 100
            )

            positive_count = int(
                overall_counts.get("positive", 0)
            )

            negative_count = int(
                overall_counts.get("negative", 0)
            )

            sentiment_balance = (
                positive_count - negative_count
            )

            platform_insights = (
                insight_data.groupby(
                    ["platform", "sentiment_label"]
                )
                .size()
                .unstack(fill_value=0)
            )

            for label in ["positive", "neutral", "negative"]:
                if label not in platform_insights.columns:
                    platform_insights[label] = 0

            platform_insights["total"] = (
                platform_insights[
                    ["positive", "neutral", "negative"]
                ].sum(axis=1)
            )

            platform_insights["negative_percentage"] = (
                platform_insights["negative"]
                / platform_insights["total"] * 100
            )

            highest_negative_platform = (
                platform_insights[
                    "negative_percentage"
                ].idxmax()
            )

            highest_negative_share = (
                platform_insights.loc[
                    highest_negative_platform,
                    "negative_percentage"
                ]
            )

            highest_negative_records = int(
                platform_insights.loc[
                    highest_negative_platform,
                    "total"
                ]
            )

            # INSIGHT CARDS

            insight1, insight2 = st.columns(2)

            with insight1:

                st.markdown("**Dominant Sentiment**")

                st.metric(
                    "Most frequent prediction",
                    dominant_sentiment.title(),
                    f"{dominant_percentage:.1f}% of records"
                )

            with insight2:

                st.markdown("**Positive vs Negative Balance**")

                if sentiment_balance > 0:
                    balance_message = (
                        f"{sentiment_balance} more positive predictions"
                    )
                elif sentiment_balance < 0:
                    balance_message = (
                        f"{abs(sentiment_balance)} more negative predictions"
                    )
                else:
                    balance_message = "Equal positive and negative counts"

                st.metric(
                    "Positive minus Negative",
                    f"{sentiment_balance:+,}",
                    balance_message
                )

            st.markdown("**Platform with Highest Negative Share**")

            st.write(
                f"**{highest_negative_platform}** has the highest "
                f"negative prediction proportion at "
                f"**{highest_negative_share:.1f}%**, based on "
                f"**{highest_negative_records} records**."
            )

            if highest_negative_records < 10:
                st.warning(
                    "This platform has fewer than 10 records. "
                    "Its negative sentiment percentage is based on "
                    "a small sample and should not be treated as "
                    "a reliable platform-level conclusion."
                )

            # AUTOMATED INTERPRETATION

            st.markdown("**Analytical Summary**")

            st.info(
                f"Across {total_insights} selected records, "
                f"{dominant_sentiment} sentiment is the most "
                f"frequent prediction, representing "
                f"{dominant_percentage:.1f}% of the dataset. "
                f"The difference between positive and negative "
                f"predictions is {sentiment_balance:+,} records."
            )

        else:
            st.info(
                "No records are available to generate sentiment insights."
            )



        st.warning(
            "Sentiment predictions use a pretrained Twitter-style model. "
            "Results on YouTube metadata and Hacker News text are exploratory "
            "and have not been validated against human-labelled data."
        )


# ==================================================
# 3. TOPIC MODELLING
# ==================================================



with topics_tab:

    st.header("Topic Modelling")
    st.caption(
        "Explore discovered discussion themes, their distribution, "
        "platform coverage, and representative posts."
    )

    if filtered_topics.empty:
        show_empty_message("No topic records available.")

    else:

        # ------------------------------------------
        # TOPIC LABELS
        # ------------------------------------------

        topic_labels = {
            -1: "Unassigned",
            0: "Artificial Intelligence",
            1: "Climate Change",
            2: "Educational Technology",
            3: "Technology & Development",
            4: "General Technology Discussions",
        }

        # ------------------------------------------
        # TOPIC SUMMARY
        # ------------------------------------------

        topic_summary = (
            filtered_topics.groupby("topic_id")
            .agg(
                post_count=("post_id", "count"),
                average_strength=("topic_strength", "mean"),
            )
            .reset_index()
        )

        topic_summary["topic_id"] = (
            topic_summary["topic_id"].astype(int)
        )

        topic_summary["topic_name"] = (
            topic_summary["topic_id"].map(
                lambda topic: topic_labels.get(
                    topic, f"Topic {topic}"
                )
            )
        )

        if "topic_keywords" in filtered_topics.columns:

            keyword_map = (
                filtered_topics.groupby("topic_id")["topic_keywords"]
                .first()
                .to_dict()
            )

            def format_keywords(value):
                if isinstance(value, list):
                    return ", ".join(str(word) for word in value)
                if isinstance(value, str):
                    return value
                return "Keywords unavailable"

            topic_summary["keywords"] = (
                topic_summary["topic_id"].map(
                    lambda topic: format_keywords(
                        keyword_map.get(topic)
                    )
                )
            )

        else:
            topic_summary["keywords"] = "Keywords unavailable"

        topic_summary = topic_summary.sort_values(
            "post_count", ascending=False
        )

        # ------------------------------------------
        # KPI CARDS
        # ------------------------------------------

        total_posts = len(filtered_topics)

        assigned_topics = filtered_topics[
            filtered_topics["topic_id"] != -1
        ]

        topic_count = assigned_topics["topic_id"].nunique()

        average_strength = assigned_topics["topic_strength"].mean()

        assigned_summary = topic_summary[
            topic_summary["topic_id"] != -1
        ]

        if not assigned_summary.empty:
            dominant_topic = assigned_summary.iloc[0]["topic_name"]
            dominant_count = int(
                assigned_summary.iloc[0]["post_count"]
            )
        else:
            dominant_topic = "N/A"
            dominant_count = 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Posts Analysed", total_posts)
        col2.metric("Discovered Topics", topic_count)

        col3.metric(
            "Average Topic Strength",
            f"{average_strength:.3f}"
            if pd.notna(average_strength) else "N/A",
        )

        col4.metric(
            "Most Discussed Topic",
            dominant_topic,
            f"{dominant_count} posts",
        )

        st.divider()

        # ------------------------------------------
        # TOPIC DISTRIBUTION
        # ------------------------------------------

        st.subheader("Topic Distribution")

        left, right = st.columns(2)

        with left:

            fig_topic = px.pie(
                topic_summary,
                names="topic_name",
                values="post_count",
                hole=0.55,
                title="Share of Posts by Topic",
            )

            fig_topic.update_layout(
                legend_title="Topics",
                margin=dict(t=50, b=20, l=10, r=10),
            )

            st.plotly_chart(
                fig_topic,
                use_container_width=True,
            )

        with right:

            fig_topic_bar = px.bar(
                topic_summary.sort_values("post_count"),
                x="post_count",
                y="topic_name",
                orientation="h",
                text="post_count",
                title="Number of Posts per Topic",
                labels={
                    "post_count": "Number of Posts",
                    "topic_name": "Topic",
                },
            )

            fig_topic_bar.update_layout(
                yaxis_title="",
                xaxis_title="Number of Posts",
                margin=dict(t=50, b=20, l=10, r=10),
            )

            st.plotly_chart(
                fig_topic_bar,
                use_container_width=True,
            )

        st.divider()

        # ------------------------------------------
        # PLATFORM-WISE TOPIC COMPARISON
        # ------------------------------------------

        st.subheader("Topic Distribution Across Platforms")

        if "platform" in filtered_topics.columns:

            platform_topics = (
                filtered_topics.groupby(["platform", "topic_id"])
                .size()
                .reset_index(name="post_count")
            )

            platform_topics["topic_name"] = (
                platform_topics["topic_id"].map(
                    lambda topic: topic_labels.get(
                        int(topic), f"Topic {topic}"
                    )
                )
            )

            fig_platform_topics = px.bar(
                platform_topics,
                x="platform",
                y="post_count",
                color="topic_name",
                barmode="stack",
                title="Topic Coverage by Platform",
                labels={
                    "platform": "Platform",
                    "post_count": "Number of Posts",
                    "topic_name": "Topic",
                },
            )

            fig_platform_topics.update_layout(
                margin=dict(t=50, b=20, l=10, r=10),
            )

            st.plotly_chart(
                fig_platform_topics,
                use_container_width=True,
            )

        st.divider()

        # ------------------------------------------
        # TOPIC STRENGTH ANALYSIS
        # ------------------------------------------

        st.subheader("Topic Strength Analysis")

        fig_strength = px.histogram(
            filtered_topics,
            x="topic_strength",
            color="topic_id",
            nbins=20,
            title="Distribution of Topic Strength",
            labels={
                "topic_strength": "Topic Strength",
                "topic_id": "Topic ID",
            },
        )

        st.plotly_chart(
            fig_strength,
            use_container_width=True,
        )

        st.caption(
            "Topic strength represents a normalized NMF component "
            "contribution. It is not a probability or a measure of "
            "model confidence."
        )

        st.divider()

        # ------------------------------------------
        # INTERACTIVE TOPIC EXPLORER
        # ------------------------------------------

        st.subheader("Explore Individual Topics")

        topic_options = {
            row["topic_name"]: int(row["topic_id"])
            for _, row in topic_summary.iterrows()
        }

        selected_topic_name = st.selectbox(
            "Select a topic to explore",
            list(topic_options.keys()),
        )

        selected_topic_id = topic_options[selected_topic_name]

        selected_topic_data = filtered_topics[
            filtered_topics["topic_id"] == selected_topic_id
        ]

        selected_summary = topic_summary[
            topic_summary["topic_id"] == selected_topic_id
        ].iloc[0]

        st.markdown(f"### {selected_topic_name}")

        st.write(
            f"**Posts assigned:** {len(selected_topic_data)}"
        )

        st.write(
            f"**Top keywords:** {selected_summary['keywords']}"
        )

        st.write("**Representative posts**")

        explorer_columns = [
            column for column in [
                "platform",
                "author",
                "text",
                "topic_strength",
            ]
            if column in selected_topic_data.columns
        ]

        st.dataframe(
            selected_topic_data[explorer_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # ------------------------------------------
        # COMPLETE TOPIC SUMMARY
        # ------------------------------------------

        st.subheader("Topic Summary")

        st.dataframe(
            topic_summary[
                [
                    "topic_name",
                    "post_count",
                    "average_strength",
                    "keywords",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------------------
        # MODEL LIMITATIONS
        # ------------------------------------------

        st.warning(
            "Topics were discovered using TF-IDF and Non-negative "
            "Matrix Factorization (NMF). Topic labels are interpretive "
            "descriptions of the model keywords, not manually verified "
            "categories. Some topics may contain broad or overlapping "
            "discussions. Topic strength is not a calibrated probability."
        )


# ==================================================
# 4. MISINFORMATION ANALYSIS
# ==================================================

with misinfo_tab:

    st.header("Misinformation Intelligence")
    st.caption(
        "Explore experimental text-classifier predictions "
        "and evaluate model performance."
    )

    st.warning(
        "Important: These are experimental text-classifier predictions, "
        "not verified fact-checks. The model was trained on the LIAR "
        "political-claim dataset. Its performance may not transfer reliably "
        "to general YouTube or Hacker News content. Every social post remains "
        "marked as Unverified."
    )

    if filtered_misinfo.empty:
        show_empty_message("No misinformation records available.")

    else:

        # ------------------------------------------
        # PREDICTION SUMMARY
        # ------------------------------------------

        prediction_counts = (
            filtered_misinfo["misinformation_prediction"]
            .value_counts()
        )

        potentially_misleading = int(
            prediction_counts.get("Potentially misleading", 0)
        )

        likely_reliable = int(
            prediction_counts.get("Likely reliable", 0)
        )

        total_predictions = len(filtered_misinfo)

        misleading_percentage = (
            potentially_misleading / total_predictions * 100
            if total_predictions else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Analysed Posts", total_predictions)

        col2.metric(
            "Potentially Misleading",
            potentially_misleading,
            f"{misleading_percentage:.1f}% of posts",
        )

        col3.metric("Likely Reliable", likely_reliable)

        col4.metric(
            "Unverified Records",
            int(
                (filtered_misinfo["fact_check_status"] == "Unverified").sum()
            ) if "fact_check_status" in filtered_misinfo.columns else "N/A",
        )

        st.divider()

        # ------------------------------------------
        # PREDICTION DISTRIBUTION
        # ------------------------------------------

        st.subheader("Prediction Distribution")

        left, right = st.columns(2)

        with left:

            prediction_chart = (
                prediction_counts
                .rename_axis("Prediction")
                .reset_index(name="Posts")
            )

            fig_prediction = px.pie(
                prediction_chart,
                names="Prediction",
                values="Posts",
                hole=0.55,
                title="Classification Breakdown",
            )

            st.plotly_chart(
                fig_prediction,
                use_container_width=True,
            )

        with right:

            st.subheader("Prediction by Platform")

            prediction_platform = (
                filtered_misinfo.groupby(
                    ["platform", "misinformation_prediction"]
                )
                .size()
                .reset_index(name="post_count")
            )

            fig_platform = px.bar(
                prediction_platform,
                x="platform",
                y="post_count",
                color="misinformation_prediction",
                barmode="group",
                title="Platform-wise Classification",
                labels={
                    "platform": "Platform",
                    "post_count": "Number of Posts",
                    "misinformation_prediction": "Prediction",
                },
            )

            st.plotly_chart(
                fig_platform,
                use_container_width=True,
            )

        st.divider()

        # ------------------------------------------
        # MODEL EVALUATION
        # ------------------------------------------

        st.subheader("Model Evaluation — Held-out LIAR Test Set")

        st.caption(
            "The following metrics were measured on 1,267 held-out "
            "LIAR test records, not on the collected social media posts."
        )

        metric1, metric2, metric3, metric4, metric5 = st.columns(5)

        metric1.metric("Accuracy", "60.62%")
        metric2.metric("ROC-AUC", "0.6286")
        metric3.metric("Precision", "54.36%")
        metric4.metric("Recall", "78.34%")
        metric5.metric("F1-Score", "64.18%")

        st.divider()

        # ------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------

        st.subheader("Confusion Matrix")

        confusion_matrix = [
            [511, 307],
            [192, 257],
        ]

        confusion_labels = [
            "Likely Reliable (0)",
            "Potentially Misleading (1)",
        ]

        fig_cm = px.imshow(
            confusion_matrix,
            x=confusion_labels,
            y=confusion_labels,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            labels={
                "x": "Predicted Class",
                "y": "Actual Class",
                "color": "Records",
            },
        )

        fig_cm.update_layout(
            title="LIAR Test Set — Actual vs Predicted",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True,
        )

        st.caption(
            "Class 0 represents likely reliable claims; class 1 represents "
            "potentially misleading claims under this project's binary "
            "label mapping."
        )

        st.divider()

        # ------------------------------------------
        # PREDICTION CONFIDENCE
        # ------------------------------------------

        st.subheader("Prediction Confidence Analysis")

        if "misinformation_confidence" in filtered_misinfo.columns:

            confidence_data = filtered_misinfo.dropna(
                subset=["misinformation_confidence"]
            ).copy()

            confidence_data["misinformation_confidence"] = pd.to_numeric(
                confidence_data["misinformation_confidence"],
                errors="coerce",
            )

            confidence_data = confidence_data.dropna(
                subset=["misinformation_confidence"]
            )

            if not confidence_data.empty:

                confidence1, confidence2, confidence3 = st.columns(3)

                confidence1.metric(
                    "Average Confidence",
                    f"{confidence_data['misinformation_confidence'].mean():.3f}",
                )

                confidence2.metric(
                    "Median Confidence",
                    f"{confidence_data['misinformation_confidence'].median():.3f}",
                )

                confidence3.metric(
                    "High Confidence Predictions",
                    int(
                        (
                            confidence_data["misinformation_confidence"] >= 0.80
                        ).sum()
                    ),
                )

                fig_confidence = px.histogram(
                    confidence_data,
                    x="misinformation_confidence",
                    color="misinformation_prediction",
                    nbins=20,
                    title="Distribution of Prediction Confidence",
                    labels={
                        "misinformation_confidence": "Model Confidence",
                        "misinformation_prediction": "Prediction",
                    },
                )

                st.plotly_chart(
                    fig_confidence,
                    use_container_width=True,
                )

                st.caption(
                    "Confidence is the classifier's predicted probability "
                    "for its selected class. It does not establish whether "
                    "a statement is factually true."
                )

            else:
                st.info("No valid confidence scores are available.")

        else:
            st.info("Prediction confidence data is unavailable.")

        st.divider()

        # ------------------------------------------
        # CLASSIFICATION RESULTS
        # ------------------------------------------

        st.subheader("Social Media Classification Results")

        search_text = st.text_input(
            "Search posts by text or author",
            placeholder="Enter a keyword or username...",
        )

        display_data = filtered_misinfo.copy()

        if search_text.strip():

            search_mask = (
                display_data["text"].fillna("").astype(str)
                .str.contains(search_text, case=False, na=False)
            )

            if "author" in display_data.columns:
                author_mask = (
                    display_data["author"].fillna("").astype(str)
                    .str.contains(search_text, case=False, na=False)
                )

                search_mask = search_mask | author_mask

            display_data = display_data[search_mask]

        misinfo_columns = [
            column for column in [
                "platform",
                "author",
                "text",
                "misinformation_prediction",
                "misinformation_confidence",
                "fact_check_status",
            ]
            if column in display_data.columns
        ]

        st.caption(f"Showing {len(display_data)} matching records")

        st.dataframe(
            display_data[misinfo_columns],
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------------------
        # MODEL LIMITATIONS
        # ------------------------------------------

        st.warning(
            "Interpretation limitation: The classifier was trained on the "
            "LIAR political-claim dataset and achieved approximately 60.62% "
            "accuracy on its held-out test set. Applying it to general social "
            "media content involves domain shift. Predictions should be "
            "treated as screening signals for further review, not evidence "
            "of misinformation or factual verification."
        )



# ==================================================
# 5. NETWORK ANALYTICS
# ==================================================

with network_tab:

    st.header("Hacker News Interaction Network")

    st.markdown(
        """
        This section explores the interaction structure derived from
        sampled Hacker News comments and story authors.
        """
    )

    # Display network only for All or Hacker News selection
    if selected_platform in ("All", "Hacker News"):

        if not centrality_df.empty:

            col1, col2, col3 = st.columns(3)

            col1.metric("Network Nodes", len(centrality_df))

            if not interactions_df.empty:
                col2.metric(
                    "Recorded Interactions",
                    len(interactions_df),
                )
            else:
                col2.metric("Recorded Interactions", "N/A")

            col3.metric(
                "Average Degree Centrality",
                f"{centrality_df['degree_centrality'].mean():.4f}",
            )

            st.divider()

            st.subheader("Interaction Network Visualization")

            graph_image = FIGURES / "hackernews_interaction_network.png"

            if graph_image.exists():
                st.image(
                    str(graph_image),
                    use_container_width=True,
                    caption="Hacker News sampled interaction network",
                )
            else:
                show_empty_message(
                    "Network visualization image not found."
                )

            st.divider()

            st.subheader("Top Users by Degree Centrality")

            top_users = centrality_df.sort_values(
                "degree_centrality",
                ascending=False,
            ).head(10)

            st.bar_chart(
                top_users.set_index("username")["degree_centrality"]
            )

            st.subheader("User Centrality Table")

            st.dataframe(
                centrality_df.sort_values(
                    "degree_centrality",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

            if not interactions_df.empty:
                st.divider()
                st.subheader("Recorded User Interactions")

                st.dataframe(
                    interactions_df,
                    use_container_width=True,
                    hide_index=True,
                )

            st.info(
                "Network analysis uses up to two sampled direct comments "
                "per story from the collected Hacker News dataset. It does "
                "not represent the complete Hacker News interaction network."
            )

        else:
            show_empty_message(
                "Centrality report not found. Check "
                "reports/evaluation/user_centrality.csv."
            )

    else:

        st.info(
            f"Network analysis is currently available only for Hacker News. "
            f"You selected {selected_platform}. "
            "YouTube and Reddit interaction networks have not yet "
            "been implemented."
        )



# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Social Media Intelligence Platform | "
    "Academic Capstone Project | 2026"
)
