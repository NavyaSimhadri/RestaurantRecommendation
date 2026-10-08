import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f7f8fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}


/* ==========================================================
   PAGE HEADER
   ========================================================== */

.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #202124;
    margin-bottom: 4px;
}

.page-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 28px;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

.metric-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 20px;
    min-height: 115px;
}

.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
    color: #202124;
    margin-top: 10px;
}

.metric-help {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 4px;
}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #202124;
    margin-top: 28px;
    margin-bottom: 14px;
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {
    background-color: #e23744;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #c92f3b;
    color: white;
}


/* ==========================================================
   TABLE
   ========================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MAIN DATA
# ============================================================

@st.cache_data
def load_data():

    # --------------------------------------------------------
    # Increase CSV field size limit.
    #
    # The Zomato dataset contains very large review fields.
    # --------------------------------------------------------

    csv.field_size_limit(10_000_000)

    # --------------------------------------------------------
    # Only load columns actually required by the application.
    #
    # reviews_list and menu_item are NOT needed for:
    # - dashboard
    # - filtering
    # - recommendation system
    #
    # This makes loading much safer and faster.
    # --------------------------------------------------------

    required_columns = [
        "name",
        "location",
        "rest_type",
        "dish_liked",
        "cuisines",
        "online_order",
        "book_table",
        "rate",
        "votes",
        "approx_cost(for two people)",
        "listed_in(type)",
        "listed_in(city)"
    ]

    data = pd.read_csv(
        "zomato.csv",
        usecols=required_columns,
        engine="python",
        quotechar='"',
        doublequote=True,
        on_bad_lines="warn"
    )

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    data = data.drop_duplicates()

    # --------------------------------------------------------
    # Clean rating
    # --------------------------------------------------------

    data["rate"] = (
        data["rate"]
        .astype(str)
        .str.replace("/5", "", regex=False)
        .str.strip()
    )

    data["rate"] = pd.to_numeric(
        data["rate"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean cost
    # --------------------------------------------------------

    data["approx_cost(for two people)"] = (
        data["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    data["approx_cost(for two people)"] = pd.to_numeric(
        data["approx_cost(for two people)"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean text columns
    # --------------------------------------------------------

    text_columns = [
        "name",
        "location",
        "rest_type",
        "dish_liked",
        "cuisines",
        "online_order",
        "book_table",
        "listed_in(type)",
        "listed_in(city)"
    ]

    for column in text_columns:

        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return data


df = load_data()


# ============================================================
# CREATE UNIQUE RESTAURANT DATASET
# ============================================================

@st.cache_data
def create_restaurant_data(data):

    restaurant_data = data.copy()

    # --------------------------------------------------------
    # Sort by votes.
    # --------------------------------------------------------

    restaurant_data = restaurant_data.sort_values(
        "votes",
        ascending=False
    )

    # --------------------------------------------------------
    # Treat name + location as one restaurant.
    #
    # The original dataset can contain multiple records
    # for the same restaurant.
    # --------------------------------------------------------

    restaurant_data = restaurant_data.drop_duplicates(
        subset=[
            "name",
            "location"
        ],
        keep="first"
    )

    restaurant_data = restaurant_data.reset_index(
        drop=True
    )

    return restaurant_data


restaurant_df = create_restaurant_data(df)


# ============================================================
# LOAD PYSPARK RESULTS
# ============================================================

@st.cache_data
def load_spark_results():

    output_dir = "spark_output"

    files = {
        "top_locations": "top_locations.csv",
        "top_cuisines": "top_cuisines.csv",
        "location_ratings": "location_ratings.csv",
        "online_order": "online_order_analysis.csv",
        "table_booking": "table_booking_analysis.csv",
        "restaurant_types": "restaurant_types.csv",
        "popular_restaurants": "popular_restaurants.csv",
        "high_rated": "high_rated_restaurants.csv"
    }

    results = {}

    for key, filename in files.items():

        path = f"{output_dir}/{filename}"

        try:

            results[key] = pd.read_csv(path)

        except FileNotFoundError:

            results[key] = pd.DataFrame()

    return results


spark_results = load_spark_results()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:700;
            color:#202124;
        ">
            Food Analytics
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size:13px;
            color:#6b7280;
            margin-top:4px;
        ">
            Restaurant intelligence dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Restaurant Analysis",
            "Spark Analytics",
            "Recommendations"
        ]
    )

    st.divider()

    st.caption("Food Delivery Analytics")
    st.caption(
        "Python • Pandas • PySpark • Plotly • Scikit-learn"
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="page-title">Food Delivery Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Restaurant performance analysis and recommendation system'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# KPI VALUES
# ============================================================

total_restaurants = len(
    restaurant_df
)

average_rating = df["rate"].mean()

average_cost = (
    df["approx_cost(for two people)"].mean()
)

total_locations = (
    df["location"]
    .replace("", pd.NA)
    .nunique()
)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Restaurants</div>'
        f'<div class="metric-value">{total_restaurants:,}</div>'
        '<div class="metric-help">Unique restaurants</div>'
        '</div>',
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Average Rating</div>'
        f'<div class="metric-value">{average_rating:.2f}</div>'
        '<div class="metric-help">Out of 5</div>'
        '</div>',
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Average Cost</div>'
        f'<div class="metric-value">₹{average_cost:,.0f}</div>'
        '<div class="metric-help">For two people</div>'
        '</div>',
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Locations</div>'
        f'<div class="metric-value">{total_locations}</div>'
        '<div class="metric-help">Restaurant locations</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        'Restaurant Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # RATING DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        rating_data = df.dropna(
            subset=["rate"]
        )

        fig = px.histogram(
            rating_data,
            x="rate",
            nbins=15,
            title="Restaurant Rating Distribution"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Rating",
            yaxis_title="Number of Restaurants"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # TOP LOCATIONS
    # --------------------------------------------------------

    with col2:

        location_data = (
            df[df["location"] != ""]
            ["location"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        location_data.columns = [
            "Location",
            "Restaurants"
        ]

        fig = px.bar(
            location_data,
            x="Restaurants",
            y="Location",
            orientation="h",
            title="Top Restaurant Locations"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # CUISINE ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Cuisine Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    cuisine_data = (
        df[df["cuisines"] != ""]
        ["cuisines"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(15)
        .reset_index()
    )

    cuisine_data.columns = [
        "Cuisine",
        "Restaurants"
    ]

    fig = px.bar(
        cuisine_data,
        x="Restaurants",
        y="Cuisine",
        orientation="h",
        title="Most Common Cuisines"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # ONLINE ORDER + TABLE BOOKING
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # ONLINE ORDER
    # --------------------------------------------------------

    with col1:

        online_data = (
            df["online_order"]
            .value_counts()
            .reset_index()
        )

        online_data.columns = [
            "Online Order",
            "Restaurants"
        ]

        fig = px.bar(
            online_data,
            x="Online Order",
            y="Restaurants",
            title="Online Ordering Availability"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # TABLE BOOKING
    # --------------------------------------------------------

    with col2:

        booking_data = (
            df["book_table"]
            .value_counts()
            .reset_index()
        )

        booking_data.columns = [
            "Table Booking",
            "Restaurants"
        ]

        fig = px.bar(
            booking_data,
            x="Table Booking",
            y="Restaurants",
            title="Table Booking Availability"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# RESTAURANT ANALYSIS PAGE
# ============================================================

elif page == "Restaurant Analysis":

    st.markdown(
        '<div class="section-title">'
        'Restaurant Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # FILTERS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    with col1:

        locations = sorted(
            df[
                df["location"] != ""
            ]["location"]
            .unique()
            .tolist()
        )

        location_filter = st.selectbox(
            "Location",
            ["All"] + locations
        )


    # --------------------------------------------------------
    # CUISINE
    # --------------------------------------------------------

    with col2:

        cuisine_filter = st.text_input(
            "Cuisine contains",
            placeholder="Example: Biryani"
        )


    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    with col3:

        minimum_rating = st.slider(
            "Minimum Rating",
            0.0,
            5.0,
            0.0,
            0.1
        )


    # --------------------------------------------------------
    # ONLINE ORDER
    # --------------------------------------------------------

    with col4:

        order_filter = st.selectbox(
            "Online Ordering",
            ["All", "Yes", "No"]
        )


    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered = restaurant_df.copy()


    if location_filter != "All":

        filtered = filtered[
            filtered["location"]
            == location_filter
        ]


    if cuisine_filter.strip() != "":

        filtered = filtered[
            filtered["cuisines"]
            .str.contains(
                cuisine_filter,
                case=False,
                na=False
            )
        ]


    filtered = filtered[
        filtered["rate"]
        .fillna(0)
        >= minimum_rating
    ]


    if order_filter != "All":

        filtered = filtered[
            filtered["online_order"]
            == order_filter
        ]


    st.write(
        f"Showing **{len(filtered):,} unique restaurants**"
    )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Restaurant Results'
        '</div>',
        unsafe_allow_html=True
    )

    result_columns = [
        "name",
        "location",
        "rest_type",
        "cuisines",
        "rate",
        "votes",
        "approx_cost(for two people)",
        "online_order"
    ]

    result = (
        filtered
        .sort_values(
            "votes",
            ascending=False
        )
        .head(25)
    )

    st.dataframe(
        result[result_columns],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PRICE VS RATING
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Price and Rating'
        '</div>',
        unsafe_allow_html=True
    )

    chart_data = (
        filtered
        .dropna(
            subset=[
                "rate",
                "approx_cost(for two people)"
            ]
        )
        .head(3000)
    )

    fig = px.scatter(
        chart_data,
        x="approx_cost(for two people)",
        y="rate",
        size="votes",
        hover_name="name",
        title="Restaurant Cost vs Rating"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Cost for Two",
        yaxis_title="Rating"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SPARK ANALYTICS PAGE
# ============================================================

elif page == "Spark Analytics":

    st.markdown(
        '<div class="section-title">'
        'PySpark Big Data Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Analytics generated using PySpark from the Zomato dataset."
    )


    # ========================================================
    # TOP LOCATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Top Restaurant Locations'
        '</div>',
        unsafe_allow_html=True
    )

    top_locations = spark_results["top_locations"]

    if not top_locations.empty:

        fig = px.bar(
            top_locations,
            x="restaurant_count",
            y="location",
            orientation="h",
            title="Top 10 Locations by Restaurant Count"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Number of Restaurants",
            yaxis_title="Location"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Top location Spark results are not available."
        )


    # ========================================================
    # TOP CUISINES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Top Cuisines'
        '</div>',
        unsafe_allow_html=True
    )

    top_cuisines = spark_results["top_cuisines"]

    if not top_cuisines.empty:

        fig = px.bar(
            top_cuisines,
            x="restaurant_count",
            y="cuisine",
            orientation="h",
            title="Most Common Cuisines"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Number of Restaurants",
            yaxis_title="Cuisine"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Cuisine Spark results are not available."
        )


    # ========================================================
    # LOCATION RATINGS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Highest Rated Locations'
        '</div>',
        unsafe_allow_html=True
    )

    location_ratings = spark_results["location_ratings"]

    if not location_ratings.empty:

        fig = px.bar(
            location_ratings,
            x="average_rating",
            y="location",
            orientation="h",
            title="Top Locations by Average Rating"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Average Rating",
            yaxis_title="Location"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Location rating Spark results are not available."
        )


    # ========================================================
    # ONLINE ORDER + TABLE BOOKING
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # ONLINE ORDER
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">'
            'Online Ordering'
            '</div>',
            unsafe_allow_html=True
        )

        online_order = spark_results["online_order"]

        if not online_order.empty:

            fig = px.pie(
                online_order,
                names="online_order",
                values="restaurant_count",
                title="Online Ordering Availability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Online order data unavailable."
            )


    # --------------------------------------------------------
    # TABLE BOOKING
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">'
            'Table Booking'
            '</div>',
            unsafe_allow_html=True
        )

        table_booking = spark_results["table_booking"]

        if not table_booking.empty:

            fig = px.pie(
                table_booking,
                names="book_table",
                values="restaurant_count",
                title="Table Booking Availability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Table booking data unavailable."
            )


    # ========================================================
    # RESTAURANT TYPES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Restaurant Types'
        '</div>',
        unsafe_allow_html=True
    )

    restaurant_types = spark_results["restaurant_types"]

    if not restaurant_types.empty:

        fig = px.bar(
            restaurant_types,
            x="restaurant_count",
            y="rest_type",
            orientation="h",
            title="Most Common Restaurant Types"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Number of Restaurants",
            yaxis_title="Restaurant Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Restaurant type data unavailable."
        )


    # ========================================================
    # POPULAR RESTAURANTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Most Popular Restaurants'
        '</div>',
        unsafe_allow_html=True
    )

    popular = spark_results["popular_restaurants"]

    if not popular.empty:

        st.dataframe(
            popular,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Popular restaurant data unavailable."
        )


    # ========================================================
    # HIGH-RATED RESTAURANTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Highest Rated Restaurants'
        '</div>',
        unsafe_allow_html=True
    )

    high_rated = spark_results["high_rated"]

    if not high_rated.empty:

        st.dataframe(
            high_rated,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "High-rated restaurant data unavailable."
        )


# ============================================================
# HYBRID RECOMMENDATION SYSTEM
# ============================================================

elif page == "Recommendations":

    st.markdown(
        '<div class="section-title">'
        'Restaurant Recommendations'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Recommendations combine cuisine, location, "
        "restaurant type, rating, popularity and price."
    )


    # ========================================================
    # PREPARE RECOMMENDATION DATA
    # ========================================================

    recommendation_df = restaurant_df.copy()

    recommendation_df["features"] = (
        recommendation_df["location"]
        .fillna("")
        + " "
        + recommendation_df["rest_type"]
        .fillna("")
        + " "
        + recommendation_df["cuisines"]
        .fillna("")
        + " "
        + recommendation_df["listed_in(type)"]
        .fillna("")
    )


    # ========================================================
    # TF-IDF
    # ========================================================

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        recommendation_df["features"]
    )


    # ========================================================
    # RESTAURANT SELECTION
    # ========================================================

    restaurant_names = sorted(
        recommendation_df["name"]
        .dropna()
        .unique()
        .tolist()
    )

    restaurant = st.selectbox(
        "Select a restaurant",
        restaurant_names
    )


    # ========================================================
    # RECOMMENDATION BUTTON
    # ========================================================

    if st.button(
        "Find Recommended Restaurants"
    ):

        matches = recommendation_df[
            recommendation_df["name"]
            == restaurant
        ]


        if matches.empty:

            st.error(
                "Restaurant not found."
            )

        else:

            # =================================================
            # SELECTED RESTAURANT
            # =================================================

            index = matches.index[0]

            selected = recommendation_df.loc[index]


            # =================================================
            # CONTENT SIMILARITY
            # =================================================

            selected_vector = (
                tfidf_matrix[index]
            )

            similarity_result = cosine_similarity(
                selected_vector,
                tfidf_matrix,
                dense_output=False
            )

            content_similarity = (
                similarity_result
                .toarray()
                .flatten()
            )


            # =================================================
            # RATING SCORE
            # =================================================

            mean_rating = (
                recommendation_df["rate"]
                .mean()
            )

            rating_score = (
                recommendation_df["rate"]
                .fillna(mean_rating)
                / 5
            )


            # =================================================
            # POPULARITY SCORE
            # =================================================

            popularity = np.log1p(
                recommendation_df["votes"]
                .fillna(0)
            )

            max_popularity = (
                popularity.max()
            )

            if max_popularity > 0:

                popularity_score = (
                    popularity
                    / max_popularity
                )

            else:

                popularity_score = (
                    pd.Series(
                        0,
                        index=recommendation_df.index
                    )
                )


            # =================================================
            # PRICE SIMILARITY
            # =================================================

            price_column = (
                "approx_cost(for two people)"
            )

            median_price = (
                recommendation_df[
                    price_column
                ]
                .median()
            )

            prices = (
                recommendation_df[
                    price_column
                ]
                .fillna(median_price)
            )

            selected_price = (
                selected[price_column]
            )

            if pd.isna(selected_price):

                selected_price = median_price

            max_price = prices.max()

            if max_price > 0:

                price_difference = (
                    abs(
                        prices
                        - selected_price
                    )
                    / max_price
                )

                price_similarity = (
                    1
                    - price_difference
                )

            else:

                price_similarity = (
                    pd.Series(
                        1,
                        index=recommendation_df.index
                    )
                )


            # =================================================
            # HYBRID SCORE
            # =================================================

            recommendation_df[
                "recommendation_score"
            ] = (

                0.50
                * content_similarity

                +

                0.25
                * rating_score.values

                +

                0.15
                * popularity_score.values

                +

                0.10
                * price_similarity.values
            )


            # =================================================
            # REMOVE SELECTED RESTAURANT
            # =================================================

            results = recommendation_df[
                recommendation_df.index != index
            ].copy()


            # Remove same restaurant name

            results = results[
                results["name"].str.lower()
                != restaurant.lower()
            ]


            # Remove duplicate restaurants

            results = results.drop_duplicates(
                subset=[
                    "name",
                    "location"
                ]
            )


            # =================================================
            # SORT
            # =================================================

            results = results.sort_values(
                "recommendation_score",
                ascending=False
            )


            # =================================================
            # TOP 10
            # =================================================

            results = results.head(10)


            # =================================================
            # DISPLAY DATA
            # =================================================

            results = results[
                [
                    "name",
                    "location",
                    "cuisines",
                    "rest_type",
                    "rate",
                    "votes",
                    price_column,
                    "recommendation_score"
                ]
            ]

            results.columns = [
                "Restaurant",
                "Location",
                "Cuisine",
                "Restaurant Type",
                "Rating",
                "Votes",
                "Cost for Two",
                "Recommendation Score"
            ]

            results["Recommendation Score"] = (
                results["Recommendation Score"]
                .round(3)
            )


            # =================================================
            # DISPLAY RESULTS
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Recommended Restaurants'
                '</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # EXPLANATION
            # =================================================

            st.info(
                "Recommendation score: "
                "50% content similarity + "
                "25% rating + "
                "15% popularity + "
                "10% price similarity."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Food Delivery Analytics and Restaurant Recommendation System"
)