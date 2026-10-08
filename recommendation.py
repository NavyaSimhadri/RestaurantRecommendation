import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("zomato.csv")

print("Dataset loaded successfully!")
print("Number of restaurants:", len(df))


# ==========================================
# 2. CLEAN DATA
# ==========================================

# Fill missing values
df["name"] = df["name"].fillna("")
df["online_order"] = df["online_order"].fillna("")
df["book_table"] = df["book_table"].fillna("")
df["listed_in(type)"] = df["listed_in(type)"].fillna("")


# ==========================================
# 3. CREATE FEATURES
# ==========================================

# Combine available categorical information
df["features"] = (
    df["online_order"].astype(str) + " " +
    df["book_table"].astype(str) + " " +
    df["listed_in(type)"].astype(str)
)


# ==========================================
# 4. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    df["features"]
)


# ==========================================
# 5. COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(tfidf_matrix)


# ==========================================
# 6. RECOMMENDATION FUNCTION
# ==========================================

def recommend_restaurants(restaurant_name):

    # Find restaurant
    matches = df[
        df["name"].str.lower() ==
        restaurant_name.lower()
    ]

    if matches.empty:
        print("\nRestaurant not found.")
        print("Please enter a restaurant name from the dataset.")
        return

    index = matches.index[0]

    # Get similarity scores
    scores = list(
        enumerate(similarity[index])
    )

    # Sort from highest similarity
    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\n===================================")
    print("Recommended Restaurants")
    print("===================================\n")

    count = 0

    for i, score in scores:

        # Don't recommend the same restaurant
        if i == index:
            continue

        print(
            f"Restaurant : {df.iloc[i]['name']}"
        )

        print(
            f"Online Order: {df.iloc[i]['online_order']}"
        )

        print(
            f"Table Booking: {df.iloc[i]['book_table']}"
        )

        print(
            f"Type: {df.iloc[i]['listed_in(type)']}"
        )

        print(
            f"Rating: {df.iloc[i]['rate']}"
        )

        print(
            f"Similarity: {score:.2f}"
        )

        print("-----------------------------------")

        count += 1

        if count == 5:
            break


# ==========================================
# 7. RUN PROGRAM
# ==========================================

restaurant = input(
    "\nEnter restaurant name: "
)

recommend_restaurants(restaurant)