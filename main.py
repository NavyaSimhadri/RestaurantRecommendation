import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("zomato.csv")

print("\n===== DATASET INFORMATION =====")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. DATA CLEANING
# ==========================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where restaurant name is missing
if "name" in df.columns:
    df = df.dropna(subset=["name"])


# ==========================================
# 3. BASIC ANALYSIS
# ==========================================

print("\n===== RESTAURANT ANALYSIS =====")

# Restaurant types
if "listed_in(type)" in df.columns:
    print("\nRestaurant Types:")
    print(df["listed_in(type)"].value_counts())


# Online ordering
if "online_order" in df.columns:
    print("\nOnline Order:")
    print(df["online_order"].value_counts())


# Table booking
if "book_table" in df.columns:
    print("\nTable Booking:")
    print(df["book_table"].value_counts())


# ==========================================
# 4. TOP RESTAURANTS BY VOTES
# ==========================================

if "votes" in df.columns:

    print("\n===== MOST POPULAR RESTAURANTS =====")

    popular = df.sort_values(
        "votes",
        ascending=False
    ).head(10)

    print(
        popular[["name", "votes"]]
        .to_string(index=False)
    )


# ==========================================
# 5. RATING ANALYSIS
# ==========================================

if "rate" in df.columns:

    # Convert rating into numbers
    df["rate"] = (
        df["rate"]
        .astype(str)
        .str.replace("/5", "", regex=False)
    )

    df["rate"] = pd.to_numeric(
        df["rate"],
        errors="coerce"
    )

    print("\n===== RATING STATISTICS =====")

    print("Mean:", df["rate"].mean())
    print("Median:", df["rate"].median())
    print("Variance:", df["rate"].var())
    print("Standard Deviation:", df["rate"].std())


# ==========================================
# 6. COST ANALYSIS
# ==========================================

if "approx_cost(for two people)" in df.columns:

    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df["approx_cost(for two people)"] = pd.to_numeric(
        df["approx_cost(for two people)"],
        errors="coerce"
    )

    print("\n===== COST ANALYSIS =====")

    print(
        "Average cost for two:",
        df["approx_cost(for two people)"].mean()
    )


# ==========================================
# 7. VISUALIZATION
# ==========================================

if "rate" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["rate"].dropna(),
        bins=10
    )

    plt.xlabel("Restaurant Rating")
    plt.ylabel("Number of Restaurants")
    plt.title("Restaurant Rating Distribution")

    plt.show()


# ==========================================
# 8. TOP RESTAURANTS VISUALIZATION
# ==========================================

if "votes" in df.columns:

    top10 = df.sort_values(
        "votes",
        ascending=False
    ).head(10)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=top10,
        x="votes",
        y="name"
    )

    plt.title("Top 10 Most Popular Restaurants")
    plt.xlabel("Number of Votes")
    plt.ylabel("Restaurant")

    plt.show()


print("\n===== ANALYSIS COMPLETED =====")