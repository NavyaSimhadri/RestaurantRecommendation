from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    count,
    desc,
    split,
    explode,
    trim,
    regexp_replace,
    when
)
import os


# ============================================================
# 1. START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("FoodDeliveryAnalytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading Zomato dataset...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("quote", '"')
    .option("escape", '"')
    .option("mode", "PERMISSIVE")
    .csv("zomato.csv")
)

print("\nDataset columns:")
print(df.columns)

print("\nNumber of rows:", df.count())

print("\nSample data:")

df.select(
    "name",
    "location",
    "rate",
    "votes",
    "rest_type",
    "cuisines",
    "approx_cost(for two people)"
).show(5, truncate=False)


# ============================================================
# 3. CLEAN RATING
# ============================================================

df = df.withColumn(
    "rate_clean",
    regexp_replace(col("rate"), "/5", "")
)

df = df.withColumn(
    "rate_clean",
    when(
        col("rate_clean").rlike(r"^[0-9]+(\.[0-9]+)?$"),
        col("rate_clean").cast("double")
    ).otherwise(None)
)


# ============================================================
# 4. CLEAN COST
# ============================================================

df = df.withColumn(
    "cost_clean",
    regexp_replace(
        col("approx_cost(for two people)"),
        ",",
        ""
    )
)

df = df.withColumn(
    "cost_clean",
    when(
        col("cost_clean").rlike(r"^[0-9]+(\.[0-9]+)?$"),
        col("cost_clean").cast("double")
    ).otherwise(None)
)


# ============================================================
# 5. BASIC STATISTICS
# ============================================================

print("\n================ BASIC STATISTICS ================")

basic_stats = df.select(
    avg("rate_clean").alias("average_rating"),
    avg("cost_clean").alias("average_cost"),
    avg("votes").alias("average_votes")
)

basic_stats.show()


# ============================================================
# 6. TOP LOCATIONS
# ============================================================

print("\n================ TOP LOCATIONS ================")

top_locations = (
    df
    .filter(col("location").isNotNull())
    .groupBy("location")
    .agg(
        count("*").alias("restaurant_count")
    )
    .orderBy(desc("restaurant_count"))
    .limit(10)
)

top_locations.show(truncate=False)


# ============================================================
# 7. LOCATION-WISE AVERAGE RATING
# ============================================================

print("\n================ LOCATION RATINGS ================")

location_ratings = (
    df
    .filter(col("rate_clean").isNotNull())
    .filter(col("location").isNotNull())
    .groupBy("location")
    .agg(
        avg("rate_clean").alias("average_rating"),
        count("*").alias("restaurant_count")
    )
    .filter(col("restaurant_count") >= 10)
    .orderBy(desc("average_rating"))
    .limit(10)
)

location_ratings.show(truncate=False)


# ============================================================
# 8. ONLINE ORDER ANALYSIS
# ============================================================

print("\n================ ONLINE ORDER ================")

online_order_analysis = (
    df
    .filter(col("online_order").isNotNull())
    .groupBy("online_order")
    .agg(
        count("*").alias("restaurant_count")
    )
    .orderBy(desc("restaurant_count"))
)

online_order_analysis.show()


# ============================================================
# 9. TABLE BOOKING ANALYSIS
# ============================================================

print("\n================ TABLE BOOKING ================")

table_booking_analysis = (
    df
    .filter(col("book_table").isNotNull())
    .groupBy("book_table")
    .agg(
        count("*").alias("restaurant_count")
    )
    .orderBy(desc("restaurant_count"))
)

table_booking_analysis.show()


# ============================================================
# 10. TOP CUISINES
# ============================================================

print("\n================ TOP CUISINES ================")

cuisine_df = (
    df
    .filter(col("cuisines").isNotNull())
    .select(
        explode(
            split(col("cuisines"), ",")
        ).alias("cuisine")
    )
)

top_cuisines = (
    cuisine_df
    .withColumn(
        "cuisine",
        trim(col("cuisine"))
    )
    .filter(col("cuisine") != "")
    .groupBy("cuisine")
    .agg(
        count("*").alias("restaurant_count")
    )
    .orderBy(desc("restaurant_count"))
    .limit(15)
)

top_cuisines.show(truncate=False)


# ============================================================
# 11. RESTAURANT TYPES
# ============================================================

print("\n================ RESTAURANT TYPES ================")

restaurant_types = (
    df
    .filter(col("rest_type").isNotNull())
    .groupBy("rest_type")
    .agg(
        count("*").alias("restaurant_count")
    )
    .orderBy(desc("restaurant_count"))
    .limit(15)
)

restaurant_types.show(truncate=False)


# ============================================================
# 12. POPULAR RESTAURANTS
# ============================================================

print("\n================ POPULAR RESTAURANTS ================")

popular_restaurants = (
    df
    .filter(col("name").isNotNull())
    .groupBy(
        "name",
        "location"
    )
    .agg(
        avg("votes").alias("average_votes"),
        avg("rate_clean").alias("average_rating")
    )
    .orderBy(desc("average_votes"))
    .limit(10)
)

popular_restaurants.show(truncate=False)


# ============================================================
# 13. HIGH-RATED RESTAURANTS
# ============================================================

print("\n================ HIGH RATED RESTAURANTS ================")

high_rated_restaurants = (
    df
    .filter(col("rate_clean").isNotNull())
    .filter(col("name").isNotNull())
    .groupBy(
        "name",
        "location"
    )
    .agg(
        avg("rate_clean").alias("average_rating"),
        avg("votes").alias("average_votes")
    )
    .orderBy(
        desc("average_rating"),
        desc("average_votes")
    )
    .limit(10)
)

high_rated_restaurants.show(truncate=False)


# ============================================================
# 14. CREATE OUTPUT DIRECTORY
# ============================================================

print("\n================ SAVING RESULTS ================")

os.makedirs(
    "spark_output",
    exist_ok=True
)


# ============================================================
# 15. SAVE FUNCTION
# ============================================================

def save_result(spark_df, filename):

    # Spark performs the actual analysis.
    # Only the small aggregated result is converted
    # to Pandas for saving on Windows without Hadoop.

    pandas_df = spark_df.toPandas()

    output_path = os.path.join(
        "spark_output",
        filename
    )

    pandas_df.to_csv(
        output_path,
        index=False
    )

    print("Saved:", output_path)


# ============================================================
# 16. SAVE ALL RESULTS
# ============================================================

save_result(
    top_locations,
    "top_locations.csv"
)

save_result(
    top_cuisines,
    "top_cuisines.csv"
)

save_result(
    location_ratings,
    "location_ratings.csv"
)

save_result(
    online_order_analysis,
    "online_order_analysis.csv"
)

save_result(
    table_booking_analysis,
    "table_booking_analysis.csv"
)

save_result(
    restaurant_types,
    "restaurant_types.csv"
)

save_result(
    popular_restaurants,
    "popular_restaurants.csv"
)

save_result(
    high_rated_restaurants,
    "high_rated_restaurants.csv"
)


# ============================================================
# 17. FINISH
# ============================================================

print("\n================================================")
print("SPARK ANALYSIS COMPLETED SUCCESSFULLY")
print("================================================")

spark.stop()