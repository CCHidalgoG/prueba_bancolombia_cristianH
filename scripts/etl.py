import logging

import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType

from scripts.settings import DOMAIN_DESCRIPTIONS

# crear spark session
spark = ps.sql.SparkSession.builder \
    .appName("ETL") \
    .getOrCreate()


def load_data_to_df(data_dir):
    # Carga los datos en un solo dataframe
    df = spark.read.option("header", "true").csv(data_dir)
    return df


def preprocess_data(df):
    # Get column names from settings
    columns = DOMAIN_DESCRIPTIONS["fraude_bancos"]["columns"]

    # Numeric columns based on their descriptions
    numeric_columns = ["ing_mes", "egresos_mes", "trn_monto"]

    # Preprocess each column
    for col in df.columns:
        # Replace "None" strings with NA
        df = df.withColumn(col, F.when(F.col(col) == "None", None).otherwise(F.col(col)))

        # Convert numeric columns to double
        if col in numeric_columns:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
        else:
            # Ensure other columns are strings
            df = df.withColumn(col, F.col(col).cast(StringType()))

        # Clean whitespace from string columns
        if col not in numeric_columns:
            df = df.withColumn(col, F.trim(F.col(col)))

        # conver all non-numeric columns to lower case
        if col not in numeric_columns:
            df = df.withColumn(col, F.lower(F.col(col)))

        # Remplace non-numeric column proportion labels lower 5% with "Other"
        if col not in numeric_columns + ["cli", "profn", "ocup", "nivel_riesgo_ciiu"]:
            logging.info(f"Processing column {col}")  # Fixed this line
            # Compute proportion of labels in each column
            label_counts = df.groupBy(col).count()
            total = df.count()
            label_counts = label_counts.withColumn("percentage", F.col("count") / total)
             # print(label_counts.show())
            # Get labels with proportion lower than 5%
            labels_to_replace = label_counts.filter(F.col("percentage") < 0.05).select(col).distinct().rdd.flatMap(lambda x: x).collect()
            # Replace labels
            df = df.withColumn(col, F.when(F.col(col).isin(labels_to_replace), "Other").otherwise(F.col(col)))

    return df