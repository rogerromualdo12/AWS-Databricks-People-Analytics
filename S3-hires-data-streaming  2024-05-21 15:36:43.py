# Databricks notebook source
# Define the S3 path
s3_path = "s3a://hires-2024/"

# List files in the S3 bucket
try:
    files = dbutils.fs.ls(s3_path)
    print("Access to S3 bucket successful. Contents:")
    for file in files:
        print(file.path)
except Exception as e:
    print("Access to S3 bucket failed:", e)


# COMMAND ----------

# MAGIC %md
# MAGIC __Generating the pipeline__

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DateType, StringType


spark = SparkSession.builder.appName("hires-visualization").getOrCreate()

s3_path = "s3a://hires-2024/"
schema_location = "s3a://databricks-workspace-stack-42d6c-bucket/unity-catalog-hires/"
chekpoint_location = "s3a://databricks-workspace-stack-42d6c-bucket/unity-catalog-hires-checkpoint/"

schema = StructType([
    StructField("First Name", StringType(), True),
    StructField("Last Name", StringType(), True),
    StructField("Email", StringType(), True),
    StructField("Application Date", DateType(), True),
    StructField("Country", StringType(), True),
    StructField("YOE", IntegerType(), True),
    StructField("Seniority", StringType(), True),
    StructField("Technology", StringType(), True),
    StructField("Code Challenge Score", IntegerType(), True),
    StructField("Technical Interview Score", IntegerType(), True)
])

df = (spark.readStream
      .format("cloudFiles")
      .option("cloudFiles.format","csv")
      .option("header","true")
      .option("delimiter",";")
      .schema(schema)
      .option("cloudFiles.schemaLocation",schema_location)
      .load(s3_path))

# COMMAND ----------

# MAGIC %md
# MAGIC __Data Filtering__

# COMMAND ----------

#Filtering and Aggregations
df_hires = df.filter((df["Code Challenge Score"] >= 7) & (df["Technical Interview Score"] >= 7))
df_hires_by_technology = df_hires.groupBy("Technology").agg(F.count("Technology").alias("technology_count"))

# COMMAND ----------

# MAGIC %md
# MAGIC __Streaming aggregated tables__

# COMMAND ----------


query_hires_by_technology = (df_hires_by_technology.writeStream
                          .format("memory")
                          .queryName("hires_by_technology")
                          .outputMode("complete")
                          .start())


# COMMAND ----------

# MAGIC %md
# MAGIC __Streaming the entire data: medallion architecture approach__

# COMMAND ----------

query = (df_hires.writeStream
                          .format("memory")
                          .queryName("hires")
                          .outputMode("append")
                          .start())

# COMMAND ----------

import time
time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC __Generating the queries__

# COMMAND ----------

hires_by_technology = spark.sql("SELECT * FROM hires_by_technology ORDER BY technology_count DESC LIMIT 10").toPandas()

# COMMAND ----------

hires_by_country_over_the_years = spark.sql("SELECT Country, YEAR(`Application Date`) AS APPLICATION_YEAR, COUNT(*) AS Cantidad FROM hires GROUP BY Country, YEAR(`Application Date`) ORDER BY 1 ASC,2 ASC LIMIT 100").toPandas()
hires_by_seniority = spark.sql("SELECT Seniority, COUNT(*) as Seniority_Count FROM hires GROUP BY Seniority ORDER BY Seniority_Count DESC LIMIT 100").toPandas()
hires_by_year = spark.sql("SELECT YEAR(`Application Date`) AS APPLICATION_YEAR, COUNT(*) AS Hires_Count FROM hires GROUP BY APPLICATION_YEAR ORDER BY APPLICATION_YEAR ASC LIMIT 10").toPandas()

# COMMAND ----------

# MAGIC %md __Input aggregated tables__

# COMMAND ----------

# Hires by technology
hires_by_technology.head()

# COMMAND ----------

# Hires by country over the years
pivot_df = hires_by_country_over_the_years.pivot(index='Country', columns = 'APPLICATION_YEAR', values='Cantidad')
pivot_df.head()

# COMMAND ----------

# Hires by Seniority
hires_by_seniority.head()

# COMMAND ----------

#Hires by year
hires_by_year.head()

# COMMAND ----------

# MAGIC %md
# MAGIC _Hires by technology_

# COMMAND ----------

import matplotlib.pyplot as plt


plt.figure(figsize=(100,6))
plt.bar(hires_by_technology['Technology'],hires_by_technology['technology_count'])
plt.xlabel('Technology', fontsize=30)
plt.ylabel('technology_count', fontsize=30)
plt.tick_params(axis='both', labelsize=25)
plt.title('Hires by technology', fontsize=50)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC _Hires by Country over the years_

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Pivot data to the required format
pivot_df = hires_by_country_over_the_years.pivot(index='Country', columns='APPLICATION_YEAR', values='Cantidad')

# Plotting bars for each year
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate the width of each bar and positions
width = 0.25  # width of each bar, can adjust as necessary
num_years = len(pivot_df.columns)
indices = np.arange(len(pivot_df)) * (width * num_years + 0.5)  # increasing the spacing by 0.5

for i, year in enumerate(sorted(pivot_df.columns)):
    # Calculate the position for each bar
    pos = indices + i * width
    ax.bar(pos, pivot_df[year], width=width, label=year)

# Adding labels, title, and legend
ax.set_xlabel('Country')
ax.set_ylabel('Cantidad')
ax.set_title('Quantity by Country and Year')
ax.set_xticks(indices + width * (num_years - 1) / 2)
ax.set_xticklabels(pivot_df.index)
ax.legend(title='APPLICATION_YEAR', loc='upper left')

# Optional: add vertical spacing lines between groups for better visual separation
for idx in indices[1:]:
    ax.axvline(x=idx - width, color='grey', linestyle='--')

# Show the plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC _Hires by Seniority_

# COMMAND ----------

plt.figure(figsize=(100,20))
plt.bar(hires_by_seniority['Seniority'],hires_by_seniority['Seniority_Count'])
plt.xlabel('Seniority', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.tick_params(axis='both', labelsize=25)
plt.title('Hires by Seniority', fontsize=50)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC _Hires by year_

# COMMAND ----------

plt.figure(figsize=(100,20))
plt.barh(hires_by_year['APPLICATION_YEAR'],hires_by_year['Hires_Count'])
plt.xlabel('APPLICATION_YEAR', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.tick_params(axis='both', labelsize=25)
plt.title('Hires by year', fontsize=50)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC __End of file__
