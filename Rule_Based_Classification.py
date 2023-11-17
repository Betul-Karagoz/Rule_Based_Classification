#############################################
# Calculating Return on Leads with Rule-Based Classification
#############################################

#############################################
# Business Problem
#############################################
# A game company uses some of its customers' characteristics to create new level-based customer personas
# create segments according to these new customer definitions and according to these segments, new customers who may come to the company
# wants to estimate how much they can earn on average.

# For example: It is desired to determine how much a 25-year-old male IOS user from Turkey can earn on average.

#############################################
# Data Set Story
#############################################
# Persona.csv dataset contains the prices of products sold by an international gaming company and some of the users who purchased these products
# contains demographic information. The dataset consists of the records generated in each sales transaction. This means that the table
# is not singularized. In other words, a user with certain demographic characteristics may have made more than one purchase.

# Price: Customer's spending amount
# Source: The type of device the client is connected to
# Sex Gender of the client
# Country: Customer's country
# Age: Age of the customer

#############################################
# PROJECT TASKS
#############################################

#############################################
# TASK 1: Answer the following questions.
#############################################

################
# Question 1: Read the persona.csv file and show the general information about the dataset.

import seaborn as sns
import pandas as pd
pd.set_option("display.max_rows", None)
df=pd.read_csv("DataScience/datasets/persona.csv")
df.head()
df.tail()
df.shape
df.columns
df.index
df.describe().T
df.isnull().values.any()

################
# Question 2: How many unique SOURCEs are there? What is their frequency?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

################
# Question 3: How many unique PRICEs are there?

df["PRICE"].nunique()

################
# Question 4: How many sales were made from which PRICE?

df["PRICE"].value_counts()

################
# Question 5: How many sales were from which country?

df["COUNTRY"].value_counts()

################
# Question 6: How much was earned from sales by country?

df.groupby("COUNTRY").agg({"PRICE":"sum"})

###############
# Question 7: What is the number of sales by SOURCE types?

df["SOURCE"].unique()
df["SOURCE"].value_counts()
df.groupby("SOURCE").agg({"PRICE":["sum","count"]})

################
# Question 8: What is the average PRICE by country?

df.groupby("COUNTRY").agg({"PRICE":"mean"})
df.groupby("COUNTRY").agg({"PRICE":["mean"]})

################
# Question 9: What is the average PRICE by SOURCE?

df.groupby("SOURCE").agg({"PRICE":"mean"})

################
# Question 10: What are the PRICE averages by COUNTRY-SOURCE?

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})

#############################################
# TASK 2: What are the average earnings by COUNTRY, SOURCE, SEX, AGE?
#############################################

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})

#############################################
# TASK 3: Sort the output by PRICE.
#############################################

agg_df=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values(by="PRICE",ascending=False)
agg_df.head()

#############################################
# TASK 4: Convert the names in the index to variable names.
#############################################

agg_df = agg_df.reset_index()
agg_df.head()

#############################################
# TASK 5: Convert AGE to a categorical variable and add it to agg_df.
#############################################

# Let's specify where to divide the AGE variable:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Let's express what the nomenclature will be for the dividing points:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# split age
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

#############################################
# TASK 6: Define new level based customers and add them as variables to the data set.
#############################################

agg_df["CUSTOMERS_LEVEL_BASED"]=agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].agg(lambda x: "_".join(x).upper(), axis=1)

##### An other method

agg_df["CUSTOMERS_LEVEL_BASED"] = ['_'.join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]

#############################################
# TASK 7: Segment new customers (USA_ANDROID_MALE_0_18)
#############################################

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})

#############################################
# TASK 8: Classify new customers and estimate how much revenue they can generate.
#############################################
# Which segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is she expected to earn on average?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]

# Which segment and how much income would a 35-year-old French woman using IOS be expected to earn on average?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]