import pandas as pd # import panda library as pd for data manipulation
import matplotlib.pyplot as plt # import matplotlib as plt for data visualitzation
from matplotlib import style
import numpy as np # import nump as np
import seaborn as sns # seaborn is data visualization library build on matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""### Implement a Function Which return Original DataFrame, Transposed DataFrames"""

def transpose_file(filename: str):

    # Read the file into a pandas dataframe
    dataframe = pd.read_csv(filename)

    # Transpose the dataframe
    df_transposed = dataframe.transpose()

    # Populate the header of the transposed dataframe with the header information

    # silice the dataframe to get the year as columns
    df_transposed.columns = df_transposed.iloc[1]

    # As year is now columns so we don't need it as rows
    transposed_year = df_transposed[0:].drop('year')

    # silice the dataframe to get the country as columns
    df_transposed.columns = df_transposed.iloc[0]

    # As country is now columns so we don't need it as rows
    transposed_country = df_transposed[0:].drop('country')

    return dataframe, transposed_country, transposed_year

# Passing filename to Real Worldbank data function
# will return three dataframe:
# org dataframe, transposed country as columns and transposed year as column

org_df, df_by_country, df_by_year = transpose_file('worldbank_dataset.csv')

"""### Original DataFrame"""

# show the first 5 row
org_df.head(5)

"""### show the statistics of Original Data"""

org_df.describe() #describe method show the statistic of dataframe

"""### DataFrame In Which Countries are Columns"""

# show the first 5 row
df_by_country.head(5)

"""### DataFrame In Which Year are Columns"""

# show the first 5 row
df_by_year

org_df.columns

"""### Create DataFrame related to Renewable Energy Consumption
### For All the countries and years
"""

# we want to see countries renewable_energy_consumption over specfic years
# we need to filter our original data frame to get specific fields

# Filter the data for non-null values
renewable_energy_consumption = org_df[['country', 'year', 'renewable_energy_consumption']].dropna()

"""### Get Data to Specific Years from 1990 to 2020"""

import random

# Define the years for which you want to plot data
years_to_plot = [1990, 2000, 2010, 2015, 2020]

# Get a list of all named colors in Matplotlib
all_colors = list(mcolors.CSS4_COLORS.keys())

# Select a specific number of random colors from the list
num_colors_to_select = 10  # You can change this number as needed
selected_colors = random.sample(all_colors, num_colors_to_select)

countries = renewable_energy_consumption.country.unique()
countries

"""### Plot Barplot"""

# Create a figure and set its size
plt.figure(figsize=(15, 10))

# Set width of bars
barWidth = 0.1

for i, year in enumerate(years_to_plot):
    data = renewable_energy_consumption[renewable_energy_consumption['year'] == year]
    plt.bar(np.arange(data.shape[0]) + (0.2 * i), data['renewable_energy_consumption'], color=selected_colors[i], width=barWidth, label=str(year))

# Show legends, labels, and title
plt.legend()
plt.xlabel('Country', fontsize=15)
plt.title("Renewable Energy Consumption", fontsize=15)

# Add country names to the x-axis ticks
plt.xticks(np.arange(len(countries)) + 0.2, countries, fontsize=10, rotation=45)

# Show the plot
plt.show()

org_df.columns

"""### Get data of greenhouse_gas_emissions over the years"""

# we want to see countries greenhouse_gas_emissions over specfic years
# we need to filter our original data frame to get specific fields

# Filter the data for non-null values
greenhouse_gas_emissions = org_df[['country', 'year', 'greenhouse_gas_emissions']].dropna()

"""### Filter from specific year from 1990 to 2020"""

import random

# Define the years for which you want to plot data
years_to_plot = [1990, 2000, 2010, 2015, 2020]

# Get a list of all named colors in Matplotlib
all_colors = list(mcolors.CSS4_COLORS.keys())

# Select a specific number of random colors from the list
num_colors_to_select = 10  # You can change this number as needed
selected_colors = random.sample(all_colors, num_colors_to_select)

countries = greenhouse_gas_emissions.country.unique()

"""### PLOT barplot"""

# Create a figure and set its size
plt.figure(figsize=(15, 10))

# Set width of bars
barWidth = 0.1

for i, year in enumerate(years_to_plot):
    data = greenhouse_gas_emissions[greenhouse_gas_emissions['year'] == year]
    plt.bar(np.arange(data.shape[0]) + (0.2 * i), data['greenhouse_gas_emissions'], color=selected_colors[i], width=barWidth, label=str(year))

# Show legends, labels, and title
plt.legend()
plt.xlabel('Country', fontsize=15)
plt.title("Greenhouse Gas Emissions", fontsize=15)

# Add country names to the x-axis ticks
plt.xticks(np.arange(len(countries)) + 0.2, countries, fontsize=10, rotation=45)

# Show the plot
plt.show()

org_df.country.unique()

"""### Making a DataFrame related to Zimbabwe"""

# making dataframe of Zimbabwe data from the original dataframe
zim = org_df[org_df['country'] == 'Zimbabwe']

"""### Implement a Function which removes Null values and return clean data"""

def remove_null_values(feature):
    return np.array(feature.dropna())

"""### For the Features Present In Zimbabwe DataFrame remove the null values
### Print Each Features Size
"""

org_df.columns

# List of columns to extract
columns_of_interest = ['co2_emissions', 'greenhouse_gas_emissions',
       'cereal_yield', 'population_growth', 'GDP', 'fresh_water',
       'urban_population', 'renewable_electricity', 'total_population',
       'renewable_energy_consumption']

# Dictionary to store feature data after removing null values
feature_data = {}

# Loop through each column to extract and clean the data
for column in columns_of_interest:
    feature_data[column] = remove_null_values(zim[[column]])
    print(f'{column} Length = {len(feature_data[column])}')

# Create data_sources dictionary dynamically
data_sources = {column: feature_data[column] for column in columns_of_interest}

# Determine the number of rows to include
num_rows = 26

# Create the DataFrame using dictionary comprehension
zim_clean_data = pd.DataFrame({
    key: [data_sources[key][x][0] for x in range(num_rows)] for key in data_sources
})

import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix
correlation_matrix = zim_clean_data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Correlation Heatmap for Zimbabwe')
plt.show()

correlation_matrix

org_df.country.unique()

"""### Making a DataFrame related to Togo"""

# making dataframe of Togo data from the original dataframe
Togo = org_df[org_df['country'] == 'Togo']

"""### For the Features Present In DataFrame remove the null values
### Print Each Features Size
"""

Togo.columns

# List of columns to extract
columns_of_interest = ['co2_emissions', 'greenhouse_gas_emissions',
       'cereal_yield', 'population_growth', 'GDP', 'fresh_water',
       'urban_population', 'renewable_electricity', 'total_population',
       'renewable_energy_consumption']

# Dictionary to store feature data after removing null values
feature_data = {}

# Loop through each column to extract and clean the data
for column in columns_of_interest:
    feature_data[column] = remove_null_values(Togo[[column]])
    print(f'{column} Length = {len(feature_data[column])}')

# Create data_sources dictionary dynamically
data_sources = {column: feature_data[column] for column in columns_of_interest}

# Determine the number of rows to include
num_rows = 26

# Create the DataFrame using dictionary comprehension
ton_clean_data = pd.DataFrame({
    key: [data_sources[key][x][0] for x in range(num_rows)] for key in data_sources
})

import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix
correlation_matrix = ton_clean_data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap for Togo')
plt.show()

correlation_matrix

org_df.country.unique()

"""### Making a DataFrame related to Madagascar"""

# making dataframe of Madagascar data from the original dataframe
mad = org_df[org_df['country'] == 'Madagascar']

"""### For the Features Present In DataFrame remove the null values
### Print Each Features Size
"""

mad.columns

# List of columns to extract
columns_of_interest = ['co2_emissions', 'greenhouse_gas_emissions',
       'cereal_yield', 'population_growth', 'GDP', 'fresh_water',
       'urban_population', 'renewable_electricity', 'total_population',
       'renewable_energy_consumption']

# Dictionary to store feature data after removing null values
feature_data = {}

# Loop through each column to extract and clean the data
for column in columns_of_interest:
    feature_data[column] = remove_null_values(mad[[column]])
    print(f'{column} Length = {len(feature_data[column])}')

# Create data_sources dictionary dynamically
data_sources = {column: feature_data[column] for column in columns_of_interest}

# Determine the number of rows to include
num_rows = 26

# Create the DataFrame using dictionary comprehension
mad_clean_data = pd.DataFrame({
    key: [data_sources[key][x][0] for x in range(num_rows)] for key in data_sources
})

import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix
correlation_matrix = mad_clean_data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt=".2f")
plt.title('Correlation Heatmap for Madagascar')
plt.show()

correlation_matrix

org_df.columns

"""### Get the Year, Country Data Related to urban_population"""

# we want to see countries urban_population over the years
# we need to filter our original data frame to get specific fields
urban_population = org_df[['country','year','urban_population']]

# drop the null values present in the dataset
urban_population  = urban_population.dropna()

"""### Filter the Data For All the Countries"""

# Define countries of interest
countries = org_df.country.unique()
countries

"""### Line Plot of Urban Population"""

# Set fig size
plt.figure(figsize=(15, 10))

# Loop through countries and plot population_growth over the years
for country in countries:
    country_data = urban_population[urban_population['country'] == country]
    plt.plot(country_data['year'], country_data['urban_population'], label=country)

# Set X-axis label and title
plt.xlabel('Year', fontweight='bold')
plt.title("Urban Population")

# Show legends and plot
plt.legend(bbox_to_anchor=(0.89, 0.7), shadow=True)
plt.show()

"""### Get the Year, Country Data Related to renewable_electricity"""

# we want to see countries renewable_electricity over the years
# we need to filter our original data frame to get specific fields
renewable_electricity = org_df[['country','year','renewable_electricity']]

# drop the null values present in the dataset
renewable_electricity  = renewable_electricity.dropna()

"""### Filter the Data For All the Countries"""

# Define countries of interest
countries = org_df.country.unique()
countries

"""### Line Plot of renewable_electricity"""

# Set fig size
plt.figure(figsize=(15, 10))

# Loop through countries and plot renewable_electricity over the years
for country in countries:
    country_data = renewable_electricity[renewable_electricity['country'] == country]
    plt.plot(country_data['year'], country_data['renewable_electricity'], label=country)

# Set X-axis label and title
plt.xlabel('Year', fontweight='bold')
plt.title("Renewable Electricity")

# Show legends and plot
plt.legend(bbox_to_anchor=(0.2, 0.65), shadow=True)
plt.show()

org_df.columns

# we want to see countries co2_emissions over the years
co2_emissions = org_df[['country','year','co2_emissions']]

# drop the null values present in the dataset
co2_emissions = co2_emissions.dropna()

### Filter from specific year from 1990 to 2015
# filter data related to 1990
co2_emissions_1990 = co2_emissions[co2_emissions['year'] == 1990]

# filter data related to 2010
co2_emissions_2010 = co2_emissions[co2_emissions['year'] == 2010]

# filter data related to 2020
co2_emissions_2020 = co2_emissions[co2_emissions['year'] == 2020]

co2_emissions_1990

co2_emissions_2010

co2_emissions_2020