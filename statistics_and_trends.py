"""
Climate Change Indicators Analysis

Student Name: Mahendar Reddy Kongolla
Student ID: 24150793

This program analyses global climate change indicators from the World
Development Indicators dataset, focusing on greenhouse gas emissions
and energy use from 1990-2023. It performs descriptive and statistical
analysis using Python, computing the four main statistical moments and
producing relational, categorical, and statistical plots as required for the
statistics and trends assignment.
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Creates a line plot for GHG emissions (% change).

    This plot serves as the relational plot, showing the xy relationship
    between Year and GHG Emissions.
    """

    relational_indicator = ('Total greenhouse gas emissions excluding LULUCF '
                            '(% change from 1990)')
    relational_df = df[['Year', 'Country Name', relational_indicator]].dropna()
    relational_df.rename(columns={relational_indicator: 'Value'}, inplace=True)

    # Create figure and axes object
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=relational_df, x='Year', y='Value',
                 hue='Country Name', marker="o", ax=ax)

    # Set plot attributes
    ax.set_title('GHG Emissions (% Change from 1990) Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('GHG Emission % Change', fontsize=12)
    ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close(fig)
    return


def plot_categorical_plot(df):
    """Creates a bar chart comparing average energy use across countries.

    This serves as the categorical plot, comparing the mean energy
    use for each country category.
    """
    categorical_indicator = 'Energy use (kg of oil equivalent per capita) '
    categorical_df = df[['Country Name', categorical_indicator]].dropna()
    categorical_df.rename(columns={
        categorical_indicator: 'Value'
    }, inplace=True)

    avg_energy = (categorical_df.groupby('Country Name')['Value']
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index())
    # Create figure and axes object
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=avg_energy, x='Country Name', y='Value',
                palette='plasma', ax=ax)

    # Set plot attributes
    ax.set_title('Average Energy Use (kg of oil equivalent per capita)',
                 fontsize=16)
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Energy Use (kg/capita)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Save and close the plot
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close(fig)
    return


def plot_statistical_plot(df):
    """Creates a correlation heatmap showing relationships between key
    development indicators.
    This serves as the statistical plot,communicating statistical correlations.
    """
    cols_for_corr = [
        'Total greenhouse gas emissions excluding LULUCF (% change from 1990)',
        'Access to electricity (% of population)',
        'Fossil fuel energy consumption (% of total) ',
        'Energy use (kg of oil equivalent per capita) ',
        'GDP per unit of energy use (PPP $ per kg of oil equivalent) ',
        'Renewable energy consumption (% of total final energy consumption)'
    ]

    # Aggregate data by country and calculate the mean for each indicator
    df_agg = df.groupby('Country Name')[cols_for_corr].mean().dropna()

    # Simplify column names for better display
    df_agg.columns = [
        'GHG Emissions (% Change)',
        'Electricity Access (%)',
        'Fossil Fuel (%)',
        'Energy Use (kg/capita)',
        'GDP per Energy ($)',
        'Renewable Energy (%)',
    ]
    # Calculate the correlation matrix
    corr_matrix = df_agg.corr()

    # Create figure and axes object
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='plasma',
                center=0, fmt='.2f', linewidths=.5, ax=ax)

    # Set plot attributes
    ax.set_title('Correlation Matrix of Mean Development Indicators',
                 fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close(fig)
    return


def statistical_analysis(df, col: str):
    """
    Calculates the four main statistical moments for the given column,
    grouped by country.Returns a DataFrame containing mean, variance,
    skewness, and kurtosis.
    """
    grouped = df.groupby('Country Name')[col]
    stats_df = grouped.agg(['mean', 'var']).rename(columns={'var': 'variance'})
    stats_df['skewness'] = grouped.apply(lambda x: ss.skew(x.dropna()))
    stats_df['kurtosis'] = grouped.apply(lambda x: ss.kurtosis(x.dropna()))
    return stats_df


def preprocessing(df):
    """
    Preprocesses the data: converts columns to numeric and filters countries.
    Also provides a quick overview using .info() and a null value check.
    """
    print("--- Initial Data Overview ---")
    print(df.info())
    print("\n--- Null Value Check ---")
    null_summary = df.isnull().sum()
    print(null_summary[null_summary > 0])
    cols_to_numeric = [
        'Total greenhouse gas emissions excluding LULUCF (% change from 1990)',
        'Access to electricity (% of population)',
        'Fossil fuel energy consumption (% of total) ',
        'Energy use (kg of oil equivalent per capita) ',
        'GDP per unit of energy use (PPP $ per kg of oil equivalent) ',
        'Renewable energy consumption (% of total final energy consumption)'
    ]
    # Convert columns to numeric, coercing errors to NaN
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to keep only countries with at least 10 years of data
    country_counts = df['Country Name'].value_counts()
    valid_countries = country_counts[country_counts >= 10].index
    df_filtered = df[df['Country Name'].isin(valid_countries)].copy()

    print(f'''\nData filtered to {len(df_filtered['Country Name'].unique())}
          countries with sufficient data.''')
    return df_filtered


def writing(moments_df, col):
    """
    Prints the calculated statistical moments DataFrame in a formatted table
    and adds a summary interpretation of the overall distribution shape.
    """
    print(f'\n--- Statistical Moments for {col} ---')
    # Use to_string() for clean table formatting that matches desired output
    print(moments_df.to_string(float_format='%.6f'))

    # Calculate overall average moments for a summary description
    avg_skew = moments_df['skewness'].mean()
    avg_kurtosis = moments_df['kurtosis'].mean()

    # Interpret Skewness
    if avg_skew < -0.5:
        skew_desc = "left-skewed"
    elif avg_skew > 0.5:
        skew_desc = "right-skewed"
    else:
        skew_desc = "not skewed"

    # Interpret Kurtosis
    if avg_kurtosis < -0.5:
        kurt_desc = "platykurtic"
    elif avg_kurtosis > 0.5:
        kurt_desc = "leptokurtic"
    else:
        kurt_desc = "mesokurtic"

    print(f'Overall, the data was {skew_desc} and {kurt_desc}.')
    return


def main():
    """Main function to run the analysis pipeline.
    """
    # Load and preprocess the data
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: 'data.csv' not found. ")
        return

    df = preprocessing(df)

    # --- Define Columns for Analysis ---
    rel_col = ('Total greenhouse gas emissions excluding LULUCF '
               '(% change from 1990)')
    cat_col = 'Energy use (kg of oil equivalent per capita) '
    # --- Generate All Plots ---
    print("\n--- Generating Plots ---")
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    print('''Plots saved as relational_plot.png, categorical_plot.png,
          statistical_plot.png''')

    # --- Perform and Write Statistical Analysis ---
    # Calculate moments for both indicators
    moments_ghg = statistical_analysis(df, rel_col)
    moments_energy = statistical_analysis(df, cat_col)
    # the full moment tables with qualitative summary for report
    writing(moments_ghg, "GHG Emissions (% Change)")
    writing(moments_energy, "Energy Use (kg/capita)")


if __name__ == '__main__':
    main()
