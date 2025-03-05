import pandas as pd

df = pd.read_csv('../00_data/LLM_evaluation/llm_prompt_evaluation_results.csv')
print(df.head())

# Calculate mean and standard deviation of MRR column
mrr_mean = df['MRR'].mean()
mrr_std = df['MRR'].std()
print(f'Mean MRR: {mrr_mean}')
print(f'Standard deviation of MRR: {mrr_std}')

# Calculate missing values 
missing_values = (df['missing'] == 1).sum() / len(df)
print(f'Percentage of missing values: {missing_values}')