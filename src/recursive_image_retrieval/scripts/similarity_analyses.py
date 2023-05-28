import pandas as pd 

df = pd.read_json('similarity_prompts.json')

column_names = ['e1', 'e2', 'e3', 'e4', 'e5']
df_sim = pd.DataFrame(df['similarity_augmenting'].to_list(), columns=column_names)

df_sim_copy = df_sim.copy()

# Shift the copy
df_shifted = df_sim_copy.shift(periods=1, axis="columns")

# Compute the relative difference on the copy
df_rel_diff = (df_sim_copy - df_shifted) / df_shifted

# Drop the first column from the relative difference DataFrame
df_rel_diff = df_rel_diff.drop(columns=['e1'])

# Compute and print the means of the relative differences
means = df_rel_diff.mean()


print('Relative difference experiment 1 and experiment 2: ',means[0])
print('Relative difference experiment 2 and experiment 3: ',means[1])
print('Relative difference experiment 3 and experiment 4: ',means[2])
print('Relative difference experiment 4 and experiment 4: ',means[3])


count_1 = (df_sim['e2'] > df_sim['e1']).sum()
print('Percentage e2 > e1: ',count_1/len(df_sim) * 100, '%')
count_2 = (df_sim['e3'] > df_sim['e1']).sum()
print('Percentage e3 > e1: ',count_2/len(df_sim) * 100,'%')
count_3 = (df_sim['e4'] > df_sim['e1']).sum()
print('Percentage e4 > e1: ',count_3/len(df_sim) * 100,'%')
count_4 = (df_sim['e5'] > df_sim['e1']).sum()
print('Percentage e5 > e1: ',count_4/len(df_sim) * 100,'%')