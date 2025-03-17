import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Load the dataset
file_path = "c:/Users/PC/Downloads/BINF-5507-main/BINF-5507-main/Assignment1/BINF-5507/Assignment4/Data/RADCURE_Clinical_v04_20241219.xlsx"
xls = pd.ExcelFile(file_path)

# Load the main dataset sheet
df = pd.read_excel(xls, sheet_name='RADCURE_TCIA_Clinical_r2_offset')

# Preprocess the dataset
df['event'] = df['Status'].apply(lambda x: 1 if x == 'Dead' else 0)  # Convert Status to binary
df = df.rename(columns={'Length FU': 'survival_time'})  # Rename for clarity

# Select relevant columns
df_processed = df[['survival_time', 'event', 'Age', 'Sex', 'Stage', 'Tx Modality', 'Chemo']].dropna()

# Print a preview to confirm dataset is loaded
print("Dataset Loaded Successfully!")
print(df_processed.head())

# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Define treatment groups for comparison
treatment_groups = df_processed['Tx Modality'].unique()

# Plot survival curves for different treatment modalities
plt.figure(figsize=(10, 6))

for treatment in treatment_groups:
    mask = df_processed['Tx Modality'] == treatment
    kmf.fit(df_processed.loc[mask, 'survival_time'], df_processed.loc[mask, 'event'], label=treatment)
    kmf.plot_survival_function(ci_show=True)  # Show confidence intervals

plt.title("Kaplan-Meier Survival Curves by Treatment Modality")
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.legend(title="Tx Modality", loc="best")
plt.grid(True)
plt.show()

# Perform log-rank test for two key treatment groups (RT alone vs. RT+Chemo, if present)
group1_name = "RT alone"
group2_name = "RT+Chemo"

if group1_name in treatment_groups and group2_name in treatment_groups:
    group1 = df_processed[df_processed['Tx Modality'] == group1_name]
    group2 = df_processed[df_processed['Tx Modality'] == group2_name]

    results = logrank_test(group1['survival_time'], group2['survival_time'], 
                           event_observed_A=group1['event'], event_observed_B=group2['event'])
    
    print(f"Log-rank test p-value between {group1_name} and {group2_name}: {results.p_value:.4f}")

    if results.p_value < 0.05:
        print("Significant difference in survival curves.")
    else:
        print("No significant difference in survival curves.")




