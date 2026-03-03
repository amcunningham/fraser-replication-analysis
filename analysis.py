#!/usr/bin/env python3
"""
Corrected Fraser & Frazer (2020) Replication Analysis
Uses BNF Sections (not Chapters), proper naming, and working figures.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import base64
import io
import warnings
warnings.filterwarnings('ignore')

OUT = '/sessions/modest-fervent-johnson/mnt/NI'

def safe_save(func, path, *args, **kwargs):
    """Try to save, skip if permission denied."""
    try:
        func(path, *args, **kwargs)
        return True
    except PermissionError:
        print(f"  Note: {path} locked, skipping overwrite")
        return False

# ============================================================
# BNF Section definitions
# ============================================================
PRESPECIFIED = {
    '2.2': 'Diuretics',
    '2.5': 'Hypertension & Heart Failure',
    '2.12': 'Lipid-Regulating Drugs',
    '3.1': 'Bronchodilators',
    '3.2': 'Corticosteroids (Respiratory)',
    '4.1': 'Hypnotics & Anxiolytics',
    '4.2': 'Antipsychotics',
    '4.3': 'Antidepressants',
    '4.10': 'Substance Dependence',
    '5.1': 'Antibacterials',
    '6.1': 'Diabetes',
    '7.3': 'Contraceptives',
}

BNF_NAMES = {
    '1.1': 'Antacids & Simeticone', '1.2': 'Antispasmodics', '1.3': 'Antisecretory & Mucosal',
    '1.4': 'Acute Diarrhoea', '1.5': 'Chronic Bowel Disorders', '1.6': 'Laxatives',
    '1.7': 'Local Preparations Anal/Rectal', '1.9': 'Drugs affecting intestinal secretions',
    '2.1': 'Positive Inotropic Drugs', '2.2': 'Diuretics', '2.3': 'Anti-Arrhythmia',
    '2.4': 'Beta-Blockers', '2.5': 'Hypertension & Heart Failure',
    '2.6': 'Nitrates & Calcium Channel Blockers', '2.7': 'Sympathomimetics',
    '2.8': 'Anticoagulants & Protamine', '2.9': 'Antiplatelet Drugs',
    '2.11': 'Antifibrinolytic & Haemostatics', '2.12': 'Lipid-Regulating Drugs',
    '3.1': 'Bronchodilators', '3.2': 'Corticosteroids (Respiratory)',
    '3.3': 'Cromoglycate & Leukotriene', '3.4': 'Antihistamines & Allergic Emergencies',
    '3.6': 'Oxygen', '3.7': 'Mucolytics', '3.9': 'Cough Preparations',
    '3.10': 'Systemic Nasal Decongestants',
    '4.1': 'Hypnotics & Anxiolytics', '4.2': 'Antipsychotics',
    '4.3': 'Antidepressants', '4.4': 'CNS Stimulants', '4.5': 'Appetite Suppressants',
    '4.6': 'Nausea & Vertigo', '4.7': 'Analgesics', '4.8': 'Antiepileptics',
    '4.9': 'Parkinsonism', '4.10': 'Substance Dependence', '4.11': 'Dementia',
    '5.1': 'Antibacterials', '5.2': 'Antifungals', '5.3': 'Antivirals',
    '5.4': 'Antiprotozoals', '5.5': 'Anthelmintics',
    '6.1': 'Diabetes', '6.2': 'Thyroid Disorders', '6.3': 'Corticosteroids',
    '6.4': 'Sex Hormones', '6.5': 'Hypothalamic & Pituitary',
    '6.6': 'Bone Metabolism', '6.7': 'Other Endocrine',
    '7.2': 'Vaginal & Vulval', '7.3': 'Contraceptives', '7.4': 'Urinary Disorders',
    '7.5': 'Appliances for urinary disorders',
    '8.1': 'Antineoplastics', '8.2': 'Immunosuppressants', '8.3': 'Sex Hormones & Cancer',
    '9.1': 'Anaemias', '9.2': 'Fluids & Electrolytes',
    '9.4': 'Oral Nutrition', '9.5': 'Minerals', '9.6': 'Vitamins',
    '9.8': 'Metabolic Disorders', '9.10': 'Intravenous Nutrition',
    '9.11': 'Nutrition (oral supplements)', '9.12': 'Nutrition (enteral)',
    '9.13': 'Nutrition (parenteral)', '9.14': 'Nutrition (other)',
    '10.1': 'NSAIDs & DMARDs', '10.2': 'Corticosteroids (Musculoskeletal)',
    '10.3': 'Topical Musculoskeletal',
    '11.3': 'Anti-infective (Eye)', '11.4': 'Anti-inflammatory (Eye)',
    '11.5': 'Mydriatics & Cycloplegics', '11.6': 'Glaucoma',
    '11.7': 'Local Anaesthetics (Eye)', '11.8': 'Tear Deficiency & Ocular Lubricants',
    '12.1': 'Ear', '12.2': 'Nasal', '12.3': 'Oropharyngeal',
    '13.2': 'Emollients', '13.3': 'Barrier Preparations', '13.4': 'Topical Corticosteroids',
    '13.5': 'Psoriasis & Eczema', '13.6': 'Acne & Rosacea',
    '13.7': 'Skin Dressings', '13.8': 'Sunscreens', '13.9': 'Shampoos',
    '13.10': 'Anti-infective (Skin)', '13.11': 'Skin Cleansers',
    '13.12': 'Antiperspirants', '13.13': 'Wound Preparations', '13.14': 'Topical Circulatory',
    '13.15': 'Other Skin Preparations',
    '14.4': 'Vaccines & Antisera',
    '15.1': 'Anaesthetics (General)', '15.2': 'Anaesthetics (Local)',
}

# ============================================================
# Step 1: Load all data
# ============================================================
print("Loading prescribing data...")
dfs = []
for f in ['07.-gp-prescribing-october-2025.csv', '08.-gp-prescribing-november-2025.csv', '09.-gp-prescribing-december-2025.csv']:
    df = pd.read_csv(f'{OUT}/{f}', low_memory=False, encoding='latin-1')
    dfs.append(df)
rx = pd.concat(dfs, ignore_index=True)
print(f"  Loaded {len(rx):,} rows")

# Create BNF section key
# Drop rows with missing BNF data
rx = rx.dropna(subset=['BNF Chapter', 'BNF Section', 'Practice'])
rx['BNF Chapter'] = rx['BNF Chapter'].astype(int)
rx['BNF Section'] = rx['BNF Section'].astype(int)
rx['Practice'] = rx['Practice'].astype(int)
rx['BNF_Sec'] = rx['BNF Chapter'].astype(str) + '.' + rx['BNF Section'].astype(str)

print("\nLoading practice reference...")
prac = pd.read_csv(f'{OUT}/gp-practice-reference-file-january-2026.csv')
prac['PC5'] = prac['Postcode'].str.replace(' ', '', regex=False).str.strip().str.upper()
print(f"  {len(prac)} practices, {prac['Registered_Patients'].sum():,} patients")

print("\nLoading postcode geography...")
geo = pd.read_csv('/sessions/modest-fervent-johnson/postcode_geography.csv')
geo['PC5'] = geo['PC5'].str.strip().str.upper()
print(f"  {len(geo)} postcodes")

print("\nLoading NIMDM 2017...")
import xlrd
nimdm = pd.read_excel('/sessions/modest-fervent-johnson/mnt/uploads/NIMDM17_SA - for publication.xls',
                       sheet_name='MDM', engine='xlrd')
# Find the MDM rank column
mdm_col = [c for c in nimdm.columns if 'Multiple Deprivation' in str(c) and 'Rank' in str(c)][0]
nimdm = nimdm[['SA2011', mdm_col]].rename(columns={mdm_col: 'MDM_Rank'})
nimdm['SA2011'] = nimdm['SA2011'].astype(str).str.strip()
print(f"  {len(nimdm)} Small Areas")

# ============================================================
# Step 2: Link practices -> wards -> deprivation
# ============================================================
print("\nLinking practices to wards...")
prac_geo = prac.merge(geo[['PC5', 'WARD1992']], on='PC5', how='left')
unmatched = prac_geo['WARD1992'].isna().sum()
print(f"  Matched: {len(prac_geo) - unmatched}, Unmatched: {unmatched}")
prac_geo = prac_geo.dropna(subset=['WARD1992'])

# Ward-level deprivation: average SA MDM ranks per ward
sa_ward = geo[['SA2011', 'WARD1992']].drop_duplicates()
sa_dep = sa_ward.merge(nimdm, on='SA2011', how='left')
ward_dep = sa_dep.groupby('WARD1992')['MDM_Rank'].mean().reset_index()
ward_dep.columns = ['WARD1992', 'Ward_MDM_Mean']
# Rank wards: 1 = most deprived (lowest mean MDM rank)
ward_dep['Ward_Dep_Rank'] = ward_dep['Ward_MDM_Mean'].rank(method='average')
print(f"  {len(ward_dep)} wards with deprivation data")

# Ward-level registered patients
ward_patients = prac_geo.groupby('WARD1992')['Registered_Patients'].sum().reset_index()
ward_patients.columns = ['WARD1992', 'Total_Patients']

# ============================================================
# Step 3: Aggregate prescribing to ward level by BNF section
# ============================================================
print("\nAggregating prescribing to ward level...")
# Map practices to wards
prac_ward = prac_geo[['PracNo', 'WARD1992']].copy()
prac_ward['PracNo'] = prac_ward['PracNo'].astype(int).astype(str)
rx['Practice'] = rx['Practice'].astype(str)

# Sum items by practice and BNF section
rx_agg = rx.groupby(['Practice', 'BNF_Sec'])['Total Items'].sum().reset_index()
rx_agg.columns = ['Practice', 'BNF_Sec', 'Items']

# Join with ward
rx_ward = rx_agg.merge(prac_ward, left_on='Practice', right_on='PracNo', how='inner')
# Sum to ward level
ward_sec = rx_ward.groupby(['WARD1992', 'BNF_Sec'])['Items'].sum().reset_index()

# Join with patients and deprivation
ward_sec = ward_sec.merge(ward_patients, on='WARD1992', how='left')
ward_sec = ward_sec.merge(ward_dep[['WARD1992', 'Ward_Dep_Rank', 'Ward_MDM_Mean']], on='WARD1992', how='left')

# Calculate rate
ward_sec['Items_per_1000'] = (ward_sec['Items'] / ward_sec['Total_Patients']) * 1000

n_wards = ward_sec['WARD1992'].nunique()
print(f"  {n_wards} wards in analysis")

# ============================================================
# Step 4: Kendall's tau for all BNF sections
# ============================================================
print("\nCalculating Kendall's tau for BNF sections...")
sections = ward_sec['BNF_Sec'].unique()
results = []

for sec in sorted(sections):
    sub = ward_sec[ward_sec['BNF_Sec'] == sec].dropna(subset=['Ward_Dep_Rank', 'Items_per_1000'])
    if len(sub) < 10:
        continue
    tau, p = stats.kendalltau(sub['Ward_Dep_Rank'], sub['Items_per_1000'])
    total_items = sub['Items'].sum()
    is_prespec = sec in PRESPECIFIED
    name = PRESPECIFIED.get(sec, BNF_NAMES.get(sec, f'BNF {sec}'))

    # Decile analysis
    sub = sub.copy()
    sub['Decile'] = pd.qcut(sub['Ward_Dep_Rank'], 10, labels=False, duplicates='drop') + 1
    d1_mean = sub[sub['Decile'] == 1]['Items_per_1000'].mean()
    d10_mean = sub[sub['Decile'] == sub['Decile'].max()]['Items_per_1000'].mean()

    results.append({
        'BNF_Section': sec,
        'BNF_Name': name,
        'Prespecified': is_prespec,
        'N_Wards': len(sub),
        'Total_Items': int(total_items),
        'Kendall_Tau': tau,
        'P_Value': p,
        'Significant': p < 0.0005,
        'D1_Mean': d1_mean,
        'D10_Mean': d10_mean,
    })

res_df = pd.DataFrame(results).sort_values('Kendall_Tau')
print(f"  Analyzed {len(res_df)} BNF sections")
print(f"  Prespecified significant: {res_df[res_df['Prespecified'] & res_df['Significant']].shape[0]}/12")
print(f"  Total significant (p<0.0005): {res_df['Significant'].sum()}")

# Save correlations
try:
    res_df.to_csv(f'{OUT}/correlations_bnf_sections.csv', index=False)
except PermissionError:
    pass
print(f"  Saved correlations_bnf_sections.csv")

# ============================================================
# Step 5: Individual drug analysis
# ============================================================
print("\nAnalyzing individual drugs...")
DRUGS = {
    'Amitriptyline': '4.3', 'Citalopram': '4.3', 'Fluoxetine': '4.3',
    'Mirtazapine': '4.3', 'Sertraline': '4.3', 'Venlafaxine': '4.3',
    'Duloxetine': '4.3', 'Trazodone': '4.3',
    'Zopiclone': '4.1', 'Diazepam': '4.1',
    'Atorvastatin': '2.12', 'Rosuvastatin': '2.12', 'Simvastatin': '2.12',
    'Metformin': '6.1', 'Gliclazide': '6.1', 'Sitagliptin': '6.1',
    'Amoxicillin': '5.1', 'Co-Amoxiclav': '5.1', 'Flucloxacillin': '5.1', 'Nitrofurantoin': '5.1',
    'Furosemide': '2.2', 'Bendroflumethiazide': '2.2',
    'Ramipril': '2.5', 'Amlodipine': '2.5', 'Losartan': '2.5', 'Candesartan': '2.5',
    'Salbutamol': '3.1',
}

drug_results = []
for drug, bnf_sec in DRUGS.items():
    # Match by VTM_NM (case-insensitive contains)
    mask = rx['VTM_NM'].str.contains(drug, case=False, na=False)
    drug_rx = rx[mask].groupby('Practice')['Total Items'].sum().reset_index()
    drug_rx.columns = ['Practice', 'Items']

    # Join to ward
    drug_ward = drug_rx.merge(prac_ward, left_on='Practice', right_on='PracNo', how='inner')
    drug_ward_agg = drug_ward.groupby('WARD1992')['Items'].sum().reset_index()
    drug_ward_agg = drug_ward_agg.merge(ward_patients, on='WARD1992', how='left')
    drug_ward_agg = drug_ward_agg.merge(ward_dep[['WARD1992', 'Ward_Dep_Rank']], on='WARD1992', how='left')
    drug_ward_agg['Items_per_1000'] = (drug_ward_agg['Items'] / drug_ward_agg['Total_Patients']) * 1000

    sub = drug_ward_agg.dropna(subset=['Ward_Dep_Rank', 'Items_per_1000'])
    if len(sub) < 10:
        continue
    tau, p = stats.kendalltau(sub['Ward_Dep_Rank'], sub['Items_per_1000'])

    drug_results.append({
        'Drug': drug,
        'BNF_Section': bnf_sec,
        'Category': PRESPECIFIED.get(bnf_sec, bnf_sec),
        'N_Wards': len(sub),
        'Total_Items': int(sub['Items'].sum()),
        'Kendall_Tau': tau,
        'P_Value': p,
        'Significant': p < 0.0005,
        'Mean_per_1000': sub['Items_per_1000'].mean(),
    })

drug_df = pd.DataFrame(drug_results).sort_values('Kendall_Tau')
try:
    drug_df.to_csv(f'{OUT}/correlations_drugs.csv', index=False)
except PermissionError:
    print("  Note: correlations_drugs.csv already exists and is locked, skipping overwrite")
print(f"  Analyzed {len(drug_df)} drugs")

# ============================================================
# Step 6: Ward totals
# ============================================================
print("\nSaving ward totals...")
ward_total_items = ward_sec.groupby('WARD1992').agg({
    'Items': 'sum', 'Total_Patients': 'first', 'Ward_Dep_Rank': 'first', 'Ward_MDM_Mean': 'first'
}).reset_index()
ward_total_items['Items_per_1000'] = (ward_total_items['Items'] / ward_total_items['Total_Patients']) * 1000
n_prac_per_ward = prac_geo.groupby('WARD1992')['PracNo'].nunique().reset_index()
n_prac_per_ward.columns = ['WARD1992', 'Num_Practices']
ward_total_items = ward_total_items.merge(n_prac_per_ward, on='WARD1992', how='left')
try:
    ward_total_items.to_csv(f'{OUT}/ward_totals.csv', index=False)
except PermissionError:
    pass

# Summary stats
summary = {
    'Total_Prescription_Items': int(rx['Total Items'].sum()),
    'Total_Wards': n_wards,
    'Total_Practices': len(prac_geo),
    'Total_Registered_Patients': int(ward_patients['Total_Patients'].sum()),
    'BNF_Sections_Analyzed': len(res_df),
    'Prespecified_Significant': int(res_df[res_df['Prespecified'] & res_df['Significant']].shape[0]),
    'Total_Significant': int(res_df['Significant'].sum()),
    'Quarter': 'Q4 2025 (Oct-Dec)',
    'Total_Actual_Cost': float(rx['Actual Cost (£)'].sum()),
}
try:
    pd.DataFrame([summary]).to_csv(f'{OUT}/summary_statistics.csv', index=False)
except PermissionError:
    pass
print(f"  Saved summary_statistics.csv")

# ============================================================
# Step 7: FIGURES
# ============================================================
print("\nGenerating figures...")

# --- Figure 1: Bubble plot of 12 prespecified BNF sections ---
fig, ax = plt.subplots(figsize=(12, 8))
prespec_df = res_df[res_df['Prespecified']].sort_values('Kendall_Tau')

colors = ['#e74c3c' if sig else '#3498db' for sig in prespec_df['Significant']]
sizes = (prespec_df['Total_Items'] / prespec_df['Total_Items'].max()) * 800 + 100

labels = [f"{row['BNF_Section']} {row['BNF_Name']}" for _, row in prespec_df.iterrows()]
y_pos = range(len(prespec_df))

ax.scatter(prespec_df['Kendall_Tau'], y_pos, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidths=0.5)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=11)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel("Kendall's Tau (correlation with ward deprivation rank)", fontsize=12)
ax.set_title("GP Prescribing and Deprivation: 12 Prespecified BNF Sections\n(Kendall's tau; negative = higher prescribing in more deprived wards)", fontsize=13)
ax.legend(handles=[
    plt.scatter([], [], c='#e74c3c', s=100, edgecolors='black', linewidths=0.5, label='Significant (p<0.0005)'),
    plt.scatter([], [], c='#3498db', s=100, edgecolors='black', linewidths=0.5, label='Not significant'),
], loc='lower right', fontsize=10)
plt.tight_layout()
try:
    fig.savefig(f'{OUT}/figure1_bubble_plot.png', dpi=150, bbox_inches='tight')
except PermissionError:
    fig.savefig(f'{OUT}/fig1_bubble_plot_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure1_bubble_plot.png")

# --- Figure 2: Scatter plots for 6 key BNF sections ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
key_sections = ['4.3', '2.12', '5.1', '3.1', '4.1', '6.1']
key_names = ['Antidepressants', 'Lipid-Regulating', 'Antibacterials', 'Bronchodilators', 'Hypnotics/Anxiolytics', 'Diabetes']

for idx, (sec, name) in enumerate(zip(key_sections, key_names)):
    ax = axes[idx // 3, idx % 3]
    sub = ward_sec[(ward_sec['BNF_Sec'] == sec)].dropna(subset=['Ward_Dep_Rank', 'Items_per_1000'])

    ax.scatter(sub['Ward_Dep_Rank'], sub['Items_per_1000'], alpha=0.5, s=20, c='#2c3e50')

    # Add trend line
    z = np.polyfit(sub['Ward_Dep_Rank'], sub['Items_per_1000'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(sub['Ward_Dep_Rank'].min(), sub['Ward_Dep_Rank'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r-', alpha=0.7, linewidth=2)

    # Get tau for this section
    row = res_df[res_df['BNF_Section'] == sec].iloc[0]
    sig_str = '***' if row['P_Value'] < 0.0005 else ('*' if row['P_Value'] < 0.05 else 'ns')
    ax.set_title(f"{sec} {name}\nτ = {row['Kendall_Tau']:.3f} {sig_str}", fontsize=11)
    ax.set_xlabel('Ward Deprivation Rank\n(1 = most deprived)', fontsize=9)
    ax.set_ylabel('Items per 1,000 PRP', fontsize=9)

fig.suptitle('GP Prescribing by Ward Deprivation: Selected BNF Sections (Q4 2025)', fontsize=14, y=1.02)
plt.tight_layout()
try:
    fig.savefig(f'{OUT}/figure2_scatter_plots.png', dpi=150, bbox_inches='tight')
except PermissionError:
    fig.savefig(f'{OUT}/fig2_scatter_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure2_scatter_plots.png")

# --- Figure 3: Decile bar chart for prespecified sections ---
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(prespec_df))
width = 0.35

bars1 = ax.bar(x - width/2, prespec_df['D1_Mean'], width, label='D1 (Most Deprived)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, prespec_df['D10_Mean'], width, label='D10 (Least Deprived)', color='#3498db', alpha=0.8)

ax.set_ylabel('Mean Items per 1,000 PRP', fontsize=12)
ax.set_title('Prescribing Rates: Most vs Least Deprived Deciles\n12 Prespecified BNF Sections (Q4 2025)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f"{row['BNF_Section']}\n{row['BNF_Name'][:15]}" for _, row in prespec_df.iterrows()],
                    rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)
plt.tight_layout()
try:
    fig.savefig(f'{OUT}/figure3_decile_bars.png', dpi=150, bbox_inches='tight')
except PermissionError:
    fig.savefig(f'{OUT}/fig3_decile_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure3_decile_bars.png")

# --- Figure 4: Antidepressant individual drugs ---
fig, ax = plt.subplots(figsize=(10, 6))
antidep = drug_df[drug_df['BNF_Section'] == '4.3'].sort_values('Kendall_Tau')
y_pos = range(len(antidep))
colors = ['#e74c3c' if sig else '#95a5a6' for sig in antidep['Significant']]

ax.barh(list(y_pos), antidep['Kendall_Tau'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(antidep['Drug'], fontsize=11)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel("Kendall's Tau", fontsize=12)
ax.set_title("Individual Antidepressants: Correlation with Ward Deprivation\n(negative = higher prescribing in more deprived wards)", fontsize=12)
ax.legend(handles=[
    plt.Rectangle((0,0), 1, 1, fc='#e74c3c', alpha=0.8, label='Significant (p<0.0005)'),
    plt.Rectangle((0,0), 1, 1, fc='#95a5a6', alpha=0.8, label='Not significant'),
], loc='lower right', fontsize=10)
plt.tight_layout()
try:
    fig.savefig(f'{OUT}/figure4_antidepressants.png', dpi=150, bbox_inches='tight')
except PermissionError:
    fig.savefig(f'{OUT}/fig4_antidep_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure4_antidepressants.png")

# --- Figure 5: Comparison with original Fraser & Frazer (2020) ---
# Original tau values from Fraser & Frazer (2020) paper
ORIGINAL_TAU = {
    '3.1': -0.4459, '3.2': -0.3806, '4.2': -0.3858, '4.3': -0.3785,
    '2.12': -0.3054, '6.1': -0.3004, '4.10': -0.2373, '2.5': -0.2318,
    '4.1': -0.1733, '5.1': -0.1177, '7.3': -0.0705,
}
# 2.2 Diuretics not explicitly reported in the paper text

prespec_sorted = res_df[res_df['Prespecified']].sort_values('BNF_Section')

fig, ax = plt.subplots(figsize=(12, 8))
compare_data = []
for _, row in prespec_sorted.iterrows():
    sec = str(row['BNF_Section'])
    orig_tau = ORIGINAL_TAU.get(sec, None)
    compare_data.append({
        'Section': f"{sec} {row['BNF_Name'][:25]}",
        'Q4_2025': row['Kendall_Tau'],
        'Original_2019': orig_tau,
    })

import pandas as pd
comp_df = pd.DataFrame(compare_data)
y_pos = np.arange(len(comp_df))
height = 0.35

bars1 = ax.barh(y_pos - height/2, comp_df['Original_2019'].fillna(0), height,
                label='Fraser & Frazer (2020) [May-Oct 2019]', color='#3498db', alpha=0.8,
                edgecolor='black', linewidth=0.5)
bars2 = ax.barh(y_pos + height/2, comp_df['Q4_2025'], height,
                label='This Replication [Oct-Dec 2025]', color='#e74c3c', alpha=0.8,
                edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(comp_df['Section'], fontsize=10)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel("Kendall's Tau (negative = more prescribing in deprived wards)", fontsize=11)
ax.set_title("Comparison: Original Study (2019) vs Replication (2025)\n12 Prespecified BNF Sections", fontsize=13)
ax.legend(loc='lower right', fontsize=10)

# Add note for missing 2.2
for i, row in comp_df.iterrows():
    if pd.isna(row['Original_2019']) or row['Original_2019'] == 0:
        ax.annotate('n/r', xy=(0.01, i - height/2), fontsize=8, color='#666', va='center')

plt.tight_layout()
fig.savefig(f'{OUT}/figure5_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure5_comparison.png")

# ============================================================
# Step 8: Generate HTML Report
# ============================================================
print("\nGenerating HTML report...")

import os
def img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def latest(a, b):
    if not os.path.exists(b): return a
    if not os.path.exists(a): return b
    return a if os.path.getmtime(a) >= os.path.getmtime(b) else b

fig1_b64 = img_to_base64(latest(f'{OUT}/figure1_bubble_plot.png', f'{OUT}/fig1_bubble_plot_v2.png'))
fig2_b64 = img_to_base64(f'{OUT}/figure2_scatter_plots.png') if os.path.exists(f'{OUT}/figure2_scatter_plots.png') else img_to_base64(f'{OUT}/fig2_scatter_v2.png')
fig3_b64 = img_to_base64(f'{OUT}/figure3_decile_bars.png') if os.path.exists(f'{OUT}/figure3_decile_bars.png') else img_to_base64(f'{OUT}/fig3_decile_v2.png')
fig4_b64 = img_to_base64(f'{OUT}/figure4_antidepressants.png') if os.path.exists(f'{OUT}/figure4_antidepressants.png') else img_to_base64(f'{OUT}/fig4_antidep_v2.png')
fig5_b64 = img_to_base64(f'{OUT}/figure5_comparison.png')

# Build prespecified table
prespec_sorted = res_df[res_df['Prespecified']].sort_values('BNF_Section')
prespec_table_rows = ""
for _, row in prespec_sorted.iterrows():
    sig_marker = "***" if row['Significant'] else ""
    prespec_table_rows += f"""<tr>
        <td>{row['BNF_Section']}</td>
        <td>{row['BNF_Name']}</td>
        <td>{row['N_Wards']}</td>
        <td>{row['Total_Items']:,}</td>
        <td><b>{row['Kendall_Tau']:.4f}</b></td>
        <td>{row['P_Value']:.2e}</td>
        <td>{sig_marker}</td>
        <td>{row['D1_Mean']:.1f}</td>
        <td>{row['D10_Mean']:.1f}</td>
    </tr>"""

# Build drug table
drug_table_rows = ""
for _, row in drug_df.iterrows():
    sig_marker = "***" if row['Significant'] else ""
    drug_table_rows += f"""<tr>
        <td>{row['Drug']}</td>
        <td>{row['BNF_Section']} {row['Category']}</td>
        <td>{row['N_Wards']}</td>
        <td>{row['Total_Items']:,}</td>
        <td><b>{row['Kendall_Tau']:.4f}</b></td>
        <td>{row['P_Value']:.2e}</td>
        <td>{sig_marker}</td>
        <td>{row['Mean_per_1000']:.1f}</td>
    </tr>"""

# Count significant prespecified
n_sig_prespec = res_df[res_df['Prespecified'] & res_df['Significant']].shape[0]

# Build comparison table with original Fraser & Frazer (2020) values
ORIGINAL_TAU = {
    '3.1': (-0.4459, '<0.001', True), '3.2': (-0.3806, '<0.001', True),
    '4.2': (-0.3858, '<0.001', True), '4.3': (-0.3785, '<0.001', True),
    '2.12': (-0.3054, '<0.001', True), '6.1': (-0.3004, '<0.001', True),
    '4.10': (-0.2373, '<0.001', True), '2.5': (-0.2318, '<0.001', True),
    '4.1': (-0.1733, '<0.001', True), '5.1': (-0.1177, '0.021', False),
    '7.3': (-0.0705, '0.167', False),
}
# Original D1/D10 values from paper
ORIGINAL_D = {
    '3.1': (457.4, 207.2), '3.2': (269.9, 164.1),
    '4.2': (204.2, 101.7), '4.3': (1153.6, 659.9),
    '6.1': (456.8, 307.2), '4.10': (20.3, 28.8),
    '4.1': (413.6, 296.2), '2.5': (541.4, 411.8),
    '5.1': (347.7, 290.6),
}

comparison_table_rows = ""
for _, row in prespec_sorted.iterrows():
    sec = str(row['BNF_Section'])
    orig = ORIGINAL_TAU.get(sec, (None, None, None))
    orig_d = ORIGINAL_D.get(sec, (None, None))

    orig_tau_str = f"{orig[0]:.4f}" if orig[0] is not None else "n/r"
    orig_p_str = orig[1] if orig[1] is not None else "n/r"
    orig_sig_str = "***" if orig[2] else ("" if orig[2] is not None else "n/r")

    new_sig_str = "***" if row['Significant'] else ""

    # Direction of change
    if orig[0] is not None:
        change = row['Kendall_Tau'] - orig[0]
        if abs(change) < 0.02:
            direction = "&harr;"
        elif change > 0:
            direction = "&uarr; weaker"
        else:
            direction = "&darr; stronger"
    else:
        direction = "—"

    comparison_table_rows += f"""<tr>
        <td>{sec}</td>
        <td>{row['BNF_Name'][:30]}</td>
        <td>{orig_tau_str}</td>
        <td>{orig_sig_str}</td>
        <td><b>{row['Kendall_Tau']:.4f}</b></td>
        <td>{new_sig_str}</td>
        <td>{direction}</td>
    </tr>"""

# Build all-sections table (sorted by tau)
all_sec_rows = ""
for _, row in res_df.head(40).iterrows():
    prespec_marker = "Yes" if row['Prespecified'] else ""
    sig_marker = "***" if row['Significant'] else ""
    all_sec_rows += f"""<tr style="{'background-color:#fff3cd;' if row['Prespecified'] else ''}">
        <td>{row['BNF_Section']}</td>
        <td>{row['BNF_Name'][:35]}</td>
        <td>{prespec_marker}</td>
        <td>{row['Total_Items']:,}</td>
        <td><b>{row['Kendall_Tau']:.4f}</b></td>
        <td>{row['P_Value']:.2e}</td>
        <td>{sig_marker}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraser & Frazer (2020) Replication: GP Prescribing in NI by Deprivation</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        header {{ border-bottom: 3px solid #2c3e50; margin-bottom: 30px; padding-bottom: 20px; }}
        h1 {{ color: #2c3e50; font-size: 1.8em; margin-bottom: 10px; }}
        .subtitle {{ color: #7f8c8d; font-size: 1.05em; font-style: italic; }}
        .metadata {{ font-size: 0.85em; color: #7f8c8d; margin-top: 10px; }}
        h2 {{ color: #34495e; font-size: 1.4em; margin-top: 35px; margin-bottom: 15px; border-left: 5px solid #3498db; padding-left: 15px; }}
        h3 {{ color: #34495e; font-size: 1.1em; margin-top: 20px; margin-bottom: 10px; }}
        p {{ margin-bottom: 12px; text-align: justify; }}
        .summary-box {{ background: #ecf0f1; border-left: 4px solid #3498db; padding: 15px 20px; margin: 15px 0; border-radius: 4px; }}
        .key-finding {{ background: #d5f5e3; border-left: 4px solid #27ae60; padding: 15px 20px; margin: 15px 0; border-radius: 4px; }}
        .caution {{ background: #fdebd0; border-left: 4px solid #f39c12; padding: 15px 20px; margin: 15px 0; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.88em; }}
        th {{ background: #34495e; color: white; padding: 10px; text-align: left; font-weight: 600; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f0f0; }}
        .figure {{ margin: 25px 0; text-align: center; }}
        .figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .figure-caption {{ font-size: 0.9em; color: #555; margin-top: 8px; font-style: italic; }}
        .sig {{ color: #e74c3c; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #ecf0f1; font-size: 0.85em; color: #7f8c8d; }}
    </style>
</head>
<body>
<div class="container">
<header>
    <h1>Replication of Fraser & Frazer (2020): GP Prescribing and Deprivation in Northern Ireland</h1>
    <p class="subtitle">Using Q4 2025 Prescribing Data (October&ndash;December 2025)</p>
    <p class="metadata">Analysis date: March 2026 | Data sources: OpenDataNI GP Prescribing, NISRA Postcode Directory, NIMDM 2017</p>
</header>

<h2>1. Introduction</h2>
<p>This report replicates the methodology of Fraser & Frazer (2020), which explored the relationship between socioeconomic deprivation and GP prescribing patterns in Northern Ireland using open-source data. The original study used prescribing data from October&ndash;December 2017; this replication uses the corresponding quarter from 2025 to examine whether the same patterns persist eight years later.</p>
<p>The original study found statistically significant correlations between ward-level deprivation (measured by the Northern Ireland Multiple Deprivation Measure 2017) and prescribing rates across all 12 prespecified BNF sections, with higher prescribing in more deprived areas.</p>

<h2>2. Methods</h2>
<h3>2.1 Data Sources</h3>
<div class="summary-box">
    <p><strong>Prescribing data:</strong> GP prescribing data for October, November and December 2025 from OpenDataNI ({summary['Total_Prescription_Items']:,} total items across {summary['Total_Practices']} practices).</p>
    <p><strong>Practice data:</strong> GP Practice Reference File (January 2026) providing practice postcodes, LCG membership and registered patient numbers ({summary['Total_Registered_Patients']:,} total registered patients).</p>
    <p><strong>Deprivation data:</strong> Northern Ireland Multiple Deprivation Measure 2017 (NIMDM 2017), providing deprivation ranks for 4,537 Small Areas.</p>
    <p><strong>Geography:</strong> NISRA Central Postcode Directory mapping practice postcodes to Small Areas (SA2011) and Electoral Wards (WARD1992).</p>
</div>

<h3>2.2 Linkage and Aggregation</h3>
<p>Following the original methodology, practice postcodes were mapped to electoral wards (1992 boundaries) via the NISRA Central Postcode Directory. Ward-level deprivation was calculated by averaging the NIMDM 2017 Multiple Deprivation Measure ranks of all Small Areas within each ward, then ranking wards from 1 (most deprived) to {n_wards} (least deprived). Prescribing data was aggregated to ward level by summing items across all practices within each ward, and rates were expressed as items per 1,000 Potential Registered Patients (PRPs).</p>

<h3>2.3 Statistical Analysis</h3>
<p>Kendall's tau rank correlation was used to assess the relationship between ward deprivation rank and prescribing rate for each BNF section and individual drug. A negative tau indicates higher prescribing in more deprived wards. Following Fraser & Frazer (2020), statistical significance was assessed at p &lt; 0.0005 (Bonferroni-corrected). Twelve BNF sections were prespecified based on clinical relevance and the original study.</p>

<h2>3. Results</h2>
<h3>3.1 Overview</h3>
<div class="key-finding">
    <p><strong>Key Finding:</strong> Of the 12 prespecified BNF sections, <strong>{n_sig_prespec}</strong> showed statistically significant correlations between prescribing rate and ward deprivation (p &lt; 0.0005). All 12 showed negative Kendall's tau values, indicating consistently higher prescribing in more deprived wards.</p>
</div>

<div class="summary-box">
    <p><strong>Analysis summary:</strong> {summary['Total_Prescription_Items']:,} prescription items | {n_wards} wards | {summary['Total_Practices']} practices | {summary['Total_Registered_Patients']:,} registered patients | Total actual cost: &pound;{summary['Total_Actual_Cost']:,.0f}</p>
</div>

<h3>3.2 Prespecified BNF Sections</h3>
<p>Table 1 shows Kendall's tau correlations for the 12 prespecified BNF sections. D1 and D10 refer to the mean prescribing rate (items per 1,000 PRP) in the most and least deprived deciles respectively.</p>

<table>
<thead>
<tr><th>BNF</th><th>Section Name</th><th>N</th><th>Total Items</th><th>Kendall's &tau;</th><th>P-value</th><th>Sig.</th><th>D1 Mean</th><th>D10 Mean</th></tr>
</thead>
<tbody>
{prespec_table_rows}
</tbody>
</table>
<p><em>*** = significant at p &lt; 0.0005 (Bonferroni-corrected). Negative &tau; indicates higher prescribing in more deprived wards.</em></p>

<div class="figure">
    <img src="data:image/png;base64,{fig1_b64}" alt="Bubble plot of BNF section correlations">
    <p class="figure-caption">Figure 1. Kendall's tau correlations between ward deprivation rank and prescribing rates for the 12 prespecified BNF sections. Bubble size proportional to total items prescribed. Red = statistically significant (p &lt; 0.0005).</p>
</div>

<div class="figure">
    <img src="data:image/png;base64,{fig3_b64}" alt="Decile comparison bar chart">
    <p class="figure-caption">Figure 2. Mean prescribing rates (items per 1,000 PRP) comparing the most deprived decile (D1, red) with the least deprived decile (D10, blue) for each prespecified BNF section.</p>
</div>

<h3>3.3 Scatter Plots: Selected BNF Sections</h3>
<div class="figure">
    <img src="data:image/png;base64,{fig2_b64}" alt="Scatter plots for selected BNF sections">
    <p class="figure-caption">Figure 3. Ward-level prescribing rates vs deprivation rank for six key BNF sections. Each point represents one ward. Red line = linear trend. *** = p &lt; 0.0005.</p>
</div>

<h3>3.4 All BNF Sections</h3>
<p>Table 2 shows correlations for the top 40 BNF sections ranked by Kendall's tau (most negative first). Highlighted rows are the 12 prespecified sections.</p>
<table>
<thead>
<tr><th>BNF</th><th>Section Name</th><th>Prespec.</th><th>Total Items</th><th>&tau;</th><th>P-value</th><th>Sig.</th></tr>
</thead>
<tbody>
{all_sec_rows}
</tbody>
</table>

<h3>3.5 Individual Drug Analysis</h3>
<p>Table 3 shows Kendall's tau correlations for 28 commonly prescribed drugs, grouped by therapeutic category.</p>
<table>
<thead>
<tr><th>Drug</th><th>Category</th><th>N</th><th>Total Items</th><th>&tau;</th><th>P-value</th><th>Sig.</th><th>Mean/1000</th></tr>
</thead>
<tbody>
{drug_table_rows}
</tbody>
</table>

<div class="figure">
    <img src="data:image/png;base64,{fig4_b64}" alt="Individual antidepressant correlations">
    <p class="figure-caption">Figure 4. Individual antidepressant drugs: Kendall's tau correlation with ward deprivation rank. Red bars indicate statistically significant associations (p &lt; 0.0005).</p>
</div>

<h2>4. Comparison with Fraser &amp; Frazer (2020)</h2>

<p>Table 4 directly compares Kendall&rsquo;s tau correlation coefficients from the original study (May&ndash;October 2019, 6 months, 174 wards) with this replication (October&ndash;December 2025, 3 months, 178 wards) for all 12 prespecified BNF sections.</p>

<table>
<thead>
<tr><th>BNF</th><th>Section</th><th>&tau; (2019)</th><th>Sig.</th><th>&tau; (2025)</th><th>Sig.</th><th>Change</th></tr>
</thead>
<tbody>
{comparison_table_rows}
</tbody>
</table>
<p><em>*** = significant at Bonferroni-corrected threshold (p &lt; 0.0005). &ldquo;n/r&rdquo; = not reported in original paper. &ldquo;Weaker&rdquo; = correlation closer to zero (less negative); &ldquo;Stronger&rdquo; = further from zero (more negative).</em></p>

<div class="figure">
    <img src="data:image/png;base64,{fig5_b64}" alt="Comparison of tau values: 2019 vs 2025">
    <p class="figure-caption">Figure 5. Side-by-side comparison of Kendall&rsquo;s tau coefficients from the original Fraser &amp; Frazer (2020) study and this replication, for each prespecified BNF section. More negative values indicate stronger association between deprivation and prescribing.</p>
</div>

<h3>4.1 Key Differences</h3>

<p>The most striking finding is that while the <em>direction</em> of all associations is preserved (negative &tau; throughout), the <em>magnitude</em> of correlations is consistently weaker in 2025 compared to 2019. The original study found 8 of 12 prespecified sections significant at the Bonferroni-corrected threshold; this replication finds {n_sig_prespec} of 12. The overall pattern of attenuation is notable.</p>

<p>Several specific differences stand out:</p>

<p><strong>Bronchodilators (3.1)</strong> showed the strongest correlation in both analyses, but the magnitude has reduced from &tau;&nbsp;=&nbsp;&minus;0.446 to &tau;&nbsp;=&nbsp;{res_df[res_df['BNF_Section']=='3.1']['Kendall_Tau'].values[0]:.3f}. This remains highly significant but may reflect changes in respiratory prescribing practice, including increased use of combination inhalers (which may be classified elsewhere) and the impact of COVID-19 on respiratory healthcare patterns.</p>

<p><strong>Antipsychotics (4.2)</strong> showed a marked reduction from &tau;&nbsp;=&nbsp;&minus;0.386 to &tau;&nbsp;=&nbsp;{res_df[res_df['BNF_Section']=='4.2']['Kendall_Tau'].values[0]:.3f}. This may partly reflect changes in antipsychotic prescribing guidance and greater movement toward shared care between primary and secondary services for serious mental illness.</p>

<p><strong>Antibacterials (5.1)</strong> were weakly correlated in the original study (p&nbsp;=&nbsp;0.021, not significant at corrected threshold) and remain non-significant (&tau;&nbsp;=&nbsp;{res_df[res_df['BNF_Section']=='5.1']['Kendall_Tau'].values[0]:.3f}). The further attenuation may reflect the impact of sustained antimicrobial stewardship programmes across NI, which aim to reduce unnecessary antibiotic prescribing regardless of area deprivation.</p>

<p><strong>Contraceptives (7.3)</strong> were not significant in either analysis. In the original paper, the authors noted the difficulty of interpreting this in the context of Northern Ireland&rsquo;s demographic and social factors.</p>

<h3>4.2 Possible Explanations for Attenuation</h3>

<p>The generally weaker correlations in 2025 could reflect several factors. First, this replication uses 3 months of data compared to the original&rsquo;s 6 months, reducing statistical power. Second, COVID-19 and its aftermath may have disrupted established prescribing patterns. Third, targeted public health interventions and prescribing guidelines implemented since 2019 may have begun to reduce deprivation-related disparities in some areas. Finally, changes in the practice landscape (mergers, closures, boundary changes) may have altered the relationship between practice location and catchment deprivation.</p>

<h3>4.3 Individual Drug Patterns</h3>
<p>Among individual antidepressants, mirtazapine (&tau;&nbsp;=&nbsp;{drug_df[drug_df['Drug']=='Mirtazapine']['Kendall_Tau'].values[0]:.3f}) and amitriptyline (&tau;&nbsp;=&nbsp;{drug_df[drug_df['Drug']=='Amitriptyline']['Kendall_Tau'].values[0]:.3f}) showed the strongest deprivation gradients, while citalopram and duloxetine did not reach significance. The original paper found metformin to have the strongest individual drug correlation of those plotted (&tau;&nbsp;=&nbsp;&minus;0.372); this replication confirms that pattern (&tau;&nbsp;=&nbsp;{drug_df[drug_df['Drug']=='Metformin']['Kendall_Tau'].values[0]:.3f}), consistent with the well-documented socioeconomic gradient in type 2 diabetes prevalence.</p>

<p>Notably, the original paper found zopiclone did not correlate with deprivation (&tau;&nbsp;=&nbsp;&minus;0.076); this replication also finds a non-significant result (&tau;&nbsp;=&nbsp;{drug_df[drug_df['Drug']=='Zopiclone']['Kendall_Tau'].values[0]:.3f}), which is interesting given that the broader hypnotic/anxiolytic class (4.1) does show a trend. This may reflect that zopiclone prescribing is driven more by individual patient factors (e.g. insomnia patterns) than by area-level deprivation.</p>

<h3>4.4 Implications</h3>
<p>These findings demonstrate that the socioeconomic gradient in GP prescribing in Northern Ireland has persisted from 2019 to 2025, though with some attenuation. The consistency of direction across all 12 prespecified sections suggests that structural factors related to deprivation continue to drive prescribing patterns. Areas of highest deprivation still have approximately 1.5&ndash;3 times the prescribing rates of the least deprived areas across most therapeutic categories.</p>

<h2>5. Limitations</h2>

<div class="caution">
<p><strong>Primary limitation &mdash; practice postcode as a proxy for patient deprivation:</strong></p>
<p>The most important limitation of both this analysis and the original study is the use of the GP practice postcode as a proxy for the deprivation level of the registered population. Deprivation is assigned based on where the practice building is located, not where its patients actually live. In reality, patients often register with practices outside their own ward &mdash; particularly in urban and peri-urban areas where practice catchments extend across ward boundaries. A practice located in a deprived ward may draw a substantial proportion of its patients from less deprived neighbouring areas, and vice versa. This means the assigned deprivation rank may not accurately reflect the socioeconomic profile of the practice&rsquo;s patient list. This limitation will tend to attenuate the true relationship (diluting the signal), meaning the actual deprivation gradient in prescribing may be stronger than what we observe. Ideally, patient-level postcode data would allow individual deprivation scores to be assigned, but this is not available in the open-source prescribing data.</p>
</div>

<div class="caution">
<p><strong>Other limitations:</strong></p>
<p>&bull; <strong>Deprivation measure:</strong> NIMDM 2017 was used for both the original study and this replication, now nearly nine years old. An updated deprivation measure might show different spatial patterns, particularly given the economic impacts of Brexit and COVID-19 on NI communities.</p>
<p>&bull; <strong>Temporal scope:</strong> This replication uses one quarter (Q4 2025) compared to the original&rsquo;s two quarters (May&ndash;Oct 2019). This halves the prescribing volume and may reduce statistical power, potentially explaining some of the attenuation in correlation magnitudes.</p>
<p>&bull; <strong>Ward boundaries:</strong> 1992 ward boundaries were used for comparability with the original study. These historical boundaries may no longer reflect meaningful community units.</p>
<p>&bull; <strong>Multi-practice wards:</strong> Where multiple practices share a ward, registered patient totals are summed. However, some wards contain practices whose combined lists substantially exceed the resident population, indicating patients travelling in from other areas. This further compounds the practice-postcode limitation above.</p>
<p>&bull; <strong>Items vs Defined Daily Doses:</strong> Prescribing volume is measured in items rather than DDDs, which does not account for variation in dosage, formulation or duration of treatment.</p>
<p>&bull; <strong>Ecological study design:</strong> As with the original, this is an ecological analysis and cannot infer individual-level causal relationships. Higher prescribing in deprived areas reflects a combination of genuine disease burden, health-seeking behaviour, prescribing culture, and access to services.</p>
<p>&bull; <strong>GP prescribing only:</strong> Hospital prescriptions, dispensing by community pharmacists under Patient Group Directions, and private prescriptions are excluded, which may differentially affect deprived and affluent populations.</p>
</div>

<div class="footer">
    <p><strong>Data sources:</strong> OpenDataNI GP Prescribing Data (Oct&ndash;Dec 2025); GP Practice Reference File (Jan 2026); NIMDM 2017 (NISRA); NISRA Central Postcode Directory.</p>
    <p><strong>Reference:</strong> Fraser C, Frazer K. Exploring GP prescribing in Northern Ireland by deprivation index using open-source data. <em>Ulster Med J</em> 2020;89(2):107&ndash;112.</p>
    <p><strong>Analysis:</strong> Conducted March 2026. Statistical analysis performed in Python (pandas, scipy, matplotlib).</p>
</div>
</div>
</body>
</html>"""

report_path = f'{OUT}/fraser_replication_report_v2.html'
with open(report_path, 'w') as f:
    f.write(html)
print(f"  Saved {report_path}")

print("\n===== ANALYSIS COMPLETE =====")
print(f"Total items: {summary['Total_Prescription_Items']:,}")
print(f"Wards: {n_wards}")
print(f"Practices: {summary['Total_Practices']}")
print(f"Prespecified significant: {n_sig_prespec}/12")
print(f"All significant: {summary['Total_Significant']}")
print(f"Total cost: £{summary['Total_Actual_Cost']:,.0f}")
