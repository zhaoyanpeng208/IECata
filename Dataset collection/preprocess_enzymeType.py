import pandas as pd
import re

df = pd.read_csv(r"F:\SecondStudy\Kotori\combination_clean_by_smiles.tsv", sep='\t')

def process_row(row):
    if 'wildtype' in row:
        return 'wildtype'
    elif 'wildtype' not in row:
        match = re.findall('[A-Z]\d+[A-Z]', row)
        if match:
            return ''.join(match)
        else:
            return ''

df['Processed'] = df['EnzymeType'].apply(process_row)
df.to_csv(r"F:\SecondStudy\Kotori\combination_clean_by_Enzymetype.tsv", sep='\t', index=False)
