import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=== Named Entity Recognition Dataset Analysis ===")
print("Loading and exploring the NER dataset...")

# Load the dataset
# Note: Replace 'ner_dataset.csv' with the actual path to your dataset
try:
    df = pd.read_csv('ner_dataset.csv')
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Dataset file 'ner_dataset.csv' not found. Please ensure the file is in the correct location.")
    # Create a sample dataset for demonstration
    sample_data = {
        'Sentence #': ['Sentence: 1'] * 11 + ['Sentence: 2'] * 8,
        'Word': ['Today', 'Michael', 'Jackson', 'and', 'Mark', 'ate', 'lasagna', 'at', 'New', 'Delhi', '.', 
                'Apple', 'Inc.', 'is', 'located', 'in', 'California', '.'],
        'POS': ['NN', 'NNP', 'NNP', 'CC', 'NNP', 'VBD', 'NN', 'IN', 'NNP', 'NNP', '.', 
               'NNP', 'NNP', 'VBZ', 'VBN', 'IN', 'NNP', '.'],
        'Tag': ['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O',
               'B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-geo', 'O']
    }
    df = pd.DataFrame(sample_data)
    print("Using sample dataset for demonstration.")

print("\n=== Dataset Overview ===")
print(df.head(15))
print(f"\nDataset Info:")
print(f"- Total rows: {len(df)}")
print(f"- Columns: {list(df.columns)}")
print(f"- Missing values: {df.isnull().sum().sum()}")

# Basic statistics
print(f"\n=== Basic Statistics ===")
unique_sentences = df['Sentence #'].nunique()
unique_words = df['Word'].nunique()
unique_tags = df['Tag'].nunique()

print(f"- Unique sentences: {unique_sentences}")
print(f"- Unique words: {unique_words}")
print(f"- Unique NER tags: {unique_tags}")
print(f"- Average words per sentence: {len(df) / unique_sentences:.2f}")

# NER tag distribution
print(f"\n=== NER Tag Distribution ===")
tag_counts = df['Tag'].value_counts()
print(tag_counts)

# Visualize tag distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
tag_counts.plot(kind='bar')
plt.title('NER Tag Distribution')
plt.xlabel('NER Tags')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# Tag categories analysis
tag_categories = defaultdict(int)
for tag in df['Tag'].unique():
    if tag == 'O':
        tag_categories['Outside'] += tag_counts[tag]
    elif tag.startswith('B-'):
        entity_type = tag[2:]
        tag_categories[f'B-{entity_type}'] += tag_counts[tag]
    elif tag.startswith('I-'):
        entity_type = tag[2:]
        tag_categories[f'I-{entity_type}'] += tag_counts[tag]

plt.subplot(1, 2, 2)
plt.pie(tag_categories.values(), labels=tag_categories.keys(), autopct='%1.1f%%')
plt.title('NER Tag Categories Distribution')

plt.tight_layout()
plt.savefig('ner_tag_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Sentence length analysis
print(f"\n=== Sentence Length Analysis ===")
sentence_lengths = df.groupby('Sentence #').size()
print(f"Sentence length statistics:")
print(f"- Mean: {sentence_lengths.mean():.2f}")
print(f"- Median: {sentence_lengths.median():.2f}")
print(f"- Min: {sentence_lengths.min()}")
print(f"- Max: {sentence_lengths.max()}")
print(f"- Std: {sentence_lengths.std():.2f}")

# Visualize sentence lengths
plt.figure(figsize=(10, 6))
plt.hist(sentence_lengths, bins=20, alpha=0.7, edgecolor='black')
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Number of Words per Sentence')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('sentence_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Entity analysis
print(f"\n=== Entity Analysis ===")
entities = []
current_entity = []
current_entity_type = None

for _, row in df.iterrows():
    tag = row['Tag']
    word = row['Word']
    
    if tag.startswith('B-'):
        # Save previous entity if exists
        if current_entity:
            entities.append({
                'entity': ' '.join(current_entity),
                'type': current_entity_type,
                'length': len(current_entity)
            })
        # Start new entity
        current_entity = [word]
        current_entity_type = tag[2:]
    elif tag.startswith('I-') and current_entity:
        # Continue current entity
        current_entity.append(word)
    else:
        # End current entity if exists
        if current_entity:
            entities.append({
                'entity': ' '.join(current_entity),
                'type': current_entity_type,
                'length': len(current_entity)
            })
            current_entity = []
            current_entity_type = None

# Add last entity if exists
if current_entity:
    entities.append({
        'entity': ' '.join(current_entity),
        'type': current_entity_type,
        'length': len(current_entity)
    })

entities_df = pd.DataFrame(entities)
if not entities_df.empty:
    print(f"Total entities found: {len(entities_df)}")
    print(f"\nEntity types distribution:")
    print(entities_df['type'].value_counts())
    
    print(f"\nSample entities:")
    for entity_type in entities_df['type'].unique():
        sample_entities = entities_df[entities_df['type'] == entity_type]['entity'].head(3).tolist()
        print(f"- {entity_type}: {sample_entities}")
    
    print(f"\nEntity length statistics:")
    print(entities_df['length'].describe())

# Data quality checks
print(f"\n=== Data Quality Checks ===")
print("Checking for potential issues...")

# Check for inconsistent tagging
inconsistencies = []
for sentence in df['Sentence #'].unique():
    sentence_data = df[df['Sentence #'] == sentence]
    tags = sentence_data['Tag'].tolist()
    
    # Check for I- tags without preceding B- tags
    for i, tag in enumerate(tags):
        if tag.startswith('I-'):
            entity_type = tag[2:]
            if i == 0 or not (tags[i-1] == f'B-{entity_type}' or tags[i-1] == f'I-{entity_type}'):
                inconsistencies.append(f"Sentence {sentence}: I-{entity_type} without proper B- tag")

if inconsistencies:
    print(f"Found {len(inconsistencies)} tagging inconsistencies:")
    for inc in inconsistencies[:5]:  # Show first 5
        print(f"- {inc}")
else:
    print("No tagging inconsistencies found!")

# Check for missing values
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(f"\nMissing data found:")
    print(missing_data[missing_data > 0])
else:
    print("No missing data found!")

print(f"\n=== Data Preprocessing Summary ===")
print("Dataset exploration completed successfully!")
print(f"- Dataset contains {unique_sentences} sentences with {len(df)} words")
print(f"- {unique_tags} unique NER tags identified")
print(f"- {len(entities_df) if not entities_df.empty else 0} entities extracted")
print("- Data quality checks completed")
print("\nReady for dataset splitting and model development!")
