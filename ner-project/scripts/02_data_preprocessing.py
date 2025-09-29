import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
import json

print("=== Data Preprocessing and Dataset Splitting ===")

# Load the dataset (using sample data if file not found)
try:
    df = pd.read_csv('ner_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Using sample dataset for demonstration...")
    sample_data = {
        'Sentence #': ['Sentence: 1'] * 11 + ['Sentence: 2'] * 8 + ['Sentence: 3'] * 6 + ['Sentence: 4'] * 9,
        'Word': ['Today', 'Michael', 'Jackson', 'and', 'Mark', 'ate', 'lasagna', 'at', 'New', 'Delhi', '.', 
                'Apple', 'Inc.', 'is', 'located', 'in', 'California', '.', 
                'John', 'works', 'at', 'Google', 'Inc.', '.',
                'The', 'meeting', 'is', 'scheduled', 'in', 'New', 'York', 'tomorrow', '.'],
        'POS': ['NN', 'NNP', 'NNP', 'CC', 'NNP', 'VBD', 'NN', 'IN', 'NNP', 'NNP', '.', 
               'NNP', 'NNP', 'VBZ', 'VBN', 'IN', 'NNP', '.', 
               'NNP', 'VBZ', 'IN', 'NNP', 'NNP', '.',
               'DT', 'NN', 'VBZ', 'VBN', 'IN', 'NNP', 'NNP', 'NN', '.'],
        'Tag': ['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O',
               'B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-geo', 'O',
               'B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O',
               'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O']
    }
    df = pd.DataFrame(sample_data)

# Group data by sentences
def group_by_sentences(dataframe):
    """Group words and tags by sentences"""
    sentences = []
    labels = []
    
    for sentence_id in dataframe['Sentence #'].unique():
        sentence_data = dataframe[dataframe['Sentence #'] == sentence_id]
        words = sentence_data['Word'].tolist()
        tags = sentence_data['Tag'].tolist()
        
        sentences.append(words)
        labels.append(tags)
    
    return sentences, labels

sentences, labels = group_by_sentences(df)
print(f"Processed {len(sentences)} sentences")

# Create vocabulary and tag mappings
def create_mappings(sentences, labels):
    """Create word-to-index and tag-to-index mappings"""
    word_vocab = set()
    tag_vocab = set()
    
    for sentence in sentences:
        word_vocab.update(sentence)
    
    for label_seq in labels:
        tag_vocab.update(label_seq)
    
    # Add special tokens
    word_vocab.add('<PAD>')
    word_vocab.add('<UNK>')
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(sorted(word_vocab))}
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_vocab))}
    
    # Create reverse mappings
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    return word2idx, tag2idx, idx2word, idx2tag

word2idx, tag2idx, idx2word, idx2tag = create_mappings(sentences, labels)

print(f"Vocabulary size: {len(word2idx)}")
print(f"Tag vocabulary size: {len(tag2idx)}")
print(f"Tags: {list(tag2idx.keys())}")

# Convert sentences and labels to indices
def sentences_to_indices(sentences, labels, word2idx, tag2idx):
    """Convert sentences and labels to numerical indices"""
    X = []
    y = []
    
    for sentence, label_seq in zip(sentences, labels):
        sentence_indices = [word2idx.get(word, word2idx['<UNK>']) for word in sentence]
        label_indices = [tag2idx[tag] for tag in label_seq]
        
        X.append(sentence_indices)
        y.append(label_indices)
    
    return X, y

X, y = sentences_to_indices(sentences, labels, word2idx, tag2idx)

# Split dataset into train, validation, and test sets
print("\n=== Dataset Splitting ===")
print("Splitting dataset: 60% train, 20% validation, 20% test")

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# Second split: separate train and validation from remaining 80%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=None  # 0.25 * 0.8 = 0.2 of total
)

print(f"Training set: {len(X_train)} sentences")
print(f"Validation set: {len(X_val)} sentences")
print(f"Test set: {len(X_test)} sentences")

# Padding sequences for consistent length
def pad_sequences(sequences, max_length=None, pad_value=0):
    """Pad sequences to the same length"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded_sequences.append(seq[:max_length])
        else:
            padded_sequences.append(seq + [pad_value] * (max_length - len(seq)))
    
    return np.array(padded_sequences), max_length

# Find maximum sequence length
max_len = max(len(seq) for seq in X)
print(f"\nMaximum sequence length: {max_len}")

# Pad all sequences
X_train_padded, _ = pad_sequences(X_train, max_len, word2idx['<PAD>'])
X_val_padded, _ = pad_sequences(X_val, max_len, word2idx['<PAD>'])
X_test_padded, _ = pad_sequences(X_test, max_len, word2idx['<PAD>'])

y_train_padded, _ = pad_sequences(y_train, max_len, tag2idx['O'])  # Pad with 'O' tag
y_val_padded, _ = pad_sequences(y_val, max_len, tag2idx['O'])
y_test_padded, _ = pad_sequences(y_test, max_len, tag2idx['O'])

print(f"Padded sequences shape:")
print(f"- X_train: {X_train_padded.shape}")
print(f"- X_val: {X_val_padded.shape}")
print(f"- X_test: {X_test_padded.shape}")

# Save preprocessed data and mappings
print("\n=== Saving Preprocessed Data ===")

# Save mappings
mappings = {
    'word2idx': word2idx,
    'tag2idx': tag2idx,
    'idx2word': idx2word,
    'idx2tag': idx2tag,
    'max_length': max_len,
    'vocab_size': len(word2idx),
    'num_tags': len(tag2idx)
}

with open('mappings.json', 'w') as f:
    json.dump(mappings, f, indent=2)

# Save datasets
np.save('X_train.npy', X_train_padded)
np.save('X_val.npy', X_val_padded)
np.save('X_test.npy', X_test_padded)
np.save('y_train.npy', y_train_padded)
np.save('y_val.npy', y_val_padded)
np.save('y_test.npy', y_test_padded)

# Save original sentences for evaluation
with open('sentences_train.pkl', 'wb') as f:
    pickle.dump([sentences[i] for i in range(len(X_train))], f)
with open('sentences_val.pkl', 'wb') as f:
    pickle.dump([sentences[i] for i in range(len(X_train), len(X_train) + len(X_val))], f)
with open('sentences_test.pkl', 'wb') as f:
    pickle.dump([sentences[i] for i in range(len(X_train) + len(X_val), len(sentences))], f)

print("Data preprocessing completed successfully!")
print("Files saved:")
print("- mappings.json: Vocabulary and tag mappings")
print("- X_train.npy, X_val.npy, X_test.npy: Input sequences")
print("- y_train.npy, y_val.npy, y_test.npy: Label sequences")
print("- sentences_*.pkl: Original sentences for evaluation")

# Display sample preprocessed data
print(f"\n=== Sample Preprocessed Data ===")
print("Sample training sentence (indices):", X_train_padded[0])
print("Sample training labels (indices):", y_train_padded[0])
print("Sample training sentence (words):", [idx2word[idx] for idx in X_train_padded[0] if idx != word2idx['<PAD>']])
print("Sample training labels (tags):", [idx2tag[idx] for idx in y_train_padded[0] if idx != tag2idx['O'] or X_train_padded[0][list(y_train_padded[0]).index(idx)] != word2idx['<PAD>']])

print("\nReady for baseline model development!")
