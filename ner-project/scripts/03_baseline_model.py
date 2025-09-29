import numpy as np
import json
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

print("=== Baseline NER Model Development ===")

# Load preprocessed data and mappings
print("Loading preprocessed data...")
try:
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)
    
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')
    
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Preprocessed data not found. Please run the preprocessing script first.")
    exit()

word2idx = mappings['word2idx']
tag2idx = mappings['tag2idx']
idx2word = {int(k): v for k, v in mappings['idx2word'].items()}
idx2tag = {int(k): v for k, v in mappings['idx2tag'].items()}
max_length = mappings['max_length']

print(f"Dataset shapes:")
print(f"- Training: {X_train.shape}, {y_train.shape}")
print(f"- Validation: {X_val.shape}, {y_val.shape}")
print(f"- Test: {X_test.shape}, {y_test.shape}")

class BaselineNERModel:
    """
    Baseline NER Model using simple features and Logistic Regression
    Features include:
    - Current word
    - Word shape (capitalization pattern)
    - Word length
    - Previous and next word context
    - Position in sentence
    """
    
    def __init__(self):
        self.vectorizer = DictVectorizer()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.fitted = False
    
    def word_shape(self, word):
        """Extract word shape features"""
        if word == '<PAD>':
            return 'PAD'
        elif word == '<UNK>':
            return 'UNK'
        
        shape = ''
        for char in word:
            if char.isupper():
                shape += 'X'
            elif char.islower():
                shape += 'x'
            elif char.isdigit():
                shape += 'd'
            else:
                shape += char
        return shape
    
    def extract_features(self, sentences, positions=None):
        """Extract features for each word in sentences"""
        features = []
        
        for sent_idx, sentence in enumerate(sentences):
            for word_idx, word_id in enumerate(sentence):
                word = idx2word.get(word_id, '<UNK>')
                
                # Skip padding tokens
                if word == '<PAD>':
                    continue
                
                feature_dict = {}
                
                # Current word features
                feature_dict['word'] = word.lower()
                feature_dict['word_shape'] = self.word_shape(word)
                feature_dict['word_length'] = len(word)
                feature_dict['is_capitalized'] = word[0].isupper() if word else False
                feature_dict['is_all_caps'] = word.isupper() if word else False
                feature_dict['has_digit'] = any(c.isdigit() for c in word)
                feature_dict['has_hyphen'] = '-' in word
                feature_dict['has_punctuation'] = any(c in '.,!?;:' for c in word)
                
                # Position features
                feature_dict['position_in_sentence'] = word_idx
                feature_dict['is_first_word'] = word_idx == 0
                feature_dict['is_last_word'] = word_idx == len([w for w in sentence if idx2word.get(w, '') != '<PAD>']) - 1
                
                # Context features (previous and next words)
                if word_idx > 0:
                    prev_word = idx2word.get(sentence[word_idx - 1], '<UNK>')
                    if prev_word != '<PAD>':
                        feature_dict['prev_word'] = prev_word.lower()
                        feature_dict['prev_word_shape'] = self.word_shape(prev_word)
                
                if word_idx < len(sentence) - 1:
                    next_word = idx2word.get(sentence[word_idx + 1], '<UNK>')
                    if next_word != '<PAD>':
                        feature_dict['next_word'] = next_word.lower()
                        feature_dict['next_word_shape'] = self.word_shape(next_word)
                
                # Prefix and suffix features
                if len(word) >= 2:
                    feature_dict['prefix_2'] = word[:2].lower()
                    feature_dict['suffix_2'] = word[-2:].lower()
                if len(word) >= 3:
                    feature_dict['prefix_3'] = word[:3].lower()
                    feature_dict['suffix_3'] = word[-3:].lower()
                
                features.append(feature_dict)
        
        return features
    
    def prepare_labels(self, label_sequences):
        """Flatten label sequences, excluding padding"""
        labels = []
        for sequence in label_sequences:
            for label_id in sequence:
                label = idx2tag.get(label_id, 'O')
                if label != 'O' or len(labels) < len([f for f in self.extract_features([sequence]) if f]):
                    labels.append(label)
        return labels
    
    def fit(self, X_train, y_train):
        """Train the baseline model"""
        print("Extracting features from training data...")
        train_features = self.extract_features(X_train)
        
        print("Preparing training labels...")
        train_labels = []
        for sequence in y_train:
            for label_id in sequence:
                if idx2word.get(X_train[len(train_labels) // max_length][len(train_labels) % max_length], '') != '<PAD>':
                    train_labels.append(idx2tag.get(label_id, 'O'))
                if len(train_labels) >= len(train_features):
                    break
            if len(train_labels) >= len(train_features):
                break
        
        # Ensure equal lengths
        min_len = min(len(train_features), len(train_labels))
        train_features = train_features[:min_len]
        train_labels = train_labels[:min_len]
        
        print(f"Training on {len(train_features)} samples...")
        
        # Vectorize features
        X_train_vec = self.vectorizer.fit_transform(train_features)
        
        # Train model
        self.model.fit(X_train_vec, train_labels)
        self.fitted = True
        
        print("Baseline model training completed!")
    
    def predict(self, X):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        features = self.extract_features(X)
        X_vec = self.vectorizer.transform(features)
        predictions = self.model.predict(X_vec)
        
        return predictions
    
    def predict_sentence(self, sentence_words):
        """Predict NER tags for a single sentence"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert words to indices
        sentence_indices = [word2idx.get(word, word2idx['<UNK>']) for word in sentence_words]
        
        # Pad to max length
        if len(sentence_indices) < max_length:
            sentence_indices.extend([word2idx['<PAD>']] * (max_length - len(sentence_indices)))
        else:
            sentence_indices = sentence_indices[:max_length]
        
        # Extract features
        features = self.extract_features([sentence_indices])
        
        # Make predictions
        if features:
            X_vec = self.vectorizer.transform(features)
            predictions = self.model.predict(X_vec)
            return predictions[:len(sentence_words)]
        else:
            return ['O'] * len(sentence_words)

# Initialize and train baseline model
print("\n=== Training Baseline Model ===")
baseline_model = BaselineNERModel()
baseline_model.fit(X_train, y_train)

# Make predictions on validation set
print("\n=== Evaluating on Validation Set ===")
val_predictions = baseline_model.predict(X_val)

# Prepare validation labels for evaluation
val_labels = []
for sequence in y_val:
    for label_id in sequence:
        if len(val_labels) < len(val_predictions):
            val_labels.append(idx2tag.get(label_id, 'O'))

# Ensure equal lengths
min_len = min(len(val_predictions), len(val_labels))
val_predictions = val_predictions[:min_len]
val_labels = val_labels[:min_len]

# Calculate metrics
print("Validation Results:")
print(classification_report(val_labels, val_predictions, zero_division=0))

# Test on sample sentences
print("\n=== Testing on Sample Sentences ===")
test_sentences = [
    ["Today", "John", "Smith", "visited", "New", "York"],
    ["Apple", "Inc.", "is", "located", "in", "California"],
    ["The", "meeting", "with", "Microsoft", "is", "tomorrow"]
]

for sentence in test_sentences:
    predictions = baseline_model.predict_sentence(sentence)
    print(f"\nSentence: {' '.join(sentence)}")
    print("Predictions:")
    for word, tag in zip(sentence, predictions):
        print(f"  {word:15} -> {tag}")

# Analyze model performance
print("\n=== Baseline Model Analysis ===")

# Tag distribution in predictions
pred_tag_dist = Counter(val_predictions)
true_tag_dist = Counter(val_labels)

print("Tag distribution comparison:")
print("Tag\t\tTrue\tPredicted")
for tag in sorted(set(val_labels + val_predictions)):
    print(f"{tag:10}\t{true_tag_dist[tag]:4d}\t{pred_tag_dist[tag]:4d}")

# Confusion matrix for main entity types
entity_tags = [tag for tag in set(val_labels) if tag != 'O']
if entity_tags:
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(val_labels, val_predictions, labels=sorted(set(val_labels)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(val_labels)), 
                yticklabels=sorted(set(val_labels)))
    plt.title('Confusion Matrix - Baseline Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Feature importance analysis
feature_names = baseline_model.vectorizer.get_feature_names_out()
feature_importance = np.abs(baseline_model.model.coef_).mean(axis=0)
top_features_idx = np.argsort(feature_importance)[-20:]

print(f"\nTop 20 Most Important Features:")
for idx in reversed(top_features_idx):
    print(f"{feature_names[idx]:30} -> {feature_importance[idx]:.4f}")

# Save baseline model
print("\n=== Saving Baseline Model ===")
with open('baseline_model.pkl', 'wb') as f:
    pickle.dump(baseline_model, f)

print("Baseline model saved successfully!")

# Identify shortcomings
print("\n=== Baseline Model Shortcomings Analysis ===")
print("1. CONTEXT LIMITATION:")
print("   - Only considers immediate neighbors (prev/next word)")
print("   - Cannot capture long-range dependencies")
print("   - No understanding of sentence-level context")

print("\n2. FEATURE ENGINEERING DEPENDENCY:")
print("   - Relies on hand-crafted features")
print("   - May miss important patterns in the data")
print("   - Limited ability to learn complex word representations")

print("\n3. SEQUENCE MODELING:")
print("   - Treats each word independently")
print("   - Doesn't model the sequential nature of NER tags")
print("   - No consideration of tag transition probabilities")

print("\n4. SCALABILITY ISSUES:")
print("   - Feature extraction can be computationally expensive")
print("   - Vocabulary growth leads to sparse feature vectors")
print("   - Limited generalization to unseen words")

print("\n5. IOB2 CONSISTENCY:")
print("   - No enforcement of valid IOB2 tag sequences")
print("   - May predict invalid tag transitions (e.g., I-PER after O)")

# Calculate entity-level metrics
def extract_entities(words, tags):
    """Extract entities from IOB2 tagged sequence"""
    entities = []
    current_entity = []
    current_type = None
    
    for word, tag in zip(words, tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = [word]
            current_type = tag[2:]
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
                current_entity = []
                current_type = None
    
    if current_entity:
        entities.append((' '.join(current_entity), current_type))
    
    return set(entities)

# Entity-level evaluation on sample
print("\n=== Entity-Level Evaluation Sample ===")
sample_sentence = ["Today", "Michael", "Jackson", "visited", "New", "York"]
sample_true_tags = ["O", "B-PER", "I-PER", "O", "B-geo", "I-geo"]
sample_pred_tags = baseline_model.predict_sentence(sample_sentence)

true_entities = extract_entities(sample_sentence, sample_true_tags)
pred_entities = extract_entities(sample_sentence, sample_pred_tags)

print(f"Sample sentence: {' '.join(sample_sentence)}")
print(f"True entities: {true_entities}")
print(f"Predicted entities: {pred_entities}")
print(f"Correct entities: {true_entities.intersection(pred_entities)}")

print("\nBaseline model development completed!")
print("Ready for improved model development to address identified shortcomings.")
