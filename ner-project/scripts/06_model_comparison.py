import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import pandas as pd

print("=== Comprehensive Model Comparison ===")

# Load data and mappings
try:
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)
    
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    word2idx = mappings['word2idx']
    tag2idx = mappings['tag2idx']
    idx2word = {int(k): v for k, v in mappings['idx2word'].items()}
    idx2tag = {int(k): v for k, v in mappings['idx2tag'].items()}
    
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Required files not found. Please run preprocessing first.")
    exit()

def extract_entities_from_sequence(words, tags):
    """Extract entities from word-tag sequence"""
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
    
    return entities

def sequences_to_flat_tags(X, y, idx2word, idx2tag):
    """Convert padded sequences to flat tag lists"""
    flat_tags = []
    flat_words = []
    
    for x_seq, y_seq in zip(X, y):
        for word_idx, tag_idx in zip(x_seq, y_seq):
            word = idx2word.get(word_idx, '<PAD>')
            tag = idx2tag.get(tag_idx, 'O')
            
            if word != '<PAD>':
                flat_words.append(word)
                flat_tags.append(tag)
    
    return flat_words, flat_tags

def calculate_entity_metrics(true_entities, pred_entities):
    """Calculate entity-level precision, recall, F1"""
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    correct = true_set.intersection(pred_set)
    
    precision = len(correct) / len(pred_set) if pred_set else 0
    recall = len(correct) / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

class ModelComparator:
    """Compare different NER models"""
    
    def __init__(self, idx2word, idx2tag):
        self.idx2word = idx2word
        self.idx2tag = idx2tag
        self.results = {}
    
    def evaluate_baseline_model(self, X_test, y_test):
        """Evaluate baseline model"""
        try:
            with open('baseline_model.pkl', 'rb') as f:
                baseline_model = pickle.load(f)
            
            print("Evaluating baseline model...")
            predictions = baseline_model.predict(X_test)
            
            # Convert test data to flat format
            true_words, true_tags = sequences_to_flat_tags(X_test, y_test, self.idx2word, self.idx2tag)
            
            # Ensure equal lengths
            min_len = min(len(predictions), len(true_tags))
            predictions = predictions[:min_len]
            true_tags = true_tags[:min_len]
            true_words = true_words[:min_len]
            
            # Calculate metrics
            token_f1 = f1_score(true_tags, predictions, average='weighted', zero_division=0)
            token_precision = precision_score(true_tags, predictions, average='weighted', zero_division=0)
            token_recall = recall_score(true_tags, predictions, average='weighted', zero_division=0)
            
            # Entity-level metrics
            true_entities = extract_entities_from_sequence(true_words, true_tags)
            pred_entities = extract_entities_from_sequence(true_words, predictions)
            entity_precision, entity_recall, entity_f1 = calculate_entity_metrics(true_entities, pred_entities)
            
            self.results['baseline'] = {
                'token_precision': token_precision,
                'token_recall': token_recall,
                'token_f1': token_f1,
                'entity_precision': entity_precision,
                'entity_recall': entity_recall,
                'entity_f1': entity_f1,
                'predictions': predictions,
                'true_tags': true_tags,
                'model_type': 'Logistic Regression'
            }
            
            print(f"Baseline - Token F1: {token_f1:.4f}, Entity F1: {entity_f1:.4f}")
            return True
            
        except FileNotFoundError:
            print("Baseline model not found.")
            return False
    
    def evaluate_improved_model(self, X_test, y_test):
        """Evaluate improved model"""
        try:
            import tensorflow as tf
            improved_model = tf.keras.models.load_model('improved_ner_model.h5')
            
            print("Evaluating improved model...")
            predictions = improved_model.predict(X_test, verbose=0)
            predictions = np.argmax(predictions, axis=-1)
            
            # Convert to flat format
            true_words, true_tags = sequences_to_flat_tags(X_test, y_test, self.idx2word, self.idx2tag)
            
            # Flatten predictions
            pred_tags = []
            for i, pred_seq in enumerate(predictions):
                for j, pred_idx in enumerate(pred_seq):
                    if j < len(X_test[i]) and self.idx2word.get(X_test[i][j], '') != '<PAD>':
                        pred_tags.append(self.idx2tag[pred_idx])
            
            # Ensure equal lengths
            min_len = min(len(pred_tags), len(true_tags))
            pred_tags = pred_tags[:min_len]
            true_tags = true_tags[:min_len]
            true_words = true_words[:min_len]
            
            # Calculate metrics
            token_f1 = f1_score(true_tags, pred_tags, average='weighted', zero_division=0)
            token_precision = precision_score(true_tags, pred_tags, average='weighted', zero_division=0)
            token_recall = recall_score(true_tags, pred_tags, average='weighted', zero_division=0)
            
            # Entity-level metrics
            true_entities = extract_entities_from_sequence(true_words, true_tags)
            pred_entities = extract_entities_from_sequence(true_words, pred_tags)
            entity_precision, entity_recall, entity_f1 = calculate_entity_metrics(true_entities, pred_entities)
            
            self.results['improved'] = {
                'token_precision': token_precision,
                'token_recall': token_recall,
                'token_f1': token_f1,
                'entity_precision': entity_precision,
                'entity_recall': entity_recall,
                'entity_f1': entity_f1,
                'predictions': pred_tags,
                'true_tags': true_tags,
                'model_type': 'BiLSTM'
            }
            
            print(f"Improved - Token F1: {token_f1:.4f}, Entity F1: {entity_f1:.4f}")
            return True
            
        except (FileNotFoundError, ImportError) as e:
            print(f"Improved model not found or TensorFlow not available: {e}")
            return False
    
    def create_comparison_report(self):
        """Create comprehensive comparison report"""
        if not self.results:
            print("No models to compare. Please evaluate models first.")
            return
        
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.title(),
                'Type': metrics['model_type'],
                'Token Precision': f"{metrics['token_precision']:.4f}",
                'Token Recall': f"{metrics['token_recall']:.4f}",
                'Token F1': f"{metrics['token_f1']:.4f}",
                'Entity Precision': f"{metrics['entity_precision']:.4f}",
                'Entity Recall': f"{metrics['entity_recall']:.4f}",
                'Entity F1': f"{metrics['entity_f1']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(df.to_string(index=False))
        
        # Calculate improvements
        if 'baseline' in self.results and 'improved' in self.results:
            baseline = self.results['baseline']
            improved = self.results['improved']
            
            print(f"\n" + "="*40)
            print("IMPROVEMENT ANALYSIS")
            print("="*40)
            
            improvements = {
                'Token F1': improved['token_f1'] - baseline['token_f1'],
                'Token Precision': improved['token_precision'] - baseline['token_precision'],
                'Token Recall': improved['token_recall'] - baseline['token_recall'],
                'Entity F1': improved['entity_f1'] - baseline['entity_f1'],
                'Entity Precision': improved['entity_precision'] - baseline['entity_precision'],
                'Entity Recall': improved['entity_recall'] - baseline['entity_recall']
            }
            
            for metric, improvement in improvements.items():
                status = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                print(f"{metric:20}: {improvement:+.4f} {status}")
        
        return df
    
    def plot_comparison(self):
        """Create visualization comparing models"""
        if len(self.results) < 2:
            print("Need at least 2 models for comparison plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Token-level metrics comparison
        models = list(self.results.keys())
        token_metrics = ['token_precision', 'token_recall', 'token_f1']
        
        token_data = []
        for model in models:
            for metric in token_metrics:
                token_data.append({
                    'Model': model.title(),
                    'Metric': metric.replace('token_', '').title(),
                    'Score': self.results[model][metric]
                })
        
        token_df = pd.DataFrame(token_data)
        sns.barplot(data=token_df, x='Metric', y='Score', hue='Model', ax=axes[0, 0])
        axes[0, 0].set_title('Token-Level Metrics Comparison')
        axes[0, 0].set_ylim(0, 1)
        
        # Entity-level metrics comparison
        entity_metrics = ['entity_precision', 'entity_recall', 'entity_f1']
        
        entity_data = []
        for model in models:
            for metric in entity_metrics:
                entity_data.append({
                    'Model': model.title(),
                    'Metric': metric.replace('entity_', '').title(),
                    'Score': self.results[model][metric]
                })
        
        entity_df = pd.DataFrame(entity_data)
        sns.barplot(data=entity_df, x='Metric', y='Score', hue='Model', ax=axes[0, 1])
        axes[0, 1].set_title('Entity-Level Metrics Comparison')
        axes[0, 1].set_ylim(0, 1)
        
        # F1 Score comparison
        f1_data = []
        for model in models:
            f1_data.append({
                'Model': model.title(),
                'Token F1': self.results[model]['token_f1'],
                'Entity F1': self.results[model]['entity_f1']
            })
        
        f1_df = pd.DataFrame(f1_data)
        f1_df.set_index('Model').plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Tag distribution comparison (if available)
        if 'baseline' in self.results and 'improved' in self.results:
            baseline_tags = Counter(self.results['baseline']['predictions'])
            improved_tags = Counter(self.results['improved']['predictions'])
            
            all_tags = set(baseline_tags.keys()) | set(improved_tags.keys())
            tag_comparison = []
            
            for tag in all_tags:
                tag_comparison.append({
                    'Tag': tag,
                    'Baseline': baseline_tags.get(tag, 0),
                    'Improved': improved_tags.get(tag, 0)
                })
            
            tag_df = pd.DataFrame(tag_comparison)
            tag_df.set_index('Tag').plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Tag Prediction Distribution')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_error_patterns(self):
        """Analyze common error patterns"""
        if 'baseline' not in self.results or 'improved' not in self.results:
            print("Need both baseline and improved model results for error analysis.")
            return
        
        print(f"\n" + "="*50)
        print("ERROR PATTERN ANALYSIS")
        print("="*50)
        
        baseline_pred = self.results['baseline']['predictions']
        improved_pred = self.results['improved']['predictions']
        true_tags = self.results['baseline']['true_tags']
        
        # Find common errors
        baseline_errors = [(true, pred) for true, pred in zip(true_tags, baseline_pred) if true != pred]
        improved_errors = [(true, pred) for true, pred in zip(true_tags, improved_pred) if true != pred]
        
        print(f"Baseline errors: {len(baseline_errors)}")
        print(f"Improved errors: {len(improved_errors)}")
        print(f"Error reduction: {len(baseline_errors) - len(improved_errors)} ({((len(baseline_errors) - len(improved_errors)) / len(baseline_errors) * 100):.1f}%)")
        
        # Most common error types
        baseline_error_types = Counter(baseline_errors)
        improved_error_types = Counter(improved_errors)
        
        print(f"\nTop 5 error types in baseline:")
        for (true_tag, pred_tag), count in baseline_error_types.most_common(5):
            print(f"  {true_tag} -> {pred_tag}: {count} times")
        
        print(f"\nTop 5 error types in improved model:")
        for (true_tag, pred_tag), count in improved_error_types.most_common(5):
            print(f"  {true_tag} -> {pred_tag}: {count} times")

# Run comprehensive comparison
comparator = ModelComparator(idx2word, idx2tag)

# Evaluate both models
baseline_available = comparator.evaluate_baseline_model(X_test, y_test)
improved_available = comparator.evaluate_improved_model(X_test, y_test)

if baseline_available or improved_available:
    # Create comparison report
    comparison_df = comparator.create_comparison_report()
    
    # Create visualizations
    comparator.plot_comparison()
    
    # Analyze error patterns
    comparator.analyze_error_patterns()
    
    # Save comparison results
    if comparison_df is not None:
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\nComparison results saved to 'model_comparison_results.csv'")
    
    print(f"\nModel comparison completed!")
    print("Visualizations saved as 'model_comparison.png'")
else:
    print("No models available for comparison.")
