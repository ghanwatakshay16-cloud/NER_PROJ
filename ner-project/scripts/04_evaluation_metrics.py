import numpy as np
import json
import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("=== NER Model Evaluation Metrics ===")

def load_data():
    """Load preprocessed data and mappings"""
    try:
        with open('mappings.json', 'r') as f:
            mappings = json.load(f)
        
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        
        return mappings, X_test, y_test
    except FileNotFoundError:
        print("Required files not found. Please run preprocessing first.")
        return None, None, None

def extract_entities_from_tags(words, tags):
    """Extract entities from IOB2 tagged sequence"""
    entities = []
    current_entity = []
    current_type = None
    start_idx = None
    
    for i, (word, tag) in enumerate(zip(words, tags)):
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'type': current_type,
                    'start': start_idx,
                    'end': i - 1
                })
            # Start new entity
            current_entity = [word]
            current_type = tag[2:]
            start_idx = i
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            # Continue current entity
            current_entity.append(word)
        else:
            # End current entity if exists
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'type': current_type,
                    'start': start_idx,
                    'end': i - 1
                })
                current_entity = []
                current_type = None
                start_idx = None
    
    # Add last entity if exists
    if current_entity:
        entities.append({
            'text': ' '.join(current_entity),
            'type': current_type,
            'start': start_idx,
            'end': len(words) - 1
        })
    
    return entities

class NEREvaluator:
    """Comprehensive NER evaluation metrics"""
    
    def __init__(self, idx2word, idx2tag):
        self.idx2word = {int(k): v for k, v in idx2word.items()}
        self.idx2tag = {int(k): v for k, v in idx2tag.items()}
    
    def sequences_to_words_tags(self, X, y):
        """Convert padded sequences back to words and tags"""
        sentences = []
        tag_sequences = []
        
        for x_seq, y_seq in zip(X, y):
            words = []
            tags = []
            
            for word_idx, tag_idx in zip(x_seq, y_seq):
                word = self.idx2word.get(word_idx, '<UNK>')
                tag = self.idx2tag.get(tag_idx, 'O')
                
                # Skip padding
                if word != '<PAD>':
                    words.append(word)
                    tags.append(tag)
            
            if words:  # Only add non-empty sentences
                sentences.append(words)
                tag_sequences.append(tags)
        
        return sentences, tag_sequences
    
    def token_level_metrics(self, true_tags, pred_tags):
        """Calculate token-level precision, recall, F1"""
        # Flatten all tags
        true_flat = [tag for seq in true_tags for tag in seq]
        pred_flat = [tag for seq in pred_tags for tag in seq]
        
        # Ensure equal lengths
        min_len = min(len(true_flat), len(pred_flat))
        true_flat = true_flat[:min_len]
        pred_flat = pred_flat[:min_len]
        
        # Calculate metrics
        accuracy = accuracy_score(true_flat, pred_flat)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_flat, pred_flat, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        unique_tags = sorted(set(true_flat + pred_flat))
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(true_flat, pred_flat, labels=unique_tags, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': {
                tag: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1': per_class_f1[i],
                    'support': per_class_support[i]
                }
                for i, tag in enumerate(unique_tags)
            }
        }
    
    def entity_level_metrics(self, true_sentences, true_tags, pred_tags):
        """Calculate entity-level precision, recall, F1"""
        true_entities_all = []
        pred_entities_all = []
        
        for sentence, true_seq, pred_seq in zip(true_sentences, true_tags, pred_tags):
            # Ensure equal lengths
            min_len = min(len(sentence), len(true_seq), len(pred_seq))
            sentence = sentence[:min_len]
            true_seq = true_seq[:min_len]
            pred_seq = pred_seq[:min_len]
            
            true_entities = extract_entities_from_tags(sentence, true_seq)
            pred_entities = extract_entities_from_tags(sentence, pred_seq)
            
            # Convert to comparable format (text, type)
            true_entities_set = {(ent['text'], ent['type']) for ent in true_entities}
            pred_entities_set = {(ent['text'], ent['type']) for ent in pred_entities}
            
            true_entities_all.extend(true_entities_set)
            pred_entities_all.extend(pred_entities_set)
        
        # Calculate metrics
        true_entities_set = set(true_entities_all)
        pred_entities_set = set(pred_entities_all)
        
        correct_entities = true_entities_set.intersection(pred_entities_set)
        
        precision = len(correct_entities) / len(pred_entities_set) if pred_entities_set else 0
        recall = len(correct_entities) / len(true_entities_set) if true_entities_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_entities': len(true_entities_set),
            'pred_entities': len(pred_entities_set),
            'correct_entities': len(correct_entities)
        }
    
    def entity_type_metrics(self, true_sentences, true_tags, pred_tags):
        """Calculate metrics per entity type"""
        entity_type_stats = defaultdict(lambda: {'true': 0, 'pred': 0, 'correct': 0})
        
        for sentence, true_seq, pred_seq in zip(true_sentences, true_tags, pred_tags):
            min_len = min(len(sentence), len(true_seq), len(pred_seq))
            sentence = sentence[:min_len]
            true_seq = true_seq[:min_len]
            pred_seq = pred_seq[:min_len]
            
            true_entities = extract_entities_from_tags(sentence, true_seq)
            pred_entities = extract_entities_from_tags(sentence, pred_seq)
            
            # Count by entity type
            for entity in true_entities:
                entity_type_stats[entity['type']]['true'] += 1
            
            for entity in pred_entities:
                entity_type_stats[entity['type']]['pred'] += 1
            
            # Count correct entities
            true_entities_set = {(ent['text'], ent['type']) for ent in true_entities}
            pred_entities_set = {(ent['text'], ent['type']) for ent in pred_entities}
            correct_entities = true_entities_set.intersection(pred_entities_set)
            
            for text, ent_type in correct_entities:
                entity_type_stats[ent_type]['correct'] += 1
        
        # Calculate metrics per type
        type_metrics = {}
        for ent_type, stats in entity_type_stats.items():
            precision = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
            recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            type_metrics[ent_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_count': stats['true'],
                'pred_count': stats['pred'],
                'correct_count': stats['correct']
            }
        
        return type_metrics
    
    def evaluate_model_predictions(self, X_test, y_true, y_pred):
        """Comprehensive evaluation of model predictions"""
        # Convert sequences to words and tags
        sentences, true_tags = self.sequences_to_words_tags(X_test, y_true)
        
        # Convert predictions to tag sequences
        pred_tags = []
        pred_idx = 0
        for sentence in sentences:
            sent_pred_tags = []
            for _ in sentence:
                if pred_idx < len(y_pred):
                    sent_pred_tags.append(y_pred[pred_idx])
                    pred_idx += 1
                else:
                    sent_pred_tags.append('O')
            pred_tags.append(sent_pred_tags)
        
        # Calculate all metrics
        token_metrics = self.token_level_metrics(true_tags, pred_tags)
        entity_metrics = self.entity_level_metrics(sentences, true_tags, pred_tags)
        type_metrics = self.entity_type_metrics(sentences, true_tags, pred_tags)
        
        return {
            'token_level': token_metrics,
            'entity_level': entity_metrics,
            'entity_type': type_metrics,
            'sentences': sentences,
            'true_tags': true_tags,
            'pred_tags': pred_tags
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Load data
    mappings, X_test, y_test = load_data()
    if mappings is None:
        print("Cannot proceed without data. Please run preprocessing first.")
        exit()
    
    # Initialize evaluator
    evaluator = NEREvaluator(mappings['idx2word'], mappings['idx2tag'])
    
    # Load baseline model predictions (if available)
    try:
        with open('baseline_model.pkl', 'rb') as f:
            baseline_model = pickle.load(f)
        
        print("Evaluating baseline model on test set...")
        test_predictions = baseline_model.predict(X_test)
        
        # Comprehensive evaluation
        results = evaluator.evaluate_model_predictions(X_test, y_test, test_predictions)
        
        print("\n=== TOKEN-LEVEL METRICS ===")
        token_metrics = results['token_level']
        print(f"Accuracy: {token_metrics['accuracy']:.4f}")
        print(f"Precision: {token_metrics['precision']:.4f}")
        print(f"Recall: {token_metrics['recall']:.4f}")
        print(f"F1-Score: {token_metrics['f1']:.4f}")
        
        print("\nPer-class metrics:")
        for tag, metrics in token_metrics['per_class'].items():
            print(f"{tag:10} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, "
                  f"F1: {metrics['f1']:.3f}, Support: {metrics['support']}")
        
        print("\n=== ENTITY-LEVEL METRICS ===")
        entity_metrics = results['entity_level']
        print(f"Precision: {entity_metrics['precision']:.4f}")
        print(f"Recall: {entity_metrics['recall']:.4f}")
        print(f"F1-Score: {entity_metrics['f1']:.4f}")
        print(f"True entities: {entity_metrics['true_entities']}")
        print(f"Predicted entities: {entity_metrics['pred_entities']}")
        print(f"Correct entities: {entity_metrics['correct_entities']}")
        
        print("\n=== ENTITY TYPE METRICS ===")
        for ent_type, metrics in results['entity_type'].items():
            print(f"{ent_type:10} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, "
                  f"F1: {metrics['f1']:.3f} (True: {metrics['true_count']}, "
                  f"Pred: {metrics['pred_count']}, Correct: {metrics['correct_count']})")
        
        # Visualize results
        plt.figure(figsize=(15, 10))
        
        # Token-level metrics by tag
        plt.subplot(2, 3, 1)
        tags = list(token_metrics['per_class'].keys())
        f1_scores = [token_metrics['per_class'][tag]['f1'] for tag in tags]
        plt.bar(tags, f1_scores)
        plt.title('Token-level F1 Scores by Tag')
        plt.xticks(rotation=45)
        plt.ylabel('F1 Score')
        
        # Entity-level metrics
        plt.subplot(2, 3, 2)
        entity_metrics_values = [entity_metrics['precision'], entity_metrics['recall'], entity_metrics['f1']]
        plt.bar(['Precision', 'Recall', 'F1'], entity_metrics_values)
        plt.title('Entity-level Metrics')
        plt.ylabel('Score')
        
        # Entity type F1 scores
        plt.subplot(2, 3, 3)
        if results['entity_type']:
            types = list(results['entity_type'].keys())
            type_f1s = [results['entity_type'][t]['f1'] for t in types]
            plt.bar(types, type_f1s)
            plt.title('F1 Scores by Entity Type')
            plt.xticks(rotation=45)
            plt.ylabel('F1 Score')
        
        # Sample predictions visualization
        plt.subplot(2, 1, 2)
        sample_idx = 0
        if sample_idx < len(results['sentences']):
            sample_sentence = results['sentences'][sample_idx]
            sample_true = results['true_tags'][sample_idx]
            sample_pred = results['pred_tags'][sample_idx]
            
            x_pos = range(len(sample_sentence))
            plt.scatter(x_pos, [0] * len(sample_sentence), c='blue', label='Words', s=100)
            
            for i, (word, true_tag, pred_tag) in enumerate(zip(sample_sentence, sample_true, sample_pred)):
                color = 'green' if true_tag == pred_tag else 'red'
                plt.annotate(f'{word}\nT:{true_tag}\nP:{pred_tag}', 
                           (i, 0), xytext=(0, 20), textcoords='offset points',
                           ha='center', fontsize=8, color=color)
            
            plt.title('Sample Prediction (Green=Correct, Red=Incorrect)')
            plt.ylim(-0.5, 0.5)
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nEvaluation completed! Metrics visualization saved as 'evaluation_metrics.png'")
        
    except FileNotFoundError:
        print("Baseline model not found. Please train the baseline model first.")
        
        # Demonstrate evaluation framework with dummy predictions
        print("\nDemonstrating evaluation framework with sample data...")
        
        # Create sample predictions (all 'O' tags)
        sentences, true_tags = evaluator.sequences_to_words_tags(X_test, y_test)
        dummy_predictions = ['O'] * sum(len(sent) for sent in sentences)
        
        results = evaluator.evaluate_model_predictions(X_test, y_test, dummy_predictions)
        
        print("Sample evaluation results (dummy predictions):")
        print(f"Token-level F1: {results['token_level']['f1']:.4f}")
        print(f"Entity-level F1: {results['entity_level']['f1']:.4f}")

print("\nEvaluation metrics framework ready!")
