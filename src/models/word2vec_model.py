"""
Word2Vec Model Implementation for SE Word Embeddings
Provides baseline comparison for ModernBERT evaluation
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import get_logger
from ..utils.timer import Timer, PerformanceMonitor

class Word2VecTrainingCallback(CallbackAny2Vec):
    """Callback to track Word2Vec training progress"""
    
    def __init__(self, logger):
        self.epoch = 0
        self.logger = logger
        self.start_time = time.time()
    
    def on_epoch_end(self, model):
        elapsed = time.time() - self.start_time
        self.logger.info(f"Word2Vec Epoch {self.epoch + 1} completed in {elapsed:.2f}s")
        self.epoch += 1

class Word2VecModel:
    """
    Word2Vec baseline model for SE word embeddings
    Provides traditional word embedding approach for comparison
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = get_logger(__name__)
        self.performance_monitor = PerformanceMonitor()
        
        # Model configuration
        self.model_config = config.get('models', {}).get('word2vec', {})

        # Model parameters
        self.vector_size = self.model_config.get('vector_size', 100)
        self.window = self.model_config.get('window', 5)
        self.min_count = self.model_config.get('min_count', 1)
        self.workers = self.model_config.get('workers', 4)
        self.epochs = self.model_config.get('epochs', 5)
        self.sg = self.model_config.get('sg', 0)  # 0=CBOW, 1=Skip-gram
        self.hs = self.model_config.get('hs', 0)  # 0=negative sampling, 1=hierarchical softmax
        self.negative = self.model_config.get('negative', 5)
        self.alpha = self.model_config.get('alpha', 0.025)
        self.min_alpha = self.model_config.get('min_alpha', 0.0001)

        # Model components
        self.model = None
        self.vocabulary = None
        self.training_data = []

        # Results storage
        self.training_results = {}
        self.evaluation_results = {}

        self.logger.info(f"Initialized Word2Vec model with {self.vector_size}D vectors")

    def load_training_data(self, data_path: str) -> List[List[str]]:
        """Load and prepare training data"""

        self.logger.info("📚 Loading training data for Word2Vec...")

        # Load processed training data
        train_file = Path(data_path) / 'train.json'

        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")

        with open(train_file, 'r', encoding='utf-8') as f:
            train_docs = json.load(f)

        # Prepare sentences for Word2Vec training
        sentences = []
        for doc in train_docs:
            text = doc.get('text', '')
            if text:
                # Simple tokenization (split by whitespace and clean)
                words = text.lower().split()
                # Remove very short words and non-alphabetic tokens
                words = [word for word in words if len(word) > 2 and word.isalpha()]
                if len(words) > 3:  # Only include sentences with enough words
                    sentences.append(words)

        self.training_data = sentences
        self.logger.info(f"📊 Prepared {len(sentences)} sentences for training")

        return sentences

    def train_model(self, sentences: List[List[str]]) -> Dict[str, Any]:
        """Train Word2Vec model"""

        self.logger.info("🚀 Starting Word2Vec training...")
        start_time = time.time()

        # Create callback for progress tracking
        callback = Word2VecTrainingCallback(self.logger)

        try:
            # Initialize and train Word2Vec model
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                epochs=self.epochs,
                sg=self.sg,
                hs=self.hs,
                negative=self.negative,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                callbacks=[callback]
            )

            training_time = time.time() - start_time

            # Get vocabulary statistics
            vocab_size = len(self.model.wv.key_to_index)
            self.vocabulary = list(self.model.wv.key_to_index.keys())

            self.training_results = {
                'training_time': training_time,
                'vocabulary_size': vocab_size,
                'vector_size': self.vector_size,
                'total_sentences': len(sentences),
                'epochs': self.epochs,
                'model_parameters': {
                    'vector_size': self.vector_size,
                    'window': self.window,
                    'min_count': self.min_count,
                    'sg': self.sg,
                    'hs': self.hs,
                    'negative': self.negative
                }
            }

            self.logger.info(f"✅ Word2Vec training completed in {training_time:.2f}s")
            self.logger.info(f"📊 Vocabulary size: {vocab_size}")
            self.logger.info(f"📊 Vector dimensions: {self.vector_size}")

            return self.training_results

        except Exception as e:
            self.logger.error(f"❌ Word2Vec training failed: {e}")
            raise

    def save_model(self) -> str:
        """Save trained Word2Vec model"""

        if self.model is None:
            raise ValueError("No trained model to save")

        # Create models directory
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / 'word2vec_model.bin'
        self.model.save(str(model_path))

        # Save vocabulary
        vocab_path = models_dir / 'word2vec_vocabulary.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, indent=2, ensure_ascii=False)

        # Save training results
        results_path = models_dir / 'word2vec_training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, indent=2, default=str)

        self.logger.info(f"💾 Word2Vec model saved to {model_path}")

        return str(model_path)

    def evaluate_model(self, data_path: str) -> Dict[str, Any]:
        """Evaluate Word2Vec model on various tasks"""

        self.logger.info("📊 Starting Word2Vec evaluation...")

        if self.model is None:
            raise ValueError("No trained model to evaluate")

        evaluation_results = {}

        # 1. Vocabulary coverage
        evaluation_results['vocabulary_coverage'] = self._evaluate_vocabulary_coverage(data_path)

        # 2. Word similarity examples
        evaluation_results['similarity_examples'] = self._evaluate_similarity_examples()

        # 3. Most similar words for SE terms
        evaluation_results['se_term_similarities'] = self._evaluate_se_term_similarities()

        # 4. Document classification (if enough data)
        try:
            evaluation_results['classification'] = self._evaluate_classification(data_path)
        except Exception as e:
            self.logger.warning(f"Classification evaluation failed: {e}")
            evaluation_results['classification'] = {'error': str(e)}

        self.evaluation_results = evaluation_results

        # Save evaluation results
        eval_dir = self.output_dir / 'evaluations'
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_path = eval_dir / 'word2vec_evaluation.json'
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        self.logger.info("✅ Word2Vec evaluation completed")

        return evaluation_results

    def _evaluate_vocabulary_coverage(self, data_path: str) -> Dict[str, Any]:
        """Evaluate vocabulary coverage on test data"""

        test_file = Path(data_path) / 'test.json'
        if not test_file.exists():
            return {'error': 'Test data not found'}

        with open(test_file, 'r', encoding='utf-8') as f:
            test_docs = json.load(f)

        total_words = 0
        covered_words = 0

        for doc in test_docs:
            text = doc.get('text', '')
            words = text.lower().split()
            words = [word for word in words if len(word) > 2 and word.isalpha()]

            for word in words:
                total_words += 1
                if word in self.model.wv.key_to_index:
                    covered_words += 1

        coverage = covered_words / total_words if total_words > 0 else 0

        return {
            'total_words': total_words,
            'covered_words': covered_words,
            'coverage_ratio': coverage,
            'vocabulary_size': len(self.vocabulary)
        }

    def _evaluate_similarity_examples(self) -> Dict[str, Any]:
        """Evaluate word similarity with examples"""

        examples = {}

        # Test words that should be in SE vocabulary
        test_words = ['software', 'programming', 'algorithm', 'code', 'function', 'data', 'system']

        for word in test_words:
            if word in self.model.wv.key_to_index:
                try:
                    similar_words = self.model.wv.most_similar(word, topn=5)
                    examples[word] = similar_words
                except:
                    examples[word] = 'No similar words found'
            else:
                examples[word] = 'Word not in vocabulary'

        return examples

    def _evaluate_se_term_similarities(self) -> Dict[str, Any]:
        """Evaluate similarities for SE-specific terms"""

        se_terms = {
            'programming_languages': ['python', 'java', 'javascript', 'cpp'],
            'concepts': ['algorithm', 'function', 'class', 'method'],
            'tools': ['git', 'docker', 'database', 'framework']
        }

        similarities = {}

        for category, terms in se_terms.items():
            similarities[category] = {}
            for term in terms:
                if term in self.model.wv.key_to_index:
                    try:
                        similar = self.model.wv.most_similar(term, topn=3)
                        similarities[category][term] = similar
                    except:
                        similarities[category][term] = 'No similarities found'
                else:
                    similarities[category][term] = 'Not in vocabulary'

        return similarities

    def _evaluate_classification(self, data_path: str) -> Dict[str, Any]:
        """Evaluate document classification using Word2Vec embeddings"""

        # Load train and test data
        train_file = Path(data_path) / 'train.json'
        test_file = Path(data_path) / 'test.json'

        if not train_file.exists() or not test_file.exists():
            return {'error': 'Training or test data not found'}

        with open(train_file, 'r', encoding='utf-8') as f:
            train_docs = json.load(f)

        with open(test_file, 'r', encoding='utf-8') as f:
            test_docs = json.load(f)

        # Create document embeddings by averaging word vectors
        def doc_to_vector(text):
            words = text.lower().split()
            words = [word for word in words if word in self.model.wv.key_to_index]

            if not words:
                return np.zeros(self.vector_size)

            vectors = [self.model.wv[word] for word in words]
            return np.mean(vectors, axis=0)

        # Prepare features and labels
        X_train = np.array([doc_to_vector(doc.get('text', '')) for doc in train_docs])
        X_test = np.array([doc_to_vector(doc.get('text', '')) for doc in test_docs])

        # Use source as labels for classification
        y_train = [doc.get('source', 'unknown') for doc in train_docs]
        y_test = [doc.get('source', 'unknown') for doc in test_docs]

        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)

        # Evaluate
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'num_classes': len(set(y_train)),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Complete training and evaluation pipeline"""

        self.logger.info("🚀 Starting Word2Vec training and evaluation pipeline...")

        try:
            # Load training data
            data_path = self.output_dir / 'data' / 'processed'
            sentences = self.load_training_data(str(data_path))

            # Train model
            training_results = self.train_model(sentences)

            # Save model
            model_path = self.save_model()

            # Evaluate model
            evaluation_results = self.evaluate_model(str(data_path))

            # Combine results
            complete_results = {
                'training': training_results,
                'evaluation': evaluation_results,
                'model_path': model_path,
                'status': 'completed'
            }

            self.logger.info("🎉 Word2Vec training and evaluation completed successfully!")

            return complete_results

        except Exception as e:
            self.logger.error(f"❌ Word2Vec pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a specific word"""
        if self.model and word in self.model.wv.key_to_index:
            return self.model.wv[word]
        return None

    def get_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Get most similar words"""
        if self.model and word in self.model.wv.key_to_index:
            try:
                return self.model.wv.most_similar(word, topn=topn)
            except:
                return []
        return []
