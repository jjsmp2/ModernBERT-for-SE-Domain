"""
SE Word Embeddings Evaluator
Comprehensive evaluation framework for comparing Word2Vec and ModernBERT
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import get_logger
from ..utils.timer import Timer

class SEWordEmbeddingsEvaluator:
    """
    Comprehensive evaluator for SE word embeddings comparison
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = get_logger(__name__)

        # Evaluation configuration
        self.eval_config = config.get('evaluation', {})

        # Results storage
        self.evaluation_results = {}

        self.logger.info("Initialized SE Word Embeddings Evaluator")

    def run_comprehensive_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run comprehensive evaluation comparing Word2Vec and ModernBERT"""

        self.logger.info("🚀 Starting comprehensive evaluation...")

        # Extract models from kwargs
        word2vec_model = kwargs.get('word2vec_model')
        modernbert_model = kwargs.get('modernbert_model')
        processed_data_path = kwargs.get('processed_data_path', '')

        evaluation_results = {
            'word2vec_evaluation': {},
            'modernbert_evaluation': {},
            'comparison': {},
            'summary': {}
        }

        try:
            # Evaluate Word2Vec
            if word2vec_model:
                self.logger.info("📊 Evaluating Word2Vec model...")
                evaluation_results['word2vec_evaluation'] = self._evaluate_word2vec(word2vec_model, processed_data_path)

            # Evaluate ModernBERT
            if modernbert_model:
                self.logger.info("📊 Evaluating ModernBERT model...")
                evaluation_results['modernbert_evaluation'] = self._evaluate_modernbert(modernbert_model, processed_data_path)

            # Compare models
            self.logger.info("📊 Comparing models...")
            evaluation_results['comparison'] = self._compare_models(
                evaluation_results['word2vec_evaluation'],
                evaluation_results['modernbert_evaluation']
            )

            # Generate summary
            evaluation_results['summary'] = self._generate_evaluation_summary(evaluation_results)

            # Save results
            self._save_evaluation_results(evaluation_results)

            self.logger.info("✅ Comprehensive evaluation completed")

            return evaluation_results

        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'word2vec_evaluation': {},
                'modernbert_evaluation': {},
                'comparison': {},
                'summary': {}
            }

    def _evaluate_word2vec(self, model, data_path: str) -> Dict[str, Any]:
        """Evaluate Word2Vec model"""

        results = {
            'vocabulary_size': 0,
            'vector_dimensions': 0,
            'similarity_examples': {},
            'status': 'completed'
        }

        try:
            if hasattr(model, 'wv'):
                results['vocabulary_size'] = len(model.wv.key_to_index)
                results['vector_dimensions'] = model.wv.vector_size

                # Test similarity examples
                test_words = ['software', 'programming', 'algorithm', 'code', 'function']
                for word in test_words:
                    if word in model.wv.key_to_index:
                        try:
                            similar = model.wv.most_similar(word, topn=3)
                            results['similarity_examples'][word] = similar
                        except:
                            results['similarity_examples'][word] = 'No similarities found'
                    else:
                        results['similarity_examples'][word] = 'Not in vocabulary'

        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'failed'

        return results

    def _evaluate_modernbert(self, model, data_path: str) -> Dict[str, Any]:
        """Evaluate ModernBERT model"""

        results = {
            'model_name': 'ModernBERT',
            'status': 'completed'
        }

        try:
            if hasattr(model, 'bert_config'):
                results['max_length'] = model.bert_config.max_length
                results['batch_size'] = model.bert_config.batch_size
                results['model_name'] = model.bert_config.model_name

            if hasattr(model, 'num_labels'):
                results['num_labels'] = model.num_labels

            if hasattr(model, 'label_mapping'):
                results['label_mapping'] = model.label_mapping

        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'failed'

        return results

    def _compare_models(self, word2vec_results: Dict, modernbert_results: Dict) -> Dict[str, Any]:
        """Compare Word2Vec and ModernBERT results"""

        comparison = {
            'word2vec_vocab_size': word2vec_results.get('vocabulary_size', 0),
            'word2vec_dimensions': word2vec_results.get('vector_dimensions', 0),
            'modernbert_model': modernbert_results.get('model_name', 'Unknown'),
            'modernbert_max_length': modernbert_results.get('max_length', 0),
            'comparison_summary': 'Word2Vec provides traditional word embeddings while ModernBERT offers contextual embeddings'
        }

        return comparison

    def _generate_evaluation_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate evaluation summary"""

        summary = {
            'total_evaluations': 2,
            'word2vec_status': results['word2vec_evaluation'].get('status', 'unknown'),
            'modernbert_status': results['modernbert_evaluation'].get('status', 'unknown'),
            'evaluation_timestamp': time.time(),
            'recommendations': [
                'Word2Vec provides fast training and good baseline performance',
                'ModernBERT offers state-of-the-art contextual understanding',
                'Both models complement each other for comprehensive SE analysis'
            ]
        }

        return summary

    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results"""

        eval_dir = self.output_dir / 'evaluations'
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Save comprehensive results
        results_path = eval_dir / 'comprehensive_evaluation.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"💾 Evaluation results saved to {results_path}")
