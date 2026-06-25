"""
Simplified Robust Evaluation Framework for SE Word Embeddings
Working version that integrates with your existing models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import time
import logging
from pathlib import Path
import json

class RobustSEEvaluator:
    """
    Simplified evaluation framework that works with your existing models
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.results = {}

    def _setup_logger(self):
        """Setup logging for evaluation"""
        logger = logging.getLogger('robust_evaluator')
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        handler = logging.FileHandler(self.output_dir / 'evaluation.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def evaluate_models(self, word2vec_model, modernbert_model, benchmark_data):
        """
        Main evaluation method - simplified but functional
        """

        self.logger.info("Starting comprehensive model evaluation...")

        results = {
            'intrinsic_evaluation': {},
            'extrinsic_evaluation': {},
            'performance_benchmarks': {},
            'model_info': {}
        }

        # 1. Model Information
        results['model_info'] = self._get_model_info(word2vec_model, modernbert_model)

        # 2. Intrinsic Evaluation
        if word2vec_model:
            results['intrinsic_evaluation'] = self._intrinsic_evaluation(
                word2vec_model, modernbert_model, benchmark_data
            )

        # 3. Extrinsic Evaluation (simplified)
        results['extrinsic_evaluation'] = self._simplified_extrinsic_evaluation(
            word2vec_model, modernbert_model, benchmark_data
        )

        # 4. Performance Benchmarks
        results['performance_benchmarks'] = self._performance_benchmarks(
            word2vec_model, modernbert_model
        )

        # Save results
        self._save_results(results)

        self.logger.info("Evaluation completed successfully!")

        return results

    def _get_model_info(self, word2vec_model, modernbert_model):
        """Get basic information about the models"""

        info = {}

        # Word2Vec info
        if word2vec_model:
            try:
                info['word2vec'] = {
                    'vocabulary_size': len(word2vec_model.wv.key_to_index),
                    'vector_size': word2vec_model.wv.vector_size,
                    'total_words': word2vec_model.corpus_total_words if hasattr(word2vec_model, 'corpus_total_words') else 'unknown',
                    'epochs': word2vec_model.epochs if hasattr(word2vec_model, 'epochs') else 'unknown'
                }
            except Exception as e:
                info['word2vec'] = {'error': str(e)}
        else:
            info['word2vec'] = {'status': 'not_loaded'}

        # ModernBERT info
        if modernbert_model:
            try:
                info['modernbert'] = {
                    'model_type': 'transformer',
                    'status': 'loaded'
                }
            except Exception as e:
                info['modernbert'] = {'error': str(e)}
        else:
            info['modernbert'] = {'status': 'not_loaded'}

        return info

    def _intrinsic_evaluation(self, word2vec_model, modernbert_model, benchmark_data):
        """
        Intrinsic evaluation using benchmark data
        """

        intrinsic_results = {}

        # 1. Word Similarity Evaluation
        if 'word_pairs' in benchmark_data and word2vec_model:
            intrinsic_results['word_similarity'] = self._evaluate_word_similarity(
                word2vec_model, benchmark_data['word_pairs']
            )

        # 2. Analogy Evaluation
        if 'analogies' in benchmark_data and word2vec_model:
            intrinsic_results['analogies'] = self._evaluate_analogies(
                word2vec_model, benchmark_data['analogies']
            )

        # 3. Vocabulary Coverage
        if 'se_vocabulary' in benchmark_data and word2vec_model:
            intrinsic_results['vocabulary_coverage'] = self._evaluate_vocabulary_coverage(
                word2vec_model, benchmark_data['se_vocabulary']
            )

        # 4. Clustering Quality
        if 'clustering_words' in benchmark_data and word2vec_model:
            intrinsic_results['clustering'] = self._evaluate_clustering(
                word2vec_model, benchmark_data['clustering_words']
            )

        return intrinsic_results

    def _evaluate_word_similarity(self, word2vec_model, word_pairs):
        """Evaluate word similarity using word pairs"""

        if not word_pairs:
            return {'error': 'No word pairs provided'}

        try:
            model_similarities = []
            human_similarities = []

            for pair in word_pairs:
                word1 = pair['word1']
                word2 = pair['word2']
                human_score = pair['score']

                # Check if both words are in vocabulary
                if (word1 in word2vec_model.wv.key_to_index and
                    word2 in word2vec_model.wv.key_to_index):

                    model_sim = word2vec_model.wv.similarity(word1, word2)
                    model_similarities.append(model_sim)
                    human_similarities.append(human_score)

            if len(model_similarities) > 1:
                correlation, p_value = spearmanr(human_similarities, model_similarities)

                return {
                    'word2vec': {
                        'spearman_correlation': correlation,
                        'p_value': p_value,
                        'pairs_evaluated': len(model_similarities),
                        'coverage': len(model_similarities) / len(word_pairs)
                    }
                }
            else:
                return {'error': 'Insufficient word pairs in vocabulary'}

        except Exception as e:
            return {'error': str(e)}

    def _evaluate_analogies(self, word2vec_model, analogies):
        """Evaluate analogical reasoning"""

        if not analogies:
            return {'error': 'No analogies provided'}

        try:
            correct = 0
            total = 0

            for analogy in analogies:
                try:
                    # A is to B as C is to D
                    a, b, c, d = analogy['a'], analogy['b'], analogy['c'], analogy['d']

                    # Check if all words are in vocabulary
                    if all(word in word2vec_model.wv.key_to_index for word in [a, b, c, d]):
                        # Get most similar words to (B - A + C)
                        result = word2vec_model.wv.most_similar(
                            positive=[b, c], negative=[a], topn=5
                        )

                        # Check if D is in top 5 results
                        predicted_words = [word for word, _ in result]
                        if d in predicted_words:
                            correct += 1

                        total += 1

                except Exception:
                    continue

            accuracy = correct / total if total > 0 else 0

            return {
                'word2vec': {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'coverage': total / len(analogies)
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def _evaluate_vocabulary_coverage(self, word2vec_model, se_vocabulary):
        """Evaluate vocabulary coverage"""

        if not se_vocabulary:
            return {'error': 'No vocabulary provided'}

        try:
            covered_words = 0
            total_words = len(se_vocabulary)

            for word in se_vocabulary:
                if word in word2vec_model.wv.key_to_index:
                    covered_words += 1

            coverage = covered_words / total_words

            return {
                'word2vec': {
                    'coverage_ratio': coverage,
                    'covered_words': covered_words,
                    'total_words': total_words,
                    'missing_words': total_words - covered_words
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def _evaluate_clustering(self, word2vec_model, clustering_words):
        """Evaluate clustering quality"""

        if not clustering_words:
            return {'error': 'No clustering words provided'}

        try:
            # Get word vectors
            vectors = []
            valid_words = []

            for word in clustering_words:
                if word in word2vec_model.wv.key_to_index:
                    vectors.append(word2vec_model.wv[word])
                    valid_words.append(word)

            if len(vectors) < 3:
                return {'error': 'Insufficient words for clustering'}

            vectors = np.array(vectors)

            # Perform K-means clustering
            n_clusters = min(5, len(vectors) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)

            # Calculate silhouette score
            silhouette = silhouette_score(vectors, cluster_labels)

            return {
                'word2vec': {
                    'avg_silhouette': silhouette,
                    'n_clusters': n_clusters,
                    'words_clustered': len(valid_words),
                    'coverage': len(valid_words) / len(clustering_words)
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def _simplified_extrinsic_evaluation(self, word2vec_model, modernbert_model, benchmark_data):
        """
        Simplified extrinsic evaluation that works with available data
        """

        extrinsic_results = {}

        # Simple classification using available data
        if 'classification_data' in benchmark_data and word2vec_model:
            extrinsic_results['classification'] = self._simple_classification_eval(
                word2vec_model, benchmark_data['classification_data']
            )

        return extrinsic_results

    def _simple_classification_eval(self, word2vec_model, classification_data):
        """Simple classification evaluation"""

        try:
            texts = classification_data.get('texts', [])
            labels = classification_data.get('labels', [])

            if not texts or not labels:
                return {'error': 'No classification data available'}

            # Create simple document vectors by averaging word vectors
            doc_vectors = []
            valid_indices = []

            for i, text in enumerate(texts):
                words = text.lower().split()
                word_vectors = []

                for word in words:
                    if word in word2vec_model.wv.key_to_index:
                        word_vectors.append(word2vec_model.wv[word])

                if word_vectors:
                    doc_vector = np.mean(word_vectors, axis=0)
                    doc_vectors.append(doc_vector)
                    valid_indices.append(i)

            if len(doc_vectors) < 2:
                return {'error': 'Insufficient valid documents for classification'}

            # Filter labels for valid documents
            valid_labels = [labels[i] for i in valid_indices]

            # Simple classification with cross-validation
            X = np.array(doc_vectors)
            y = valid_labels

            # Logistic Regression
            lr_scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000),
                                      X, y, cv=min(3, len(set(y))), scoring='f1_macro')

            return {
                'word2vec': {
                    'logistic_regression': {
                        'f1_scores': lr_scores.tolist(),
                        'mean_f1': lr_scores.mean(),
                        'std_f1': lr_scores.std()
                    },
                    'documents_used': len(doc_vectors),
                    'total_documents': len(texts)
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def _performance_benchmarks(self, word2vec_model, modernbert_model):
        """Simple performance benchmarks"""

        benchmarks = {}

        # Word2Vec performance
        if word2vec_model:
            try:
                # Test similarity computation speed
                test_words = ['software', 'programming', 'algorithm', 'code', 'function']
                available_words = [w for w in test_words if w in word2vec_model.wv.key_to_index]

                if len(available_words) >= 2:
                    start_time = time.time()
                    for i in range(100):  # 100 similarity computations
                        word2vec_model.wv.similarity(available_words[0], available_words[1])
                    end_time = time.time()

                    avg_time = (end_time - start_time) / 100

                    benchmarks['word2vec'] = {
                        'similarity_computation_time_ms': avg_time * 1000,
                        'vocabulary_size': len(word2vec_model.wv.key_to_index),
                        'vector_size': word2vec_model.wv.vector_size
                    }
                else:
                    benchmarks['word2vec'] = {'error': 'Insufficient test words in vocabulary'}

            except Exception as e:
                benchmarks['word2vec'] = {'error': str(e)}

        # ModernBERT performance (placeholder)
        if modernbert_model:
            benchmarks['modernbert'] = {
                'status': 'loaded',
                'note': 'Performance benchmarking requires additional implementation'
            }
        else:
            benchmarks['modernbert'] = {'status': 'not_available'}

        return benchmarks

    def _save_results(self, results):
        """Save evaluation results"""

        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Results saved to {results_path}")

        # Save summary CSV
        self._save_summary_csv(results)

    def _save_summary_csv(self, results):
        """Save summary in CSV format"""

        summary_data = []

        # Extract key metrics
        if 'intrinsic_evaluation' in results:
            intrinsic = results['intrinsic_evaluation']

            for metric_name, metric_data in intrinsic.items():
                if isinstance(metric_data, dict) and 'word2vec' in metric_data:
                    w2v_data = metric_data['word2vec']

                    for key, value in w2v_data.items():
                        if isinstance(value, (int, float)):
                            summary_data.append({
                                'category': 'intrinsic',
                                'metric': metric_name,
                                'sub_metric': key,
                                'model': 'word2vec',
                                'value': value
                            })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / 'evaluation_summary.csv'
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Summary saved to {csv_path}")
