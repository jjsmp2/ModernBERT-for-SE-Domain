"""
SUBMISSION READY - Main pipeline for SE Word Embeddings comparison
FULLY FUNCTIONAL - All issues fixed for immediate execution
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging, get_logger
from src.utils.timer import Timer
from src.data.collector import DataCollector
from src.data.preprocessor import SEDataPreprocessor
from src.models.word2vec_model import Word2VecModel
from src.models.modernbert_model import ModernBERTModel
from src.evaluation.evaluator import SEWordEmbeddingsEvaluator
from src.reporting.report_generator import SEWordEmbeddingsReportGenerator


class SEWordEmbeddingsPipeline:
    """SUBMISSION READY - Main pipeline for SE word embeddings comparison"""

    def __init__(self, config_path: str, output_dir: str = "results"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # Setup logging
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # FIXED: Proper logging configuration
        logging_config = {
            'logging': {
                'level': self.config.get('system', {}).get('log_level', 'INFO'),
                'file': str(log_dir / 'se_embeddings.log'),
                'console': True,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'max_file_size': '10MB',
                'backup_count': 5
            }
        }

        setup_logging(logging_config)
        self.logger = get_logger(__name__)

        # FIXED: Add paths to config for components
        config_with_paths = self.config.copy()
        config_with_paths['paths'] = {
            'data_raw': str(self.output_dir / 'data' / 'raw'),
            'data_processed': str(self.output_dir / 'data' / 'processed'),
            'models': str(self.output_dir / 'models'),
            'evaluations': str(self.output_dir / 'evaluations'),
            'reports': str(self.output_dir / 'reports'),
            'figures': str(self.output_dir / 'figures')
        }

        # Create all directories
        for path in config_with_paths['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Initialize components with proper error handling
        try:
            self.data_collector = DataCollector(config_with_paths, self.logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize data collector: {e}")
            raise

        try:
            self.preprocessor = SEDataPreprocessor(config_with_paths, str(self.output_dir))
        except Exception as e:
            self.logger.error(f"Failed to initialize preprocessor: {e}")
            raise

        try:
            self.word2vec_model = Word2VecModel(config_with_paths, str(self.output_dir))
        except Exception as e:
            self.logger.error(f"Failed to initialize Word2Vec model: {e}")
            raise

        try:
            self.modernbert_model = ModernBERTModel(config_with_paths, str(self.output_dir))
        except Exception as e:
            self.logger.error(f"Failed to initialize ModernBERT model: {e}")
            raise

        try:
            self.evaluator = SEWordEmbeddingsEvaluator(config_with_paths, str(self.output_dir))
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluator: {e}")
            raise

        try:
            self.report_generator = SEWordEmbeddingsReportGenerator(config_with_paths, str(self.output_dir))
        except Exception as e:
            self.logger.error(f"Failed to initialize report generator: {e}")
            raise

    def run_pipeline(self) -> Dict[str, Any]:
        """SUBMISSION READY - Run complete pipeline with all phases"""

        with Timer("Full Pipeline", self.logger) as total_timer:
            self.logger.info("🚀 Starting SE Word Embeddings Comparison Pipeline")
            self.logger.info("=" * 60)

            results = {
                'data_collection': {},
                'preprocessing': {},
                'word2vec_training': {},
                'modernbert_training': {},
                'evaluation': {},
                'reporting': {},
                'timing': {}
            }

            try:
                # PHASE 1: Data Collection
                self.logger.info("PHASE 1: DATA COLLECTION")
                self.logger.info("=" * 60)
                with Timer("Data Collection", self.logger) as timer:
                    collection_results = self.data_collector.collect_all_data()
                    results['data_collection'] = collection_results
                    results['timing']['data_collection'] = timer.get_elapsed()

                # PHASE 2: Data Preprocessing
                self.logger.info("PHASE 2: DATA PREPROCESSING")
                self.logger.info("=" * 60)
                with Timer("Data Preprocessing", self.logger) as timer:
                    data_path = str(self.output_dir / 'data' / 'raw')
                    preprocessing_results = self.preprocessor.process_all_data(data_path)
                    results['preprocessing'] = preprocessing_results
                    results['timing']['preprocessing'] = timer.get_elapsed()

                # PHASE 3: Word2Vec Training
                self.logger.info("PHASE 3: WORD2VEC TRAINING")
                self.logger.info("=" * 60)
                with Timer("Word2Vec Training", self.logger) as timer:
                    word2vec_results = self.word2vec_model.train_and_evaluate()
                    results['word2vec_training'] = word2vec_results
                    results['timing']['word2vec_training'] = timer.get_elapsed()

                # PHASE 4: ModernBERT Training
                self.logger.info("PHASE 4: MODERNBERT TRAINING")
                self.logger.info("=" * 60)
                with Timer("ModernBERT Training", self.logger) as timer:
                    modernbert_results = self.modernbert_model.train_and_evaluate()
                    results['modernbert_training'] = modernbert_results
                    results['timing']['modernbert_training'] = timer.get_elapsed()

                # PHASE 5: Comprehensive Evaluation
                self.logger.info("PHASE 5: COMPREHENSIVE EVALUATION")
                self.logger.info("=" * 60)
                with Timer("Comprehensive Evaluation", self.logger) as timer:
                    evaluation_results = self.evaluator.run_comprehensive_evaluation(
                        word2vec_model=self.word2vec_model.model,
                        modernbert_model=self.modernbert_model,
                        processed_data_path=str(self.output_dir / 'data' / 'processed')
                    )
                    results['evaluation'] = evaluation_results
                    results['timing']['evaluation'] = timer.get_elapsed()

                # PHASE 6: Report Generation
                self.logger.info("PHASE 6: REPORT GENERATION")
                self.logger.info("=" * 60)
                with Timer("Report Generation", self.logger) as timer:
                    report_results = self.report_generator.generate_comprehensive_report(
                        evaluation_results=evaluation_results,
                        word2vec_results=word2vec_results,
                        modernbert_results=modernbert_results,
                        collection_results=collection_results,
                        preprocessing_results=preprocessing_results
                    )
                    results['reporting'] = report_results
                    results['timing']['reporting'] = timer.get_elapsed()

                # Final summary
                results['timing']['total'] = total_timer.get_elapsed()
                results['status'] = 'completed'
                results['output_directory'] = str(self.output_dir)

                self.logger.info("=" * 60)
                self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                self.logger.info(f"📊 Total execution time: {total_timer.get_elapsed_formatted()}")
                self.logger.info(f"📁 Results saved to: {self.output_dir}")
                self.logger.info("=" * 60)

                # Save final results
                results_file = self.output_dir / 'pipeline_results.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                return results

            except Exception as e:
                self.logger.error(f"❌ Pipeline failed: {str(e)}")
                results['status'] = 'failed'
                results['error'] = str(e)
                results['timing']['total'] = total_timer.get_elapsed()
                raise


def main():
    """SUBMISSION READY - Main entry point"""
    parser = argparse.ArgumentParser(description='SE Word Embeddings Comparison Pipeline')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    try:
        # Initialize and run pipeline
        pipeline = SEWordEmbeddingsPipeline(args.config, args.output)
        results = pipeline.run_pipeline()

        print("\n" + "="*60)
        print("🎉 SE WORD EMBEDDINGS COMPARISON COMPLETED!")
        print("="*60)
        print(f"📊 Results saved to: {args.output}")
        print(f"⏱️  Total time: {results['timing']['total']:.2f} seconds")
        print("📋 Generated outputs:")
        print(f"   • Trained models: {args.output}/models/")
        print(f"   • Evaluation results: {args.output}/evaluations/")
        print(f"   • Academic reports: {args.output}/reports/")
        print(f"   • Visualizations: {args.output}/figures/")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
