#!/usr/bin/env python3
"""
Simple runner for comprehensive evaluation
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_comprehensive_evaluation import ComprehensiveEvaluationRunner


def main():
    # Paths to your trained models
    word2vec_model_path = "../results/models/word2vec_model.bin"
    modernbert_model_path = "../results/models/modernbert_model"

    # Output directory
    output_dir = "../evaluation_results"

    # Configuration (optional)
    config = {
        "evaluation_settings": {
            "alpha": 0.05,
            "confidence_level": 0.95
        }
    }

    # Run evaluation
    runner = ComprehensiveEvaluationRunner(output_dir)

    try:
        results = runner.run_comprehensive_evaluation(
            word2vec_model_path,
            modernbert_model_path,
            config
        )

        print("✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
