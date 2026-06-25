"""
SE Data Preprocessor for Word Embeddings
Handles cleaning, filtering, and splitting of collected SE data
"""
"""
SE Data Preprocessor for Word Embeddings
Handles cleaning, filtering, and splitting of collected SE data
"""

import os
import json
import re
import time
import ssl
import nltk
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Fix SSL certificate issue on Mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with SSL fix
try:
    nltk.download('punkt', quiet=True )
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from ..utils.logger import get_logger
from ..utils.timer import Timer

# Rest of your preprocessor code stays the same...

import os
import json
import re
import time
import nltk
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from ..utils.logger import get_logger
from ..utils.timer import Timer


class SEDataPreprocessor:
    """Preprocessor for SE text data with comprehensive cleaning and filtering"""

    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = get_logger(__name__)

        # Preprocessing configuration
        self.preprocess_config = config.get('preprocessing', {})

        # Paths
        self.paths = config.get('paths', {})
        self.data_raw_dir = Path(self.paths.get('data_raw', 'results/data/raw'))
        self.data_processed_dir = Path(self.paths.get('data_processed', 'results/data/processed'))
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.lemmatizer = None
            self.logger.warning("NLTK components not available, using basic preprocessing")

        # SE-specific keywords for relevance filtering
        self.se_keywords = set(self.preprocess_config.get('se_keywords', [
            'software', 'programming', 'algorithm', 'code', 'function',
            'class', 'method', 'variable', 'database', 'framework',
            'library', 'api', 'development', 'engineering', 'computer',
            'system', 'application', 'interface', 'architecture', 'design'
        ]))

        # Statistics
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'filtered_out': 0,
            'train_documents': 0,
            'val_documents': 0,
            'test_documents': 0
        }

    def process_all_data(self, data_path: str) -> Dict[str, Any]:
        """Process all collected data from the specified path"""

        self.logger.info("🔄 Starting data preprocessing...")
        start_time = time.time()

        # Load all collected data
        documents = self._load_collected_data(data_path)
        self.stats['total_documents'] = len(documents)

        if not documents:
            self.logger.error("No documents found to process!")
            return {'status': 'failed', 'error': 'No documents found'}

        self.logger.info(f"📊 Loaded {len(documents)} documents for preprocessing")

        # Clean and filter documents
        cleaned_documents = self._clean_documents(documents)
        filtered_documents = self._filter_documents(cleaned_documents)

        self.stats['processed_documents'] = len(filtered_documents)
        self.stats['filtered_out'] = len(documents) - len(filtered_documents)

        if not filtered_documents:
            self.logger.error("No documents remaining after filtering!")
            return {'status': 'failed', 'error': 'All documents filtered out'}

        self.logger.info(f"✅ Processed {len(filtered_documents)} documents after cleaning and filtering")

        # Split into train/val/test sets
        train_docs, val_docs, test_docs = self._split_documents(filtered_documents)

        self.stats['train_documents'] = len(train_docs)
        self.stats['val_documents'] = len(val_docs)
        self.stats['test_documents'] = len(test_docs)

        # Save processed data
        self._save_processed_data(train_docs, val_docs, test_docs)

        # Generate preprocessing report
        processing_time = time.time() - start_time
        results = {
            'status': 'completed',
            'processing_time': processing_time,
            'statistics': self.stats,
            'output_files': {
                'train': str(self.data_processed_dir / 'train.json'),
                'val': str(self.data_processed_dir / 'val.json'),
                'test': str(self.data_processed_dir / 'test.json'),
                'metadata': str(self.data_processed_dir / 'metadata.json')
            }
        }

        self.logger.info(f"🎉 Data preprocessing completed in {processing_time:.2f} seconds")
        self.logger.info(f"📊 Final statistics: {self.stats}")

        return results

    def _load_collected_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load all collected data from JSON files"""

        documents = []
        data_dir = Path(data_path)

        # Look for data files
        data_files = [
            'wikipedia_articles.json',
            'github_repos.json',
            'stackoverflow_posts.json',
            'arxiv_papers.json',
            'sample_data.json'
        ]

        for filename in data_files:
            file_path = data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            documents.extend(data)
                            self.logger.info(f"📁 Loaded {len(data)} documents from {filename}")
                        else:
                            self.logger.warning(f"⚠️  Unexpected data format in {filename}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {filename}: {e}")

        return documents

    def _clean_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean document text content"""

        self.logger.info("🧹 Cleaning document text...")
        cleaned_docs = []

        for doc in documents:
            try:
                # Extract text content
                text = doc.get('text', '')
                if not text:
                    # Try alternative text fields
                    text = doc.get('body', '') or doc.get('abstract', '') or doc.get('description', '')

                if not text:
                    continue

                # Clean text
                cleaned_text = self._clean_text(text)

                if cleaned_text and len(cleaned_text.strip()) > 0:
                    cleaned_doc = doc.copy()
                    cleaned_doc['text'] = cleaned_text
                    cleaned_doc['word_count'] = len(cleaned_text.split())
                    cleaned_doc['char_count'] = len(cleaned_text)
                    cleaned_docs.append(cleaned_doc)

            except Exception as e:
                self.logger.warning(f"Failed to clean document: {e}")
                continue

        return cleaned_docs

    def _clean_text(self, text: str) -> str:
        """Clean individual text content"""

        if not text:
            return ""

        # Remove HTML tags if enabled
        if self.preprocess_config.get('remove_html', True):
            text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs if enabled
        if self.preprocess_config.get('remove_urls', True):
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\ ),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove email addresses if enabled
        if self.preprocess_config.get('remove_emails', True):
            text = re.sub(r'\S+@\S+', ' ', text)

        # Remove special characters if enabled
        if self.preprocess_config.get('remove_special_chars', True):
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', ' ', text)

        # Normalize whitespace if enabled
        if self.preprocess_config.get('normalize_whitespace', True):
            text = re.sub(r'\s+', ' ', text)

        # Basic cleaning
        text = text.strip()

        return text

    def _filter_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter documents based on quality and relevance criteria"""

        self.logger.info("🔍 Filtering documents by quality and relevance...")
        filtered_docs = []

        min_doc_length = self.preprocess_config.get('min_doc_length', 50)
        max_doc_length = self.preprocess_config.get('max_doc_length', 10000)
        min_word_count = self.preprocess_config.get('min_word_count', 10)
        max_word_count = self.preprocess_config.get('max_word_count', 2000)
        se_relevance_threshold = self.preprocess_config.get('se_relevance_threshold', 0.1)

        for doc in documents:
            try:
                text = doc.get('text', '')
                word_count = doc.get('word_count', len(text.split()))
                char_count = doc.get('char_count', len(text))

                # Length filters
                if char_count < min_doc_length or char_count > max_doc_length:
                    continue

                if word_count < min_word_count or word_count > max_word_count:
                    continue

                # SE relevance filter
                if se_relevance_threshold > 0:
                    relevance_score = self._calculate_se_relevance(text)
                    if relevance_score < se_relevance_threshold:
                        continue
                    doc['se_relevance_score'] = relevance_score

                filtered_docs.append(doc)

            except Exception as e:
                self.logger.warning(f"Failed to filter document: {e}")
                continue

        return filtered_docs

    def _calculate_se_relevance(self, text: str) -> float:
        """Calculate SE relevance score based on keyword presence"""

        if not text or not self.se_keywords:
            return 0.0

        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.0

        # Count SE keyword occurrences
        se_word_count = sum(1 for word in words if word in self.se_keywords)

        # Calculate relevance as ratio of SE words to total words
        relevance_score = se_word_count / len(words)

        return relevance_score

    def _split_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split documents into train/validation/test sets"""

        self.logger.info("📊 Splitting documents into train/val/test sets...")

        train_ratio = self.preprocess_config.get('train_ratio', 0.7)
        val_ratio = self.preprocess_config.get('val_ratio', 0.15)
        test_ratio = self.preprocess_config.get('test_ratio', 0.15)
        random_seed = self.preprocess_config.get('random_seed', 42)

        # Ensure ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            self.logger.warning(f"Ratios don't sum to 1.0 ({total_ratio}), normalizing...")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        # First split: train vs (val + test)
        train_docs, temp_docs = train_test_split(
            documents,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )

        # Second split: val vs test
        if temp_docs:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_docs, test_docs = train_test_split(
                temp_docs,
                test_size=(1 - val_size),
                random_state=random_seed,
                shuffle=True
            )
        else:
            val_docs, test_docs = [], []

        self.logger.info(f"📊 Split: {len(train_docs)} train, {len(val_docs)} val, {len(test_docs)} test")

        return train_docs, val_docs, test_docs

    def _save_processed_data(self, train_docs: List[Dict], val_docs: List[Dict], test_docs: List[Dict]):
        """Save processed data to files"""

        self.logger.info("💾 Saving processed data...")

        # Save train/val/test splits
        datasets = {
            'train.json': train_docs,
            'val.json': val_docs,
            'test.json': test_docs
        }

        for filename, data in datasets.items():
            output_path = self.data_processed_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 Saved {len(data)} documents to {filename}")

        # Save metadata
        metadata = {
            'preprocessing_config': self.preprocess_config,
            'statistics': self.stats,
            'timestamp': time.time(),
            'total_documents': len(train_docs) + len(val_docs) + len(test_docs)
        }

        metadata_path = self.data_processed_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"💾 Saved preprocessing metadata to metadata.json")
