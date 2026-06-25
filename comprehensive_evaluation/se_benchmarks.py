"""
SE-Specific Benchmark Datasets for Word Embeddings Evaluation
Curated datasets for fair comparison in software engineering domain
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path


class SEBenchmarkDatasets:
    """
    Curated SE-specific benchmark datasets for comprehensive evaluation
    """

    def __init__(self):
        self.datasets = {}
        self._create_all_benchmarks()

    def _create_all_benchmarks(self):
        """Create all benchmark datasets"""

        # 1. SE Word Similarity Pairs
        self.datasets['word_pairs'] = self._create_se_word_similarity_pairs()

        # 2. SE Analogies
        self.datasets['analogies'] = self._create_se_analogies()

        # 3. SE Vocabulary for Coverage Testing
        self.datasets['se_vocabulary'] = self._create_se_vocabulary()

        # 4. Classification Benchmark
        self.datasets['classification_data'] = self._create_classification_benchmark()

        # 5. Clustering Words
        self.datasets['clustering_words'] = self._create_clustering_benchmark()

        # 6. Retrieval Benchmark
        self.datasets['retrieval_data'] = self._create_retrieval_benchmark()

        # 7. Semantic Similarity Tasks
        self.datasets['similarity_data'] = self._create_semantic_similarity_benchmark()

        # 8. Performance Testing Data
        self.datasets['speed_test_words'] = self._create_speed_test_words()
        self.datasets['scalability_texts'] = self._create_scalability_texts()

    def _create_se_word_similarity_pairs(self) -> List[Dict[str, Any]]:
        """
        Create SE-specific word similarity pairs with human-annotated scores
        Based on SE domain knowledge and terminology relationships
        """

        word_pairs = [
            # Programming Language Relationships
            {'word1': 'python', 'word2': 'programming', 'score': 0.9},
            {'word1': 'java', 'word2': 'programming', 'score': 0.9},
            {'word1': 'javascript', 'word2': 'web', 'score': 0.8},
            {'word1': 'html', 'word2': 'css', 'score': 0.8},
            {'word1': 'react', 'word2': 'frontend', 'score': 0.9},

            # Software Development Concepts
            {'word1': 'bug', 'word2': 'defect', 'score': 0.95},
            {'word1': 'error', 'word2': 'exception', 'score': 0.8},
            {'word1': 'function', 'word2': 'method', 'score': 0.85},
            {'word1': 'class', 'word2': 'object', 'score': 0.8},
            {'word1': 'variable', 'word2': 'parameter', 'score': 0.7},

            # Version Control and DevOps
            {'word1': 'git', 'word2': 'version', 'score': 0.8},
            {'word1': 'commit', 'word2': 'push', 'score': 0.7},
            {'word1': 'branch', 'word2': 'merge', 'score': 0.75},
            {'word1': 'docker', 'word2': 'container', 'score': 0.9},
            {'word1': 'kubernetes', 'word2': 'orchestration', 'score': 0.85},

            # Database and Data
            {'word1': 'database', 'word2': 'sql', 'score': 0.8},
            {'word1': 'query', 'word2': 'select', 'score': 0.75},
            {'word1': 'table', 'word2': 'schema', 'score': 0.7},
            {'word1': 'index', 'word2': 'performance', 'score': 0.6},

            # Testing and Quality
            {'word1': 'testing', 'word2': 'quality', 'score': 0.8},
            {'word1': 'unittest', 'word2': 'testing', 'score': 0.9},
            {'word1': 'debugging', 'word2': 'troubleshooting', 'score': 0.85},
            {'word1': 'refactoring', 'word2': 'improvement', 'score': 0.75},

            # Architecture and Design
            {'word1': 'architecture', 'word2': 'design', 'score': 0.8},
            {'word1': 'microservices', 'word2': 'architecture', 'score': 0.8},
            {'word1': 'api', 'word2': 'interface', 'score': 0.85},
            {'word1': 'rest', 'word2': 'api', 'score': 0.9},

            # Machine Learning and AI
            {'word1': 'algorithm', 'word2': 'machine', 'score': 0.7},
            {'word1': 'neural', 'word2': 'network', 'score': 0.9},
            {'word1': 'training', 'word2': 'model', 'score': 0.8},
            {'word1': 'tensorflow', 'word2': 'pytorch', 'score': 0.8},

            # Web Development
            {'word1': 'frontend', 'word2': 'backend', 'score': 0.6},
            {'word1': 'server', 'word2': 'client', 'score': 0.7},
            {'word1': 'http', 'word2': 'protocol', 'score': 0.8},
            {'word1': 'json', 'word2': 'xml', 'score': 0.7},

            # Security
            {'word1': 'security', 'word2': 'authentication', 'score': 0.8},
            {'word1': 'encryption', 'word2': 'security', 'score': 0.85},
            {'word1': 'vulnerability', 'word2': 'security', 'score': 0.8},

            # Performance and Optimization
            {'word1': 'performance', 'word2': 'optimization', 'score': 0.85},
            {'word1': 'scalability', 'word2': 'performance', 'score': 0.8},
            {'word1': 'caching', 'word2': 'performance', 'score': 0.8},

            # Low similarity pairs (negative examples )
            {'word1': 'python', 'word2': 'snake', 'score': 0.1},
            {'word1': 'java', 'word2': 'coffee', 'score': 0.1},
            {'word1': 'bug', 'word2': 'insect', 'score': 0.2},
            {'word1': 'mouse', 'word2': 'animal', 'score': 0.1},
            {'word1': 'cookie', 'word2': 'food', 'score': 0.1},
            {'word1': 'virus', 'word2': 'disease', 'score': 0.2},
            {'word1': 'cloud', 'word2': 'weather', 'score': 0.1},
            {'word1': 'framework', 'word2': 'building', 'score': 0.2},
        ]

        return word_pairs

    def _create_se_analogies(self) -> List[Dict[str, str]]:
        """
        Create SE-specific analogies for testing analogical reasoning
        Format: A is to B as C is to D
        """

        analogies = [
            # Programming Language Analogies
            {'a': 'python', 'b': 'django', 'c': 'javascript', 'd': 'react'},
            {'a': 'java', 'b': 'spring', 'c': 'python', 'd': 'flask'},
            {'a': 'html', 'b': 'structure', 'c': 'css', 'd': 'styling'},

            # Database Analogies
            {'a': 'mysql', 'b': 'relational', 'c': 'mongodb', 'd': 'nosql'},
            {'a': 'select', 'b': 'read', 'c': 'insert', 'd': 'create'},
            {'a': 'table', 'b': 'rows', 'c': 'collection', 'd': 'documents'},

            # Version Control Analogies
            {'a': 'git', 'b': 'commit', 'c': 'svn', 'd': 'checkin'},
            {'a': 'branch', 'b': 'merge', 'c': 'fork', 'd': 'pull'},

            # Testing Analogies
            {'a': 'unittest', 'b': 'python', 'c': 'jest', 'd': 'javascript'},
            {'a': 'bug', 'b': 'fix', 'c': 'feature', 'd': 'implement'},

            # Architecture Analogies
            {'a': 'monolith', 'b': 'single', 'c': 'microservices', 'd': 'multiple'},
            {'a': 'frontend', 'b': 'user', 'c': 'backend', 'd': 'server'},

            # Development Process Analogies
            {'a': 'agile', 'b': 'iterative', 'c': 'waterfall', 'd': 'sequential'},
            {'a': 'development', 'b': 'coding', 'c': 'testing', 'd': 'verification'},

            # Tool Analogies
            {'a': 'docker', 'b': 'containerization', 'c': 'kubernetes', 'd': 'orchestration'},
            {'a': 'jenkins', 'b': 'ci', 'c': 'ansible', 'd': 'deployment'},

            # Data Structure Analogies
            {'a': 'array', 'b': 'indexed', 'c': 'hash', 'd': 'keyed'},
            {'a': 'stack', 'b': 'lifo', 'c': 'queue', 'd': 'fifo'},

            # Security Analogies
            {'a': 'https', 'b': 'secure', 'c': 'http', 'd': 'plain'},
            {'a': 'authentication', 'b': 'identity', 'c': 'authorization', 'd': 'permission'},
        ]

        return analogies

    def _create_se_vocabulary(self) -> List[str]:
        """
        Create comprehensive SE vocabulary for coverage testing
        """

        se_vocabulary = [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'go', 'rust', 'cpp', 'csharp',
            'php', 'ruby', 'swift', 'kotlin', 'scala', 'clojure', 'haskell', 'erlang',

            # Frameworks and Libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'rails',
            'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy', 'matplotlib',

            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'sqlite', 'oracle', 'sqlserver', 'dynamodb', 'neo4j', 'influxdb',

            # Cloud and DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
            'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'helm',

            # Version Control
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
            'commit', 'push', 'pull', 'merge', 'branch', 'fork', 'clone',

            # Testing
            'unittest', 'pytest', 'jest', 'mocha', 'selenium', 'cypress',
            'junit', 'testng', 'cucumber', 'mockito', 'sinon',

            # Architecture Patterns
            'mvc', 'mvp', 'mvvm', 'microservices', 'monolith', 'serverless',
            'restful', 'graphql', 'soap', 'grpc', 'websocket',

            # Data Structures and Algorithms
            'array', 'list', 'stack', 'queue', 'tree', 'graph', 'hash',
            'sorting', 'searching', 'recursion', 'dynamic', 'greedy',

            # Software Engineering Concepts
            'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'cicd',
            'refactoring', 'debugging', 'profiling', 'optimization',

            # Security
            'authentication', 'authorization', 'encryption', 'ssl', 'tls',
            'oauth', 'jwt', 'csrf', 'xss', 'sql injection', 'vulnerability',

            # Web Technologies
            'html', 'css', 'dom', 'ajax', 'json', 'xml', 'http', 'https',
            'cors', 'cdn', 'spa', 'pwa', 'responsive', 'bootstrap',

            # Mobile Development
            'android', 'ios', 'react-native', 'flutter', 'xamarin',
            'cordova', 'ionic', 'swift', 'objective-c', 'kotlin',

            # Machine Learning
            'algorithm', 'model', 'training', 'validation', 'testing',
            'supervised', 'unsupervised', 'reinforcement', 'neural',
            'regression', 'classification', 'clustering', 'feature',
        ]

        return se_vocabulary

    def _create_classification_benchmark(self) -> Dict[str, Any]:
        """
        Create classification benchmark with SE texts and labels
        """

        # Sample SE texts with source labels
        texts = [
            "Python is a high-level programming language with dynamic semantics",
            "React is a JavaScript library for building user interfaces",
            "Docker containers wrap software in a complete filesystem",
            "Machine learning algorithms can identify patterns in data",
            "Git is a distributed version control system",
            "SQL databases store data in tables with rows and columns",
            "Agile methodology emphasizes iterative development",
            "REST APIs use HTTP methods for communication",
            "Unit testing verifies individual components work correctly",
            "Microservices architecture breaks applications into small services",
            "CSS styles the presentation of HTML documents",
            "DevOps combines development and operations practices",
            "Kubernetes orchestrates containerized applications",
            "JavaScript runs in web browsers and servers",
            "Debugging helps identify and fix software bugs",
        ]

        labels = [
            'programming', 'frontend', 'devops', 'ml', 'version-control',
            'database', 'methodology', 'api', 'testing', 'architecture',
            'frontend', 'devops', 'devops', 'programming', 'debugging'
        ]

        return {
            'texts': texts,
            'labels': labels,
            'num_classes': len(set(labels)),
            'description': 'SE domain text classification benchmark'
        }

    def _create_clustering_benchmark(self) -> List[str]:
        """
        Create words for semantic clustering evaluation
        """

        clustering_words = [
            # Programming Languages Cluster
            'python', 'java', 'javascript', 'cpp', 'csharp', 'go', 'rust',

            # Database Cluster
            'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',

            # Frontend Cluster
            'react', 'angular', 'vue', 'html', 'css', 'javascript', 'bootstrap',

            # DevOps Cluster
            'docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'aws',

            # Testing Cluster
            'unittest', 'pytest', 'jest', 'selenium', 'debugging', 'testing',

            # Version Control Cluster
            'git', 'github', 'commit', 'branch', 'merge', 'pull', 'push',

            # Machine Learning Cluster
            'tensorflow', 'pytorch', 'algorithm', 'model', 'training', 'neural',

            # Architecture Cluster
            'microservices', 'api', 'rest', 'graphql', 'architecture', 'design',
        ]

        return clustering_words

    def _create_retrieval_benchmark(self) -> Dict[str, Any]:
        """
        Create information retrieval benchmark
        """

        documents = [
            {
                'id': 1,
                'text': 'Python web development with Django framework',
                'keywords': ['python', 'django', 'web', 'framework']
            },
            {
                'id': 2,
                'text': 'JavaScript frontend development using React',
                'keywords': ['javascript', 'react', 'frontend', 'ui']
            },
            {
                'id': 3,
                'text': 'Docker containerization for microservices',
                'keywords': ['docker', 'container', 'microservices', 'devops']
            },
            {
                'id': 4,
                'text': 'Machine learning with TensorFlow and Python',
                'keywords': ['machine learning', 'tensorflow', 'python', 'ai']
            },
            {
                'id': 5,
                'text': 'Database design with PostgreSQL',
                'keywords': ['database', 'postgresql', 'sql', 'design']
            },
        ]

        queries = [
            {
                'query': 'python web framework',
                'relevant_docs': [1, 4]
            },
            {
                'query': 'frontend javascript library',
                'relevant_docs': [2]
            },
            {
                'query': 'container orchestration',
                'relevant_docs': [3]
            },
            {
                'query': 'machine learning framework',
                'relevant_docs': [4]
            },
            {
                'query': 'relational database',
                'relevant_docs': [5]
            },
        ]

        return {
            'documents': documents,
            'queries': queries,
            'description': 'SE domain information retrieval benchmark'
        }

    def _create_semantic_similarity_benchmark(self) -> Dict[str, Any]:
        """
        Create semantic similarity benchmark for SE texts
        """

        text_pairs = [
            {
                'text1': 'Python is a programming language',
                'text2': 'Java is a programming language',
                'similarity': 0.8,
                'category': 'programming_languages'
            },
            {
                'text1': 'Docker containers package applications',
                'text2': 'Kubernetes orchestrates containers',
                'similarity': 0.7,
                'category': 'containerization'
            },
            {
                'text1': 'React builds user interfaces',
                'text2': 'Angular creates web applications',
                'similarity': 0.75,
                'category': 'frontend_frameworks'
            },
            {
                'text1': 'Git tracks code changes',
                'text2': 'Version control manages code history',
                'similarity': 0.85,
                'category': 'version_control'
            },
            {
                'text1': 'Machine learning trains models',
                'text2': 'Deep learning uses neural networks',
                'similarity': 0.8,
                'category': 'ai_ml'
            },
        ]

        return {
            'text_pairs': text_pairs,
            'description': 'SE domain semantic similarity benchmark'
        }

    def _create_speed_test_words(self) -> List[str]:
        """
        Create words for speed testing
        """
        return [
            'software', 'programming', 'algorithm', 'code', 'function',
            'class', 'method', 'variable', 'database', 'framework',
            'library', 'api', 'development', 'engineering', 'computer'
        ]

    def _create_scalability_texts(self) -> List[str]:
        """
        Create texts for scalability testing
        """
        return [
            'Software engineering involves systematic approach to software development',
            'Programming languages provide syntax for writing computer programs',
            'Algorithms define step-by-step procedures for solving problems',
            'Data structures organize and store data efficiently',
            'Web development creates applications that run in browsers',
            'Machine learning enables computers to learn from data',
            'Database systems manage persistent data storage',
            'Version control tracks changes in source code',
            'Testing ensures software quality and correctness',
            'DevOps combines development and operations practices',
        ]

    def get_benchmark_data(self) -> Dict[str, Any]:
        """
        Get all benchmark datasets
        """
        return self.datasets

    def save_benchmarks(self, output_dir: str):
        """
        Save benchmark datasets to files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, data in self.datasets.items():
            with open(output_path / f'{name}.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Benchmark datasets saved to {output_dir}")


# Usage example
if __name__ == "__main__":
    benchmarks = SEBenchmarkDatasets()
    benchmark_data = benchmarks.get_benchmark_data()

    print("Created SE benchmark datasets:")
    for name, data in benchmark_data.items():
        if isinstance(data, list):
            print(f"  {name}: {len(data)} items")
        elif isinstance(data, dict):
            print(f"  {name}: {len(data)} keys")
        else:
            print(f"  {name}: {type(data)}")

    # Save to files
    benchmarks.save_benchmarks("se_benchmarks")
