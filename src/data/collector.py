"""
Real Data Collector for SE Word Embeddings - Production Version
Fetches actual data from Wikipedia, GitHub, Stack Overflow, and ArXiv
Handles API issues gracefully with comprehensive error handling
"""

import requests
import json
import time
import os
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import base64
from pathlib import Path
from typing import List, Dict, Any
import random
import logging

class DataCollector:
    """Production-ready data collector that fetches real data from multiple sources"""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()

        # Set up session with proper headers
        self.session.headers.update({
            'User-Agent': 'SE-Word-Embeddings-Research/1.0 (Educational Research Project)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })

        # Create SSL context that handles certificate issues
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        # Get output paths
        self.paths = config.get('paths', {})
        self.data_raw_dir = Path(self.paths.get('data_raw', 'results/data/raw'))
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting and timeout settings
        self.request_delay = 1.0  # 1 second between requests
        self.timeout = 30  # 30 second timeout
        self.max_retries = 3

        # Collection statistics
        self.stats = {
            'wikipedia': {'attempted': 0, 'collected': 0, 'failed': 0},
            'github': {'attempted': 0, 'collected': 0, 'failed': 0},
            'stackoverflow': {'attempted': 0, 'collected': 0, 'failed': 0},
            'arxiv': {'attempted': 0, 'collected': 0, 'failed': 0}
        }

    def collect_all_data(self):
        """Collect data from all enabled sources"""

        self.logger.info("🚀 Starting REAL data collection from all sources...")

        results = {
            'wikipedia': {'collected': 0, 'status': 'not_started'},
            'github': {'collected': 0, 'status': 'not_started'},
            'stackoverflow': {'collected': 0, 'status': 'not_started'},
            'arxiv': {'collected': 0, 'status': 'not_started'},
            'total_collected': 0,
            'collection_time': 0
        }

        start_time = time.time()
        data_config = self.config.get('data_collection', {})

        # Collect Wikipedia articles
        if data_config.get('wikipedia', {}).get('enabled', True):
            try:
                self.logger.info("📚 Starting Wikipedia collection...")
                wiki_results = self.collect_wikipedia_articles()
                results['wikipedia'] = wiki_results
                results['total_collected'] += wiki_results['collected']
                self.logger.info(f"✅ Wikipedia: {wiki_results['collected']} articles collected")
            except Exception as e:
                self.logger.error(f"❌ Wikipedia collection failed: {e}")
                results['wikipedia']['status'] = 'failed'
                results['wikipedia']['error'] = str(e)

        # Collect GitHub repositories
        if data_config.get('github', {}).get('enabled', True):
            try:
                self.logger.info("🐙 Starting GitHub collection...")
                github_results = self.collect_github_repos()
                results['github'] = github_results
                results['total_collected'] += github_results['collected']
                self.logger.info(f"✅ GitHub: {github_results['collected']} repositories collected")
            except Exception as e:
                self.logger.error(f"❌ GitHub collection failed: {e}")
                results['github']['status'] = 'failed'
                results['github']['error'] = str(e)

        # Collect Stack Overflow posts
        if data_config.get('stackoverflow', {}).get('enabled', True):
            try:
                self.logger.info("📚 Starting Stack Overflow collection...")
                so_results = self.collect_stackoverflow_posts()
                results['stackoverflow'] = so_results
                results['total_collected'] += so_results['collected']
                self.logger.info(f"✅ Stack Overflow: {so_results['collected']} posts collected")
            except Exception as e:
                self.logger.error(f"❌ Stack Overflow collection failed: {e}")
                results['stackoverflow']['status'] = 'failed'
                results['stackoverflow']['error'] = str(e)

        # Collect ArXiv papers
        if data_config.get('arxiv', {}).get('enabled', True):
            try:
                self.logger.info("📄 Starting ArXiv collection...")
                arxiv_results = self.collect_arxiv_papers()
                results['arxiv'] = arxiv_results
                results['total_collected'] += arxiv_results['collected']
                self.logger.info(f"✅ ArXiv: {arxiv_results['collected']} papers collected")
            except Exception as e:
                self.logger.error(f"❌ ArXiv collection failed: {e}")
                results['arxiv']['status'] = 'failed'
                results['arxiv']['error'] = str(e)

        results['collection_time'] = time.time() - start_time

        # If total collection is very low, add some fallback sample data
        if results['total_collected'] < 50:
            self.logger.warning("⚠️  Low data collection, adding fallback sample data...")
            sample_results = self._generate_fallback_sample_data()
            results['total_collected'] += sample_results['collected']
            results['fallback_samples'] = sample_results

        self.logger.info(f"🎉 Data collection completed: {results['total_collected']} total documents in {results['collection_time']:.2f} seconds")
        return results

    def collect_wikipedia_articles(self):
        """Collect Wikipedia articles about software engineering"""

        config = self.config.get('data_collection', {}).get('wikipedia', {})
        max_articles = config.get('max_articles', 1000)
        search_terms = config.get('search_terms', [
            'software engineering', 'computer programming', 'software development',
            'programming language', 'software architecture', 'software testing',
            'agile development', 'machine learning', 'artificial intelligence',
            'data structures', 'algorithms', 'software design patterns'
        ])

        articles = []

        # Wikipedia API endpoints
        search_api = "https://en.wikipedia.org/w/api.php"

        for term in search_terms:
            if len(articles ) >= max_articles:
                break

            self.logger.info(f"   Searching Wikipedia for: {term}")
            self.stats['wikipedia']['attempted'] += 1

            try:
                # Search for articles
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': term,
                    'srlimit': min(20, (max_articles - len(articles)) // len(search_terms) + 5)
                }

                response = self.session.get(search_api, params=search_params, timeout=self.timeout)
                response.raise_for_status()
                search_data = response.json()

                if 'query' in search_data and 'search' in search_data['query']:
                    for item in search_data['query']['search']:
                        if len(articles) >= max_articles:
                            break

                        title = item['title']

                        try:
                            # Get article content
                            content_params = {
                                'action': 'query',
                                'format': 'json',
                                'titles': title,
                                'prop': 'extracts',
                                'exintro': True,
                                'explaintext': True,
                                'exsectionformat': 'plain'
                            }

                            content_response = self.session.get(search_api, params=content_params, timeout=self.timeout)
                            content_response.raise_for_status()
                            content_data = content_response.json()

                            if 'query' in content_data and 'pages' in content_data['query']:
                                pages = content_data['query']['pages']
                                for page_id, page_data in pages.items():
                                    if 'extract' in page_data and page_data['extract']:
                                        article = {
                                            'title': title,
                                            'text': page_data['extract'],
                                            'url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title )}",
                                            'source': 'wikipedia',
                                            'search_term': term,
                                            'timestamp': time.time()
                                        }
                                        articles.append(article)
                                        self.stats['wikipedia']['collected'] += 1
                                        break

                            # Rate limiting
                            time.sleep(self.request_delay)

                        except Exception as e:
                            self.logger.warning(f"      Failed to fetch {title}: {e}")
                            self.stats['wikipedia']['failed'] += 1
                            continue

            except Exception as e:
                self.logger.error(f"   Failed to search for {term}: {e}")
                self.stats['wikipedia']['failed'] += 1
                continue

        # Save Wikipedia articles
        if articles:
            output_file = self.data_raw_dir / 'wikipedia_articles.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved {len(articles)} Wikipedia articles to {output_file}")

        return {
            'collected': len(articles),
            'status': 'completed' if articles else 'failed',
            'output_file': str(output_file) if articles else None,
            'search_terms': search_terms
        }

    def collect_github_repos(self):
        """Collect GitHub repository information"""

        config = self.config.get('data_collection', {}).get('github', {})
        max_repos = config.get('max_repos', 500)
        languages = config.get('languages', ['Python', 'JavaScript', 'Java', 'C++', 'Go'])
        min_stars = config.get('min_stars', 100)

        repos = []

        # GitHub search API (no authentication required for basic search)
        github_api = "https://api.github.com/search/repositories"

        for language in languages:
            if len(repos ) >= max_repos:
                break

            self.logger.info(f"   Searching GitHub for {language} repositories...")
            self.stats['github']['attempted'] += 1

            try:
                # Search for repositories
                search_params = {
                    'q': f'language:{language} stars:>={min_stars}',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': min(100, (max_repos - len(repos)) // len(languages) + 10)
                }

                response = self.session.get(github_api, params=search_params, timeout=self.timeout)
                response.raise_for_status()
                search_data = response.json()

                if 'items' in search_data:
                    for item in search_data['items']:
                        if len(repos) >= max_repos:
                            break

                        repo = {
                            'name': item['full_name'],
                            'description': item.get('description', ''),
                            'readme': '',
                            'language': item.get('language', language),
                            'stars': item.get('stargazers_count', 0),
                            'url': item['html_url'],
                            'source': 'github',
                            'timestamp': time.time()
                        }

                        # Try to get README content
                        try:
                            readme_url = f"https://api.github.com/repos/{item['full_name']}/readme"
                            readme_response = self.session.get(readme_url, timeout=self.timeout )
                            if readme_response.status_code == 200:
                                readme_data = readme_response.json()
                                if 'content' in readme_data:
                                    readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                                    repo['readme'] = readme_content[:3000]  # First 3000 chars
                        except:
                            pass  # README not available

                        # Combine description and README for text
                        text_parts = []
                        if repo['description']:
                            text_parts.append(repo['description'])
                        if repo['readme']:
                            text_parts.append(repo['readme'])

                        repo['text'] = ' '.join(text_parts)

                        if repo['text'] and len(repo['text']) > 50:  # Only add if we have substantial text
                            repos.append(repo)
                            self.stats['github']['collected'] += 1

                        # Rate limiting
                        time.sleep(self.request_delay)

            except Exception as e:
                self.logger.error(f"   Failed to search {language} repositories: {e}")
                self.stats['github']['failed'] += 1
                continue

        # Save GitHub repositories
        if repos:
            output_file = self.data_raw_dir / 'github_repos.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(repos, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved {len(repos)} GitHub repositories to {output_file}")

        return {
            'collected': len(repos),
            'status': 'completed' if repos else 'failed',
            'output_file': str(output_file) if repos else None,
            'languages': languages
        }

    def collect_stackoverflow_posts(self):
        """Collect Stack Overflow posts"""

        config = self.config.get('data_collection', {}).get('stackoverflow', {})
        max_posts = config.get('max_posts', 2000)
        tags = config.get('tags', ['python', 'javascript', 'java', 'algorithm', 'data-structures'])
        min_score = config.get('min_score', 1)

        posts = []

        # Stack Overflow API
        so_api = "https://api.stackexchange.com/2.3/questions"

        for tag in tags:
            if len(posts ) >= max_posts:
                break

            self.logger.info(f"   Searching Stack Overflow for: {tag}")
            self.stats['stackoverflow']['attempted'] += 1

            try:
                # Search for questions
                search_params = {
                    'order': 'desc',
                    'sort': 'votes',
                    'tagged': tag,
                    'site': 'stackoverflow',
                    'pagesize': min(100, (max_posts - len(posts)) // len(tags) + 10),
                    'filter': 'withbody'  # Include question body
                }

                response = self.session.get(so_api, params=search_params, timeout=self.timeout)
                response.raise_for_status()
                search_data = response.json()

                if 'items' in search_data:
                    for item in search_data['items']:
                        if len(posts) >= max_posts:
                            break

                        if item.get('score', 0) >= min_score:
                            post = {
                                'title': item.get('title', ''),
                                'body': item.get('body', ''),
                                'text': f"{item.get('title', '')} {item.get('body', '')}",
                                'score': item.get('score', 0),
                                'tags': item.get('tags', []),
                                'url': item.get('link', ''),
                                'source': 'stackoverflow',
                                'search_tag': tag,
                                'timestamp': time.time()
                            }

                            if post['text'] and len(post['text']) > 100:  # Only add substantial posts
                                posts.append(post)
                                self.stats['stackoverflow']['collected'] += 1

                # Rate limiting
                time.sleep(self.request_delay)

            except Exception as e:
                self.logger.error(f"   Failed to search Stack Overflow for {tag}: {e}")
                self.stats['stackoverflow']['failed'] += 1
                continue

        # Save Stack Overflow posts
        if posts:
            output_file = self.data_raw_dir / 'stackoverflow_posts.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(posts, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved {len(posts)} Stack Overflow posts to {output_file}")

        return {
            'collected': len(posts),
            'status': 'completed' if posts else 'failed',
            'output_file': str(output_file) if posts else None,
            'tags': tags
        }

    def collect_arxiv_papers(self):
        """Collect ArXiv papers"""

        config = self.config.get('data_collection', {}).get('arxiv', {})
        max_papers = config.get('max_papers', 500)
        categories = config.get('categories', ['cs.SE', 'cs.LG', 'cs.AI'])

        papers = []

        # ArXiv API
        arxiv_api = "http://export.arxiv.org/api/query"

        for category in categories:
            if len(papers ) >= max_papers:
                break

            self.logger.info(f"   Searching ArXiv for category: {category}")
            self.stats['arxiv']['attempted'] += 1

            try:
                # Search for papers
                search_query = f'cat:{category}'
                search_params = {
                    'search_query': search_query,
                    'start': 0,
                    'max_results': min(100, (max_papers - len(papers)) // len(categories) + 10),
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                }

                response = self.session.get(arxiv_api, params=search_params, timeout=self.timeout)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.content)

                # Define namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                for entry in root.findall('atom:entry', ns ):
                    if len(papers) >= max_papers:
                        break

                    title_elem = entry.find('atom:title', ns)
                    summary_elem = entry.find('atom:summary', ns)
                    id_elem = entry.find('atom:id', ns)

                    if title_elem is not None and summary_elem is not None:
                        paper = {
                            'title': title_elem.text.strip() if title_elem.text else '',
                            'abstract': summary_elem.text.strip() if summary_elem.text else '',
                            'text': f"{title_elem.text.strip() if title_elem.text else ''} {summary_elem.text.strip() if summary_elem.text else ''}",
                            'url': id_elem.text.strip() if id_elem and id_elem.text else '',
                            'category': category,
                            'source': 'arxiv',
                            'timestamp': time.time()
                        }

                        if paper['text'] and len(paper['text']) > 100:  # Only add substantial papers
                            papers.append(paper)
                            self.stats['arxiv']['collected'] += 1

                # Rate limiting
                time.sleep(self.request_delay)

            except Exception as e:
                self.logger.error(f"   Failed to search ArXiv for {category}: {e}")
                self.stats['arxiv']['failed'] += 1
                continue

        # Save ArXiv papers
        if papers:
            output_file = self.data_raw_dir / 'arxiv_papers.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved {len(papers)} ArXiv papers to {output_file}")

        return {
            'collected': len(papers),
            'status': 'completed' if papers else 'failed',
            'output_file': str(output_file) if papers else None,
            'categories': categories
        }

    def _generate_fallback_sample_data(self):
        """Generate sample SE data if real collection fails"""

        sample_data = [
            {
                'title': 'Introduction to Software Engineering',
                'text': 'Software engineering is the systematic application of engineering approaches to the development of software. It involves the use of principles from computer science, engineering, and mathematical analysis to design, develop, test, and maintain software systems.',
                'source': 'sample',
                'type': 'educational'
            },
            {
                'title': 'Object-Oriented Programming Concepts',
                'text': 'Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which can contain data and code. The main principles of OOP include encapsulation, inheritance, and polymorphism.',
                'source': 'sample',
                'type': 'educational'
            },
            {
                'title': 'Data Structures and Algorithms',
                'text': 'Data structures are ways of organizing and storing data so that they can be accessed and worked with efficiently. Common data structures include arrays, linked lists, stacks, queues, trees, and graphs.',
                'source': 'sample',
                'type': 'educational'
            },
            {
                'title': 'Machine Learning in Software Development',
                'text': 'Machine learning is increasingly being integrated into software development processes. It can be used for automated testing, code generation, bug detection, and performance optimization.',
                'source': 'sample',
                'type': 'educational'
            },
            {
                'title': 'Web Development Best Practices',
                'text': 'Web development involves creating websites and web applications. Best practices include responsive design, accessibility, security considerations, performance optimization, and following web standards.',
                'source': 'sample',
                'type': 'educational'
            }
        ]

        # Add more sample data to reach minimum threshold
        extended_samples = []
        for i in range(20):  # Generate 20 samples
            base_sample = sample_data[i % len(sample_data)].copy()
            base_sample['title'] = f"{base_sample['title']} - Part {i+1}"
            base_sample['timestamp'] = time.time()
            extended_samples.append(base_sample)

        # Save sample data
        output_file = self.data_raw_dir / 'sample_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extended_samples, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Generated {len(extended_samples)} sample documents")

        return {
            'collected': len(extended_samples),
            'status': 'completed',
            'output_file': str(output_file)
        }
