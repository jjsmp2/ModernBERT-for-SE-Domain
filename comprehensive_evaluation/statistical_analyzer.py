"""
Statistical Analysis Framework for SE Word Embeddings Comparison
Rigorous statistical testing and significance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, shapiro, levene
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import ttest_power
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for word embeddings comparison
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}

    def analyze_comparison(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of model comparison
        """

        analysis_results = {
            'significance_tests': {},
            'effect_sizes': {},
            'correlation_analysis': {},
            'power_analysis': {},
            'descriptive_statistics': {},
            'confidence_intervals': {},
            'multiple_comparisons': {}
        }

        # 1. Significance Testing
        analysis_results['significance_tests'] = self._perform_significance_tests(
            evaluation_results
        )

        # 2. Effect Size Calculations
        analysis_results['effect_sizes'] = self._calculate_effect_sizes(
            evaluation_results
        )

        # 3. Correlation Analysis
        analysis_results['correlation_analysis'] = self._correlation_analysis(
            evaluation_results
        )

        # 4. Power Analysis
        analysis_results['power_analysis'] = self._power_analysis(
            evaluation_results
        )

        # 5. Descriptive Statistics
        analysis_results['descriptive_statistics'] = self._descriptive_statistics(
            evaluation_results
        )

        # 6. Confidence Intervals
        analysis_results['confidence_intervals'] = self._confidence_intervals(
            evaluation_results
        )

        # 7. Multiple Comparisons Correction
        analysis_results['multiple_comparisons'] = self._multiple_comparisons_correction(
            analysis_results['significance_tests']
        )

        self.results = analysis_results
        return analysis_results

    def _perform_significance_tests(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform various significance tests comparing Word2Vec and ModernBERT
        """

        significance_results = {}

        # Extract performance metrics for comparison
        metrics_data = self._extract_comparable_metrics(evaluation_results)

        for metric_name, data in metrics_data.items():
            if len(data['word2vec']) > 0 and len(data['modernbert']) > 0:

                w2v_scores = np.array(data['word2vec'])
                bert_scores = np.array(data['modernbert'])

                # Test for normality
                w2v_normal = self._test_normality(w2v_scores)
                bert_normal = self._test_normality(bert_scores)

                # Test for equal variances
                equal_variances = self._test_equal_variances(w2v_scores, bert_scores)

                metric_results = {
                    'normality': {
                        'word2vec_normal': w2v_normal,
                        'modernbert_normal': bert_normal
                    },
                    'equal_variances': equal_variances,
                    'tests': {}
                }

                # Paired tests (if same number of samples)
                if len(w2v_scores) == len(bert_scores):

                    # Paired t-test (if both normal)
                    if w2v_normal and bert_normal:
                        t_stat, t_p_value = ttest_rel(w2v_scores, bert_scores)
                        metric_results['tests']['paired_ttest'] = {
                            'statistic': t_stat,
                            'p_value': t_p_value,
                            'significant': t_p_value < self.alpha,
                            'interpretation': self._interpret_ttest(t_stat, t_p_value)
                        }

                    # Wilcoxon signed-rank test (non-parametric alternative)
                    try:
                        w_stat, w_p_value = wilcoxon(w2v_scores, bert_scores)
                        metric_results['tests']['wilcoxon'] = {
                            'statistic': w_stat,
                            'p_value': w_p_value,
                            'significant': w_p_value < self.alpha,
                            'interpretation': self._interpret_wilcoxon(w_stat, w_p_value)
                        }
                    except ValueError as e:
                        metric_results['tests']['wilcoxon'] = {
                            'error': str(e),
                            'note': 'Cannot perform Wilcoxon test (likely identical values)'
                        }

                # Independent samples tests
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_p_value = mannwhitneyu(w2v_scores, bert_scores,
                                                     alternative='two-sided')
                    metric_results['tests']['mann_whitney'] = {
                        'statistic': u_stat,
                        'p_value': u_p_value,
                        'significant': u_p_value < self.alpha,
                        'interpretation': self._interpret_mann_whitney(u_stat, u_p_value)
                    }
                except ValueError as e:
                    metric_results['tests']['mann_whitney'] = {
                        'error': str(e)
                    }

                # Independent t-test (if both normal and equal variances)
                if w2v_normal and bert_normal and equal_variances:
                    t_stat, t_p_value = stats.ttest_ind(w2v_scores, bert_scores)
                    metric_results['tests']['independent_ttest'] = {
                        'statistic': t_stat,
                        'p_value': t_p_value,
                        'significant': t_p_value < self.alpha,
                        'interpretation': self._interpret_ttest(t_stat, t_p_value)
                    }

                significance_results[metric_name] = metric_results

        return significance_results

    def _calculate_effect_sizes(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate effect sizes for meaningful comparison
        """

        effect_sizes = {}
        metrics_data = self._extract_comparable_metrics(evaluation_results)

        for metric_name, data in metrics_data.items():
            if len(data['word2vec']) > 0 and len(data['modernbert']) > 0:
                w2v_scores = np.array(data['word2vec'])
                bert_scores = np.array(data['modernbert'])

                # Cohen's d (standardized mean difference)
                cohens_d = self._calculate_cohens_d(w2v_scores, bert_scores)

                # Glass's delta (using control group SD)
                glass_delta = self._calculate_glass_delta(w2v_scores, bert_scores)

                # Hedge's g (bias-corrected Cohen's d)
                hedges_g = self._calculate_hedges_g(w2v_scores, bert_scores)

                # Common Language Effect Size
                cles = self._calculate_cles(w2v_scores, bert_scores)

                effect_sizes[metric_name] = {
                    'cohens_d': {
                        'value': cohens_d,
                        'magnitude': self._interpret_cohens_d(cohens_d),
                        'description': 'Standardized mean difference'
                    },
                    'glass_delta': {
                        'value': glass_delta,
                        'description': 'Mean difference using control group SD'
                    },
                    'hedges_g': {
                        'value': hedges_g,
                        'magnitude': self._interpret_cohens_d(hedges_g),  # Same interpretation as Cohen's d
                        'description': 'Bias-corrected standardized mean difference'
                    },
                    'cles': {
                        'value': cles,
                        'description': 'Probability that random score from group 1 > group 2'
                    }
                }

        return effect_sizes

    def _correlation_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics
        """

        correlation_results = {}

        # Extract all metrics into a DataFrame
        metrics_df = self._create_metrics_dataframe(evaluation_results)

        if not metrics_df.empty:
            # Pearson correlations
            pearson_corr = metrics_df.corr(method='pearson')

            # Spearman correlations
            spearman_corr = metrics_df.corr(method='spearman')

            # Kendall correlations
            kendall_corr = metrics_df.corr(method='kendall')

            correlation_results = {
                'pearson': {
                    'correlation_matrix': pearson_corr.to_dict(),
                    'description': 'Linear correlations between metrics'
                },
                'spearman': {
                    'correlation_matrix': spearman_corr.to_dict(),
                    'description': 'Rank-order correlations between metrics'
                },
                'kendall': {
                    'correlation_matrix': kendall_corr.to_dict(),
                    'description': 'Tau correlations between metrics'
                }
            }

            # Significant correlations
            correlation_results['significant_correlations'] = self._find_significant_correlations(
                metrics_df
            )

        return correlation_results

    def _power_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical power analysis
        """

        power_results = {}
        metrics_data = self._extract_comparable_metrics(evaluation_results)

        for metric_name, data in metrics_data.items():
            if len(data['word2vec']) > 0 and len(data['modernbert']) > 0:

                w2v_scores = np.array(data['word2vec'])
                bert_scores = np.array(data['modernbert'])

                # Calculate effect size for power analysis
                effect_size = self._calculate_cohens_d(w2v_scores, bert_scores)

                # Current sample size
                n = min(len(w2v_scores), len(bert_scores))

                # Calculate achieved power
                try:
                    achieved_power = ttest_power(effect_size, n, self.alpha)
                except:
                    achieved_power = None

                # Calculate required sample size for 80% power
                try:
                    required_n_80 = stats.power.ttest_power(effect_size, power=0.8, alpha=self.alpha)
                except:
                    required_n_80 = None

                # Calculate required sample size for 90% power
                try:
                    required_n_90 = stats.power.ttest_power(effect_size, power=0.9, alpha=self.alpha)
                except:
                    required_n_90 = None

                power_results[metric_name] = {
                    'effect_size': effect_size,
                    'current_sample_size': n,
                    'achieved_power': achieved_power,
                    'required_n_80_power': required_n_80,
                    'required_n_90_power': required_n_90,
                    'power_adequate': achieved_power >= 0.8 if achieved_power else False
                }

        return power_results

    def _descriptive_statistics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics
        """

        descriptive_stats = {}
        metrics_data = self._extract_comparable_metrics(evaluation_results)

        for metric_name, data in metrics_data.items():

            metric_stats = {}

            for model_name in ['word2vec', 'modernbert']:
                if len(data[model_name]) > 0:
                    scores = np.array(data[model_name])

                    metric_stats[model_name] = {
                        'count': len(scores),
                        'mean': np.mean(scores),
                        'median': np.median(scores),
                        'std': np.std(scores, ddof=1),
                        'var': np.var(scores, ddof=1),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'q25': np.percentile(scores, 25),
                        'q75': np.percentile(scores, 75),
                        'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                        'skewness': stats.skew(scores),
                        'kurtosis': stats.kurtosis(scores),
                        'cv': np.std(scores, ddof=1) / np.mean(scores) if np.mean(scores) != 0 else 0
                    }

            descriptive_stats[metric_name] = metric_stats

        return descriptive_stats

    def _confidence_intervals(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence intervals for metrics
        """

        ci_results = {}
        metrics_data = self._extract_comparable_metrics(evaluation_results)

        confidence_level = 1 - self.alpha

        for metric_name, data in metrics_data.items():

            metric_cis = {}

            for model_name in ['word2vec', 'modernbert']:
                if len(data[model_name]) > 0:
                    scores = np.array(data[model_name])

                    # Bootstrap confidence interval
                    bootstrap_ci = self._bootstrap_ci(scores, confidence_level)

                    # t-distribution confidence interval
                    t_ci = self._t_confidence_interval(scores, confidence_level)

                    metric_cis[model_name] = {
                        'bootstrap_ci': bootstrap_ci,
                        't_distribution_ci': t_ci,
                        'confidence_level': confidence_level
                    }

            ci_results[metric_name] = metric_cis

        return ci_results

    def _multiple_comparisons_correction(self, significance_tests: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply multiple comparisons correction
        """

        # Collect all p-values
        p_values = []
        test_names = []

        for metric_name, metric_tests in significance_tests.items():
            if 'tests' in metric_tests:
                for test_name, test_result in metric_tests['tests'].items():
                    if 'p_value' in test_result:
                        p_values.append(test_result['p_value'])
                        test_names.append(f"{metric_name}_{test_name}")

        if len(p_values) == 0:
            return {'error': 'No p-values found for correction'}

        p_values = np.array(p_values)

        # Bonferroni correction
        bonferroni_rejected, bonferroni_corrected = multipletests(
            p_values, alpha=self.alpha, method='bonferroni'
        )[:2]

        # Benjamini-Hochberg (FDR) correction
        fdr_rejected, fdr_corrected = multipletests(
            p_values, alpha=self.alpha, method='fdr_bh'
        )[:2]

        # Holm correction
        holm_rejected, holm_corrected = multipletests(
            p_values, alpha=self.alpha, method='holm'
        )[:2]

        correction_results = {
            'original_p_values': p_values.tolist(),
            'test_names': test_names,
            'bonferroni': {
                'corrected_p_values': bonferroni_corrected.tolist(),
                'rejected': bonferroni_rejected.tolist(),
                'num_significant': sum(bonferroni_rejected)
            },
            'fdr_bh': {
                'corrected_p_values': fdr_corrected.tolist(),
                'rejected': fdr_rejected.tolist(),
                'num_significant': sum(fdr_rejected)
            },
            'holm': {
                'corrected_p_values': holm_corrected.tolist(),
                'rejected': holm_rejected.tolist(),
                'num_significant': sum(holm_rejected)
            }
        }

        return correction_results

    # Helper methods for statistical calculations
    def _test_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if data follows normal distribution"""
        if len(data) < 3:
            return False

        try:
            _, p_value = shapiro(data)
            return p_value > alpha
        except:
            return False

    def _test_equal_variances(self, data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if two groups have equal variances"""
        try:
            _, p_value = levene(data1, data2)
            return p_value > alpha
        except:
            return False

    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)

        if n1 == 0 or n2 == 0:
            return 0.0

        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                              (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _calculate_glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's delta effect size"""
        if len(group2) == 0 or np.std(group2, ddof=1) == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)

    def _calculate_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedge's g (bias-corrected Cohen's d)"""
        cohens_d = self._calculate_cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)

        if n1 + n2 <= 3:
            return cohens_d

        # Bias correction factor
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))

        return cohens_d * correction_factor

    def _calculate_cles(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Common Language Effect Size"""
        if len(group1) == 0 or len(group2) == 0:
            return 0.5

        count = 0
        total = 0

        for x in group1:
            for y in group2:
                total += 1
                if x > y:
                    count += 1
                elif x == y:
                    count += 0.5

        return count / total if total > 0 else 0.5

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d magnitude"""
        abs_d = abs(d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _bootstrap_ci(self, data: np.ndarray, confidence_level: float, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return (ci_lower, ci_upper)

    def _t_confidence_interval(self, data: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Calculate t-distribution confidence interval"""
        if len(data) < 2:
            return (np.mean(data), np.mean(data))

        mean = np.mean(data)
        sem = stats.sem(data)

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(data) - 1)

        margin_error = t_critical * sem

        return (mean - margin_error, mean + margin_error)

    def _extract_comparable_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """Extract metrics that can be compared between models"""

        comparable_metrics = {}

        # Extract from intrinsic evaluation
        if 'intrinsic_evaluation' in evaluation_results:
            intrinsic = evaluation_results['intrinsic_evaluation']

            # Word similarity correlations
            if 'word_similarity' in intrinsic:
                ws = intrinsic['word_similarity']
                comparable_metrics['word_similarity_correlation'] = {
                    'word2vec': [ws.get('word2vec', {}).get('spearman_correlation', 0)],
                    'modernbert': [ws.get('modernbert', {}).get('spearman_correlation', 0)]
                }

            # Analogy accuracy
            if 'analogies' in intrinsic:
                analogies = intrinsic['analogies']
                comparable_metrics['analogy_accuracy'] = {
                    'word2vec': [analogies.get('word2vec', {}).get('accuracy', 0)],
                    'modernbert': [analogies.get('modernbert', {}).get('accuracy', 0)]
                }

            # Clustering quality
            if 'clustering' in intrinsic:
                clustering = intrinsic['clustering']
                comparable_metrics['clustering_silhouette'] = {
                    'word2vec': [clustering.get('word2vec', {}).get('avg_silhouette', 0)],
                    'modernbert': [clustering.get('modernbert', {}).get('avg_silhouette', 0)]
                }

        # Extract from extrinsic evaluation
        if 'extrinsic_evaluation' in evaluation_results:
            extrinsic = evaluation_results['extrinsic_evaluation']

            # Classification performance
            if 'classification' in extrinsic:
                classification = extrinsic['classification']

                for clf_name in ['logistic_regression', 'random_forest', 'svm']:
                    if (clf_name in classification.get('word2vec', {}) and
                            clf_name in classification.get('modernbert', {})):
                        w2v_f1 = classification['word2vec'][clf_name].get('f1_scores', [])
                        bert_f1 = classification['modernbert'][clf_name].get('f1_scores', [])

                        comparable_metrics[f'classification_{clf_name}_f1'] = {
                            'word2vec': w2v_f1,
                            'modernbert': bert_f1
                        }

        return comparable_metrics

    def _create_metrics_dataframe(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Create DataFrame with all metrics for correlation analysis"""

        metrics_data = []

        # This is a simplified version - in practice, you'd extract more metrics
        comparable_metrics = self._extract_comparable_metrics(evaluation_results)

        for metric_name, data in comparable_metrics.items():
            for model_name in ['word2vec', 'modernbert']:
                for value in data[model_name]:
                    metrics_data.append({
                        'metric': metric_name,
                        'model': model_name,
                        'value': value
                    })

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            # Pivot to get metrics as columns
            pivot_df = df.pivot_table(index='model', columns='metric', values='value', aggfunc='mean')
            return pivot_df
        else:
            return pd.DataFrame()

    def _find_significant_correlations(self, metrics_df: pd.DataFrame, alpha: float = 0.05) -> List[Dict]:
        """Find statistically significant correlations"""

        significant_correlations = []

        if metrics_df.empty:
            return significant_correlations

        columns = metrics_df.columns

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Avoid duplicate pairs
                    try:
                        corr, p_value = pearsonr(metrics_df[col1].dropna(),
                                                 metrics_df[col2].dropna())

                        if p_value < alpha:
                            significant_correlations.append({
                                'metric1': col1,
                                'metric2': col2,
                                'correlation': corr,
                                'p_value': p_value,
                                'significance': 'significant'
                            })
                    except:
                        continue

        return significant_correlations

    # Interpretation methods
    def _interpret_ttest(self, t_stat: float, p_value: float) -> str:
        """Interpret t-test results"""
        if p_value < self.alpha:
            if t_stat > 0:
                return "Word2Vec significantly outperforms ModernBERT"
            else:
                return "ModernBERT significantly outperforms Word2Vec"
        else:
            return "No significant difference between models"

    def _interpret_wilcoxon(self, w_stat: float, p_value: float) -> str:
        """Interpret Wilcoxon test results"""
        if p_value < self.alpha:
            return "Significant difference between models (non-parametric test)"
        else:
            return "No significant difference between models (non-parametric test)"

    def _interpret_mann_whitney(self, u_stat: float, p_value: float) -> str:
        """Interpret Mann-Whitney U test results"""
        if p_value < self.alpha:
            return "Significant difference between independent groups"
        else:
            return "No significant difference between independent groups"

    def generate_statistical_report(self) -> str:
        """Generate a comprehensive statistical report"""

        if not self.results:
            return "No statistical analysis results available"

        report = []
        report.append("# Statistical Analysis Report")
        report.append("=" * 50)

        # Significance tests summary
        if 'significance_tests' in self.results:
            report.append("\n## Significance Tests")
            report.append("-" * 30)

            for metric, tests in self.results['significance_tests'].items():
                report.append(f"\n### {metric}")

                if 'tests' in tests:
                    for test_name, test_result in tests['tests'].items():
                        if 'p_value' in test_result:
                            significance = "significant" if test_result['significant'] else "not significant"
                            report.append(f"- {test_name}: p = {test_result['p_value']:.4f} ({significance})")

        # Effect sizes summary
        if 'effect_sizes' in self.results:
            report.append("\n## Effect Sizes")
            report.append("-" * 30)

            for metric, effects in self.results['effect_sizes'].items():
                report.append(f"\n### {metric}")

                if 'cohens_d' in effects:
                    d_value = effects['cohens_d']['value']
                    magnitude = effects['cohens_d']['magnitude']
                    report.append(f"- Cohen's d: {d_value:.3f} ({magnitude} effect)")

        return "\n".join(report)


# Usage example
if __name__ == "__main__":
    analyzer = StatisticalAnalyzer(alpha=0.05)

    # Example evaluation results (would come from your evaluation pipeline)
    example_results = {
        'intrinsic_evaluation': {
            'word_similarity': {
                'word2vec': {'spearman_correlation': 0.65},
                'modernbert': {'spearman_correlation': 0.72}
            },
            'analogies': {
                'word2vec': {'accuracy': 0.45},
                'modernbert': {'accuracy': 0.58}
            }
        }
    }

    analysis = analyzer.analyze_comparison(example_results)
    report = analyzer.generate_statistical_report()
    print(report)
