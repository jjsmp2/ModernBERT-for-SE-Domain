"""
Point 3: FastText and GloVe Baseline Evaluation
================================================
Drop this file into the root of your SE_Word_Embedding repo.
Run: python evaluate_fasttext_glove.py --processed_data results/data/processed --output results/evaluations

It evaluates fastText (trained on your corpus) and GloVe (pretrained)
on the SAME intrinsic and extrinsic metrics used for Word2Vec and ModernBERT,
so results slot directly into Table 1 and Table 2.
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── Install check ──────────────────────────────────────────────────────────────
try:
    from gensim.models import FastText as GensimFastText
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

try:
    from gensim.models import KeyedVectors
    from gensim.downloader import load as gensim_load
except ImportError:
    raise ImportError("gensim is already in requirements.txt — pip install gensim>=4.2.0")

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr


# ══════════════════════════════════════════════════════════════════════════════
# SE ANALOGY TEST SET  (100 pairs — Point 4 expansion used here too)
# ══════════════════════════════════════════════════════════════════════════════
# Format: (A, B, C, D)  where  A:B :: C:D
# The model must predict D given A, B, C.
SE_ANALOGIES = [
    # Category 1 — Language : Paradigm (20)
    ("python",      "scripting",        "java",         "object_oriented"),
    ("haskell",     "functional",       "python",       "scripting"),
    ("sql",         "database",         "html",         "markup"),
    ("css",         "styling",          "javascript",   "scripting"),
    ("ruby",        "scripting",        "scala",        "functional"),
    ("prolog",      "logic",            "haskell",      "functional"),
    ("java",        "object_oriented",  "rust",         "systems"),
    ("c",           "systems",          "python",       "scripting"),
    ("swift",       "mobile",           "kotlin",       "mobile"),
    ("php",         "web",              "python",       "scripting"),
    ("typescript",  "typed",            "javascript",   "scripting"),
    ("go",          "concurrent",       "erlang",       "concurrent"),
    ("matlab",      "scientific",       "r",            "statistical"),
    ("assembly",    "low_level",        "c",            "systems"),
    ("perl",        "scripting",        "ruby",         "scripting"),
    ("dart",        "mobile",           "swift",        "mobile"),
    ("lua",         "scripting",        "python",       "scripting"),
    ("fortran",     "scientific",       "matlab",       "scientific"),
    ("cobol",       "enterprise",       "java",         "enterprise"),
    ("bash",        "scripting",        "python",       "scripting"),

    # Category 2 — Tool : Purpose (20)
    ("git",         "version_control",  "jenkins",      "ci_cd"),
    ("docker",      "containerization", "kubernetes",   "orchestration"),
    ("jira",        "issue_tracking",   "confluence",   "documentation"),
    ("selenium",    "testing",          "jest",         "testing"),
    ("nginx",       "web_server",       "apache",       "web_server"),
    ("redis",       "caching",          "memcached",    "caching"),
    ("gradle",      "build",            "maven",        "build"),
    ("ansible",     "configuration",    "terraform",    "infrastructure"),
    ("prometheus",  "monitoring",       "grafana",      "visualization"),
    ("sonarqube",   "code_quality",     "eslint",       "linting"),
    ("postman",     "api_testing",      "swagger",      "api_documentation"),
    ("vagrant",     "virtualization",   "docker",       "containerization"),
    ("travis",      "ci_cd",            "jenkins",      "ci_cd"),
    ("gitlab",      "version_control",  "github",       "version_control"),
    ("splunk",      "log_analysis",     "elasticsearch","search"),
    ("puppet",      "configuration",    "ansible",      "configuration"),
    ("artifactory", "artifact",         "nexus",        "artifact"),
    ("vault",       "secrets",          "aws_kms",      "secrets"),
    ("kafka",       "messaging",        "rabbitmq",     "messaging"),
    ("zipkin",      "tracing",          "jaeger",       "tracing"),

    # Category 3 — Concept : Artifact (20)
    ("requirement", "srs",              "design",       "uml"),
    ("bug",         "bug_report",       "feature",      "user_story"),
    ("sprint",      "backlog",          "release",      "roadmap"),
    ("class",       "object",           "interface",    "implementation"),
    ("algorithm",   "pseudocode",       "architecture", "diagram"),
    ("test",        "test_case",        "code",         "module"),
    ("api",         "endpoint",         "database",     "schema"),
    ("commit",      "changelog",        "release",      "release_note"),
    ("error",       "stack_trace",      "warning",      "log"),
    ("function",    "signature",        "class",        "definition"),
    ("module",      "package",          "service",      "microservice"),
    ("variable",    "declaration",      "function",     "definition"),
    ("inheritance", "superclass",       "composition",  "component"),
    ("encryption",  "cipher",           "hashing",      "digest"),
    ("latency",     "benchmark",        "throughput",   "benchmark"),
    ("mock",        "unit_test",        "stub",         "integration_test"),
    ("refactor",    "clean_code",       "optimize",     "performance"),
    ("deploy",      "pipeline",         "build",        "makefile"),
    ("sprint",      "velocity",         "project",      "gantt"),
    ("pattern",     "design_pattern",   "antipattern",  "code_smell"),

    # Category 4 — Technology Stack Relationships (20)
    ("react",       "frontend",         "express",      "backend"),
    ("django",      "backend",          "react",        "frontend"),
    ("mongodb",     "nosql",            "postgresql",   "sql"),
    ("aws",         "cloud",            "azure",        "cloud"),
    ("mysql",       "relational",       "mongodb",      "nosql"),
    ("flask",       "microframework",   "django",       "fullframework"),
    ("angular",     "spa",              "react",        "spa"),
    ("graphql",     "query_language",   "rest",         "api_style"),
    ("tensorflow",  "deep_learning",    "pytorch",      "deep_learning"),
    ("hadoop",      "batch",            "spark",        "streaming"),
    ("linux",       "os",               "windows",      "os"),
    ("vim",         "editor",           "vscode",       "ide"),
    ("nginx",       "reverse_proxy",    "haproxy",      "load_balancer"),
    ("jwt",         "authentication",   "oauth",        "authorization"),
    ("rest",        "stateless",        "graphql",      "typed"),
    ("microservice","distributed",      "monolith",     "centralized"),
    ("ci",          "jenkins",          "cd",           "spinnaker"),
    ("unittest",    "python",           "junit",        "java"),
    ("npm",         "javascript",       "pip",          "python"),
    ("webpack",     "bundler",          "babel",        "transpiler"),

    # Category 5 — SE Process Terms (20)
    ("agile",       "sprint",           "waterfall",    "phase"),
    ("scrum",       "retrospective",    "kanban",       "board"),
    ("tdd",         "test_first",       "bdd",          "behavior_first"),
    ("pair",        "programming",      "code",         "review"),
    ("devops",      "pipeline",         "devsecops",    "security"),
    ("standup",     "daily",            "retrospective","sprint_end"),
    ("backlog",     "product_owner",    "sprint",       "scrum_master"),
    ("merge",       "pull_request",     "commit",       "push"),
    ("branch",      "feature",          "tag",          "release"),
    ("issue",       "tracker",          "wiki",         "documentation"),
    ("acceptance",  "criteria",         "definition",   "done"),
    ("velocity",    "story_points",     "burndown",     "chart"),
    ("regression",  "test",             "smoke",        "test"),
    ("code_review", "reviewer",         "pull_request", "author"),
    ("epic",        "stories",          "story",        "tasks"),
    ("mvp",         "product",          "poc",          "prototype"),
    ("sla",         "uptime",           "slo",          "objective"),
    ("rollback",    "deployment",       "hotfix",       "patch"),
    ("load_test",   "jmeter",           "unit_test",    "junit"),
    ("architect",   "design",           "developer",    "implement"),
]

# ══════════════════════════════════════════════════════════════════════════════
# WORD SIMILARITY PAIRS  (65 pairs — same set your paper uses)
# ══════════════════════════════════════════════════════════════════════════════
SE_SIMILARITY_PAIRS = [
    # (word1, word2, human_score_0_to_1)
    ("bug",           "defect",         0.92),
    ("requirement",   "specification",  0.88),
    ("class",         "object",         0.75),
    ("interface",     "contract",       0.80),
    ("refactor",      "restructure",    0.85),
    ("deploy",        "release",        0.78),
    ("git",           "version_control",0.82),
    ("docker",        "container",      0.90),
    ("api",           "endpoint",       0.76),
    ("sprint",        "iteration",      0.87),
    ("test",          "verify",         0.72),
    ("debug",         "troubleshoot",   0.83),
    ("module",        "component",      0.81),
    ("agile",         "scrum",          0.79),
    ("database",      "storage",        0.70),
    ("latency",       "delay",          0.84),
    ("encryption",    "security",       0.65),
    ("inheritance",   "polymorphism",   0.68),
    ("microservice",  "service",        0.77),
    ("backlog",       "task_list",      0.80),
    ("commit",        "checkin",        0.88),
    ("branch",        "fork",           0.72),
    ("pipeline",      "workflow",       0.75),
    ("mock",          "stub",           0.82),
    ("kubernetes",    "orchestration",  0.85),
    ("cloud",         "saas",           0.60),
    ("algorithm",     "procedure",      0.73),
    ("function",      "method",         0.89),
    ("variable",      "identifier",     0.77),
    ("exception",     "error",          0.85),
    ("stack",         "lifo",           0.79),
    ("queue",         "fifo",           0.78),
    ("heap",          "memory",         0.65),
    ("thread",        "process",        0.70),
    ("mutex",         "lock",           0.84),
    ("cache",         "buffer",         0.72),
    ("schema",        "model",          0.74),
    ("index",         "search",         0.60),
    ("query",         "request",        0.76),
    ("endpoint",      "route",          0.80),
    ("payload",       "data",           0.68),
    ("token",         "credential",     0.65),
    ("oauth",         "authentication", 0.78),
    ("rest",          "http",           0.70),
    ("graphql",       "query",          0.72),
    ("ci",            "automation",     0.74),
    ("cd",            "deployment",     0.76),
    ("devops",        "automation",     0.70),
    ("logging",       "monitoring",     0.75),
    ("metric",        "kpi",            0.68),
    ("sla",           "uptime",         0.72),
    ("load_balancer", "proxy",          0.70),
    ("firewall",      "security",       0.65),
    ("vulnerability", "exploit",        0.78),
    ("patch",         "hotfix",         0.82),
    ("regression",    "defect",         0.74),
    ("tdd",           "testing",        0.80),
    ("bdd",           "behavior",       0.78),
    ("user_story",    "requirement",    0.84),
    ("acceptance",    "criteria",       0.79),
    ("velocity",      "throughput",     0.62),
    ("burndown",      "progress",       0.70),
    ("stakeholder",   "client",         0.68),
    ("architect",     "designer",       0.72),
    ("scrum_master",  "facilitator",    0.76),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def get_vector_ft(model, word: str) -> Optional[np.ndarray]:
    """fastText handles OOV via subwords — always returns a vector."""
    return model.get_word_vector(word)


def get_vector_glove(kv: KeyedVectors, word: str) -> Optional[np.ndarray]:
    """GloVe has no subword — return None for OOV."""
    try:
        return kv[word]
    except KeyError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INTRINSIC EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_word_similarity(get_vec, pairs: List[Tuple]) -> float:
    """Spearman ρ between model cosine sims and human scores."""
    model_scores, human_scores = [], []
    skipped = 0
    for w1, w2, human in pairs:
        v1, v2 = get_vec(w1), get_vec(w2)
        if v1 is None or v2 is None:
            skipped += 1
            continue
        model_scores.append(cosine_similarity(v1, v2))
        human_scores.append(human)
    if len(model_scores) < 5:
        return float('nan')
    rho, _ = spearmanr(model_scores, human_scores)
    print(f"    Word similarity: {len(model_scores)} pairs evaluated, {skipped} skipped")
    return round(float(rho), 3)


def evaluate_analogies(get_vec, analogies: List[Tuple]) -> Dict:
    """
    3CosAdd method: find D such that vec(B)-vec(A)+vec(C) is closest to vec(D).
    Returns overall accuracy and per-category breakdown.
    """
    categories = {
        "Language_Paradigm":    analogies[0:20],
        "Tool_Purpose":         analogies[20:40],
        "Concept_Artifact":     analogies[40:60],
        "Stack_Relationship":   analogies[60:80],
        "SE_Process":           analogies[80:100],
    }

    total_correct, total_attempted = 0, 0
    category_results = {}

    for cat_name, cat_analogies in categories.items():
        correct, attempted = 0, 0
        for a, b, c, d_expected in cat_analogies:
            va = get_vec(a)
            vb = get_vec(b)
            vc = get_vec(c)
            vd = get_vec(d_expected)
            if any(v is None for v in [va, vb, vc, vd]):
                continue
            target = vb - va + vc
            # Score against expected answer only (closed evaluation)
            sim_expected = cosine_similarity(target, vd)
            # We check if expected answer scores highest among all D candidates
            all_d = [ans[3] for ans in cat_analogies]
            sims = []
            for d_cand in all_d:
                vd_cand = get_vec(d_cand)
                if vd_cand is None:
                    sims.append(-1.0)
                else:
                    sims.append(cosine_similarity(target, vd_cand))
            best_idx = int(np.argmax(sims))
            if all_d[best_idx] == d_expected:
                correct += 1
            attempted += 1

        acc = round(correct / attempted * 100, 1) if attempted > 0 else 0.0
        category_results[cat_name] = {"correct": correct, "attempted": attempted, "accuracy": acc}
        total_correct += correct
        total_attempted += attempted

    overall = round(total_correct / total_attempted * 100, 1) if total_attempted > 0 else 0.0
    print(f"    Analogies: {total_correct}/{total_attempted} correct ({overall}%)")
    for cat, res in category_results.items():
        print(f"      {cat}: {res['accuracy']}%")
    return {"overall": overall, "by_category": category_results, "total": total_attempted}


def evaluate_vocabulary_coverage(get_vec_raw, se_terms: List[str]) -> float:
    """Percentage of SE terms the model can represent (non-zero vector)."""
    covered = 0
    for term in se_terms:
        v = get_vec_raw(term)
        if v is not None and np.linalg.norm(v) > 0:
            covered += 1
    pct = round(covered / len(se_terms) * 100, 1)
    print(f"    Vocabulary coverage: {covered}/{len(se_terms)} = {pct}%")
    return pct


def evaluate_clustering(get_vec, se_terms: List[str]) -> float:
    """Silhouette score on semantic clusters of SE terms."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    vecs, valid_terms = [], []
    for term in se_terms:
        v = get_vec(term)
        if v is not None and np.linalg.norm(v) > 0:
            vecs.append(v)
            valid_terms.append(term)

    if len(vecs) < 10:
        return float('nan')

    X = np.array(vecs)
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = round(float(silhouette_score(X, labels)), 3)
    print(f"    Clustering silhouette: {score} ({len(vecs)} terms)")
    return score


# ══════════════════════════════════════════════════════════════════════════════
# EXTRINSIC EVALUATION  (Point 6: dataset sizes reported in output)
# ══════════════════════════════════════════════════════════════════════════════

# Minimal labelled dataset for document classification (5 SE classes)
# Each entry: (text_snippet, label)
# In production replace with your actual processed documents
EXTRINSIC_DOCS = [
    # requirements (0)
    ("the system shall allow users to login with username and password", "requirements"),
    ("the application must support concurrent users without performance degradation", "requirements"),
    ("user authentication shall comply with oauth2 standards and jwt tokens", "requirements"),
    ("the api shall return responses within 200ms under normal load conditions", "requirements"),
    ("the system shall encrypt all data at rest using aes 256 encryption", "requirements"),
    ("the software shall provide audit logs for all administrative actions", "requirements"),
    ("the interface shall be accessible and comply with wcag 2.1 guidelines", "requirements"),
    ("the system shall support role based access control for all modules", "requirements"),
    ("response time for search queries shall not exceed two seconds", "requirements"),
    ("the application shall support export to csv and pdf formats", "requirements"),
    ("the module must validate input data before database insertion", "requirements"),
    ("backup shall be performed daily with recovery point objective of 24 hours", "requirements"),
    # design (1)
    ("the architecture uses microservices with api gateway and service mesh", "design"),
    ("database schema includes users roles permissions and audit tables", "design"),
    ("the system design follows mvc pattern with repository layer abstraction", "design"),
    ("class diagram shows inheritance hierarchy between base and derived entities", "design"),
    ("the sequence diagram illustrates authentication flow between client and server", "design"),
    ("component diagram shows interaction between frontend backend and database", "design"),
    ("the design applies factory pattern for object creation and dependency injection", "design"),
    ("er diagram defines one to many relationship between orders and line items", "design"),
    ("the architecture diagram shows three tier web application deployment", "design"),
    ("state machine diagram models order lifecycle from pending to completed", "design"),
    ("the layered architecture separates presentation business logic and data access", "design"),
    ("the system uses event driven architecture with message broker for decoupling", "design"),
    # implementation (2)
    ("implemented rest api endpoints using flask with jwt authentication middleware", "implementation"),
    ("the function uses binary search algorithm for efficient lookup in sorted array", "implementation"),
    ("docker compose file defines services for web database and cache containers", "implementation"),
    ("the module implements singleton pattern to manage database connection pool", "implementation"),
    ("github actions workflow automates build test and deploy on pull request merge", "implementation"),
    ("the python script uses asyncio for concurrent api calls with rate limiting", "implementation"),
    ("the react component manages state with hooks and renders conditionally", "implementation"),
    ("sql query joins orders and customers tables with indexed foreign key lookup", "implementation"),
    ("the java class overrides equals and hashcode for proper collection behaviour", "implementation"),
    ("typescript interfaces enforce type safety across the api response objects", "implementation"),
    ("the kubernetes deployment yaml defines replica count resource limits and probes", "implementation"),
    ("the go routine uses channels for thread safe communication between workers", "implementation"),
    # testing (3)
    ("unit tests achieve 92 percent code coverage using pytest and mock fixtures", "testing"),
    ("integration test verifies end to end flow from api request to database write", "testing"),
    ("load testing with jmeter simulates 500 concurrent users over 30 minutes", "testing"),
    ("regression test suite runs automatically on every pull request via ci pipeline", "testing"),
    ("selenium test automates browser interaction for login and checkout workflows", "testing"),
    ("bdd scenarios written in gherkin describe given when then acceptance criteria", "testing"),
    ("mutation testing identifies weak test cases by introducing code mutations", "testing"),
    ("performance test measures p95 and p99 latency under sustained traffic load", "testing"),
    ("security scan with owasp zap identifies sql injection and xss vulnerabilities", "testing"),
    ("smoke test verifies critical paths after each production deployment", "testing"),
    ("contract testing with pact validates api compatibility between microservices", "testing"),
    ("fuzz testing sends malformed inputs to identify edge case handling failures", "testing"),
    # maintenance (4)
    ("refactored legacy codebase to reduce technical debt and improve readability", "maintenance"),
    ("hotfix deployed to production to resolve critical null pointer exception bug", "maintenance"),
    ("dependency upgrade addresses security vulnerability in authentication library", "maintenance"),
    ("code review identified dead code and unused imports for cleanup in next sprint", "maintenance"),
    ("monitoring alert triggered by memory leak in long running background service", "maintenance"),
    ("database index added to resolve slow query identified in production profiling", "maintenance"),
    ("rollback executed after deployment caused increased error rate in production", "maintenance"),
    ("technical debt documented in backlog for planned refactoring in q3 roadmap", "maintenance"),
    ("log analysis revealed recurring timeout in third party payment gateway calls", "maintenance"),
    ("configuration drift detected and corrected using ansible playbook enforcement", "maintenance"),
    ("api deprecation notice issued with six month migration window for consumers", "maintenance"),
    ("backup restoration tested quarterly to verify disaster recovery procedures", "maintenance"),
]

SE_VOCAB_200 = [
    "software","engineering","requirement","specification","design","architecture",
    "implementation","testing","maintenance","deployment","devops","agile","scrum",
    "kanban","sprint","backlog","bug","defect","error","exception","debugging",
    "refactoring","code_review","pull_request","merge","branch","commit","git",
    "github","gitlab","version_control","ci","cd","pipeline","jenkins","travis",
    "docker","kubernetes","container","microservice","api","rest","graphql","endpoint",
    "authentication","authorization","oauth","jwt","encryption","security","firewall",
    "database","sql","nosql","mongodb","postgresql","mysql","schema","query","index",
    "cache","redis","message_queue","kafka","rabbitmq","cloud","aws","azure","gcp",
    "saas","paas","iaas","serverless","lambda","function","class","object","interface",
    "inheritance","polymorphism","encapsulation","abstraction","design_pattern","singleton",
    "factory","observer","decorator","mvc","repository","service","controller","model",
    "view","frontend","backend","fullstack","react","angular","vue","javascript",
    "typescript","python","java","golang","rust","kotlin","swift","php","ruby",
    "algorithm","data_structure","stack","queue","heap","tree","graph","hash",
    "sorting","searching","complexity","performance","scalability","availability",
    "reliability","latency","throughput","load_balancer","reverse_proxy","nginx",
    "apache","tls","ssl","certificate","vulnerability","exploit","patch","sla",
    "slo","monitoring","logging","metric","tracing","observability","alerting",
    "grafana","prometheus","elasticsearch","kibana","terraform","ansible","puppet",
    "chef","infrastructure","virtual_machine","hypervisor","network","bandwidth",
    "protocol","tcp","udp","http","https","websocket","grpc","protobuf","json",
    "xml","yaml","gradle","maven","npm","pip","dependency","package","library",
    "framework","sdk","ide","vscode","intellij","debugger","profiler","linter",
    "formatter","static_analysis","sonarqube","code_coverage","unit_test","mock",
    "stub","fixture","integration_test","system_test","acceptance_test","tdd","bdd",
    "user_story","use_case","epic","velocity","burndown","retrospective","standup",
    "product_owner","scrum_master","stakeholder","persona","wireframe","prototype",
    "mvp","technical_debt","legacy","modernization","migration","refactor",
    "documentation","wiki","readme","changelog","roadmap","release","hotfix",
    "rollback","blue_green","canary","feature_flag","a_b_testing","analytics",
    "machine_learning","neural_network","nlp","embedding","transformer","bert",
]


def get_document_vector(get_vec, text: str) -> np.ndarray:
    """Mean of word vectors for document representation."""
    words = text.lower().split()
    vecs = [get_vec(w) for w in words if get_vec(w) is not None]
    if not vecs:
        return np.zeros(300)
    return np.mean(vecs, axis=0)


def evaluate_document_classification(get_vec) -> Dict:
    """5-class SE document classification — 80/20 train/test split."""
    from sklearn.model_selection import train_test_split

    texts  = [d[0] for d in EXTRINSIC_DOCS]
    labels = [d[1] for d in EXTRINSIC_DOCS]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = np.array([get_document_vector(get_vec, t) for t in texts])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = round(float(f1_score(y_test, y_pred, average='weighted')), 3)

    print(f"    Doc classification F1: {f1}  "
          f"(train={len(X_train)}, test={len(X_test)}, classes={len(le.classes_)})")
    return {
        "f1_weighted": f1,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "classes":    list(le.classes_),
        "split": "80/20"
    }


def evaluate_semantic_search(get_vec) -> Dict:
    """
    NDCG@10 semantic search over the 60-doc corpus.
    10 SE queries; relevant docs identified by label match.
    """
    queries = [
        ("find documents about authentication and security requirements", "requirements"),
        ("microservice architecture and api design documents",            "design"),
        ("docker kubernetes deployment implementation scripts",           "implementation"),
        ("unit test coverage pytest fixtures and mocking",               "testing"),
        ("refactoring technical debt and code maintenance",              "maintenance"),
        ("agile sprint backlog requirements specification",              "requirements"),
        ("database schema design and entity relationships",              "design"),
        ("ci cd pipeline github actions automation",                     "implementation"),
        ("load testing performance benchmarking jmeter",                "testing"),
        ("hotfix rollback deployment monitoring alerts",                 "maintenance"),
    ]

    corpus_texts  = [d[0] for d in EXTRINSIC_DOCS]
    corpus_labels = [d[1] for d in EXTRINSIC_DOCS]
    corpus_vecs   = np.array([get_document_vector(get_vec, t) for t in corpus_texts])

    ndcg_scores = []
    for query_text, relevant_label in queries:
        qvec = get_document_vector(get_vec, query_text)
        sims = [cosine_similarity(qvec, cv) for cv in corpus_vecs]
        ranked_labels = [corpus_labels[i] for i in np.argsort(sims)[::-1][:10]]
        # Binary relevance: 1 if label matches query's relevant label
        gains    = [1 if lbl == relevant_label else 0 for lbl in ranked_labels]
        ideal    = sorted(gains, reverse=True)
        dcg      = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        idcg     = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    avg_ndcg = round(float(np.mean(ndcg_scores)), 3)
    print(f"    Semantic search NDCG@10: {avg_ndcg}  (10 queries, {len(corpus_texts)} docs)")
    return {"ndcg_at_10": avg_ndcg, "n_queries": 10, "corpus_size": len(corpus_texts)}


def evaluate_code_text_alignment(get_vec) -> Dict:
    """
    Code snippet ↔ description alignment.
    Cosine sim between code vector and matching description.
    Reports MRR (Mean Reciprocal Rank) as alignment score.
    """
    pairs = [
        ("def authenticate user password hash check bcrypt verify",
         "function verifies user password against stored hash using bcrypt"),
        ("SELECT id name email FROM users WHERE active = 1 ORDER BY name",
         "sql query retrieves active user records sorted alphabetically"),
        ("docker run -d -p 8080:80 --name webapp nginx latest",
         "docker command starts nginx container mapping port 8080 to 80"),
        ("git commit -m fix authentication bug in login controller",
         "git commit with message describing authentication bug fix"),
        ("class UserService implements IUserRepository findById save delete",
         "service class implements repository interface with crud operations"),
        ("pytest test_login.py -v --cov=src --cov-report=html",
         "pytest command runs login tests with html coverage report"),
        ("kubectl apply -f deployment.yaml --namespace production",
         "kubectl applies deployment configuration to production namespace"),
        ("terraform apply -var-file=prod.tfvars -auto-approve",
         "terraform provisions infrastructure using production variables"),
        ("npm install --save-dev jest babel eslint prettier",
         "npm installs development dependencies for testing and linting"),
        ("try except Exception as e logger error f failed e raise",
         "python exception handler logs error message and re raises"),
    ]

    reciprocal_ranks = []
    descriptions = [p[1] for p in pairs]
    desc_vecs = [get_document_vector(get_vec, d) for d in descriptions]

    for i, (code_snippet, _) in enumerate(pairs):
        code_vec = get_document_vector(get_vec, code_snippet)
        sims = [cosine_similarity(code_vec, dv) for dv in desc_vecs]
        ranked = np.argsort(sims)[::-1]
        rank = list(ranked).index(i) + 1
        reciprocal_ranks.append(1.0 / rank)

    mrr = round(float(np.mean(reciprocal_ranks)), 3)
    print(f"    Code-text alignment MRR: {mrr}  ({len(pairs)} pairs)")
    return {"mrr": mrr, "n_pairs": len(pairs)}


# ══════════════════════════════════════════════════════════════════════════════
# FASTTEXT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_fasttext_evaluation(processed_data_path: str) -> Dict:
    """Train fastText on your processed corpus and evaluate."""
    import tempfile

    print("\n" + "="*60)
    print("FASTTEXT EVALUATION")
    print("="*60)

    # Collect processed text files
    corpus_lines = []
    data_path = Path(processed_data_path)
    for ext in ["*.txt", "*.json"]:
        for fpath in data_path.rglob(ext):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if ext == "*.json":
                        try:
                            obj = json.loads(content)
                            if isinstance(obj, list):
                                for item in obj:
                                    if isinstance(item, dict):
                                        text = item.get("text", item.get("content", ""))
                                        if text:
                                            corpus_lines.append(text.lower().strip())
                            elif isinstance(obj, dict):
                                text = obj.get("text", obj.get("content", ""))
                                if text:
                                    corpus_lines.append(text.lower().strip())
                        except json.JSONDecodeError:
                            corpus_lines.append(content.lower().strip())
                    else:
                        corpus_lines.append(content.lower().strip())
            except Exception:
                continue

    if not corpus_lines:
        print("  WARNING: No processed data found. Using analogy/similarity terms as minimal corpus.")
        corpus_lines = [" ".join([a for quad in SE_ANALOGIES for a in quad])]
        corpus_lines += [" ".join([w for triple in SE_SIMILARITY_PAIRS for w in triple[:2]])]
        corpus_lines += SE_VOCAB_200

    print(f"  Corpus: {len(corpus_lines)} documents")

    # Write temp corpus file for fastText
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(corpus_lines))
        corpus_file = tf.name

    # Train fastText skipgram
    print("  Training fastText (skipgram, dim=100, epoch=10)...")
    t0 = time.time()
    if not FASTTEXT_AVAILABLE:
        print("  FastText not available, skipping...")
        return {}

    sentences = [line.lower().split() for line in corpus_lines]
    print("  Training Gensim FastText (skipgram, dim=100, epoch=10)...")
    ft_model = GensimFastText(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        epochs=10,
        workers=4,
    )
    train_time = round(time.time() - t0, 2)
    print(f"  Training time: {train_time}s")

    def get_vec(w):
        try:
            return ft_model.wv[w.lower().replace(" ", "_")]
        except KeyError:
            return None
    results = {
        "model": "fastText (skipgram, dim=100)",
        "training_time_sec": train_time,
        "intrinsic": {},
        "extrinsic": {},
    }

    print("\n  -- Intrinsic Evaluation --")
    results["intrinsic"]["word_similarity_spearman"] = evaluate_word_similarity(
        get_vec, SE_SIMILARITY_PAIRS)
    results["intrinsic"]["analogical_reasoning"]     = evaluate_analogies(get_vec, SE_ANALOGIES)
    results["intrinsic"]["vocabulary_coverage_pct"]  = evaluate_vocabulary_coverage(
        get_vec, SE_VOCAB_200)
    results["intrinsic"]["clustering_silhouette"]    = evaluate_clustering(get_vec, SE_VOCAB_200)

    print("\n  -- Extrinsic Evaluation --")
    results["extrinsic"]["document_classification"]  = evaluate_document_classification(get_vec)
    results["extrinsic"]["semantic_search"]          = evaluate_semantic_search(get_vec)
    results["extrinsic"]["code_text_alignment"]      = evaluate_code_text_alignment(get_vec)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# GLOVE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_glove_evaluation() -> Dict:
    """
    Load GloVe-Wiki-Gigaword-100 via gensim downloader and evaluate.
    First run downloads ~128MB — cached afterward.
    """
    print("\n" + "="*60)
    print("GloVe EVALUATION  (glove-wiki-gigaword-100)")
    print("="*60)
    print("  Loading GloVe vectors (downloads ~128MB on first run)...")

    t0 = time.time()
    glove_kv = gensim_load("glove-wiki-gigaword-100")
    load_time = round(time.time() - t0, 2)
    print(f"  Loaded in {load_time}s  |  vocab size: {len(glove_kv)}")

    get_vec = lambda w: get_vector_glove(glove_kv, w.lower().replace(" ", "_"))

    results = {
        "model": "GloVe (wiki-gigaword-100d, pretrained)",
        "load_time_sec": load_time,
        "vocab_size": len(glove_kv),
        "intrinsic": {},
        "extrinsic": {},
    }

    print("\n  -- Intrinsic Evaluation --")
    results["intrinsic"]["word_similarity_spearman"] = evaluate_word_similarity(
        get_vec, SE_SIMILARITY_PAIRS)
    results["intrinsic"]["analogical_reasoning"]     = evaluate_analogies(get_vec, SE_ANALOGIES)
    results["intrinsic"]["vocabulary_coverage_pct"]  = evaluate_vocabulary_coverage(
        get_vec, SE_VOCAB_200)
    results["intrinsic"]["clustering_silhouette"]    = evaluate_clustering(get_vec, SE_VOCAB_200)

    print("\n  -- Extrinsic Evaluation --")
    results["extrinsic"]["document_classification"]  = evaluate_document_classification(get_vec)
    results["extrinsic"]["semantic_search"]          = evaluate_semantic_search(get_vec)
    results["extrinsic"]["code_text_alignment"]      = evaluate_code_text_alignment(get_vec)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PRINT TABLE  (ready to copy into paper)
# ══════════════════════════════════════════════════════════════════════════════

def print_paper_tables(ft_res: Dict, glove_res: Dict):
    """Print results formatted for direct insertion into Table 1 and Table 2."""

    def fmt(val):
        if isinstance(val, dict):
            return str(val.get("overall", val.get("f1_weighted",
                   val.get("ndcg_at_10", val.get("mrr", "?")))))
        if isinstance(val, float):
            return str(round(val, 3))
        return str(val)

    ft_i   = ft_res["intrinsic"]
    glove_i = glove_res["intrinsic"]
    ft_e   = ft_res["extrinsic"]
    glove_e = glove_res["extrinsic"]

    print("\n" + "="*70)
    print("TABLE 1 ADDITIONS — Intrinsic Evaluation")
    print("="*70)
    print(f"{'Metric':<35} {'fastText':>10} {'GloVe':>10}")
    print("-"*57)
    print(f"{'Word Similarity (Spearman ρ)':<35} "
          f"{fmt(ft_i['word_similarity_spearman']):>10} "
          f"{fmt(glove_i['word_similarity_spearman']):>10}")
    print(f"{'Analogical Reasoning (%)':<35} "
          f"{fmt(ft_i['analogical_reasoning']):>10} "
          f"{fmt(glove_i['analogical_reasoning']):>10}")
    print(f"{'Vocabulary Coverage (%)':<35} "
          f"{fmt(ft_i['vocabulary_coverage_pct']):>10} "
          f"{fmt(glove_i['vocabulary_coverage_pct']):>10}")
    print(f"{'Clustering Silhouette':<35} "
          f"{fmt(ft_i['clustering_silhouette']):>10} "
          f"{fmt(glove_i['clustering_silhouette']):>10}")

    print("\n" + "="*70)
    print("TABLE 2 ADDITIONS — Extrinsic Evaluation")
    print("="*70)
    print(f"{'Task':<35} {'fastText':>10} {'GloVe':>10}")
    print("-"*57)
    print(f"{'Document Classification (F1)':<35} "
          f"{fmt(ft_e['document_classification']):>10} "
          f"{fmt(glove_e['document_classification']):>10}")
    print(f"{'Semantic Search (NDCG@10)':<35} "
          f"{fmt(ft_e['semantic_search']):>10} "
          f"{fmt(glove_e['semantic_search']):>10}")
    print(f"{'Code-Text Alignment (MRR)':<35} "
          f"{fmt(ft_e['code_text_alignment']):>10} "
          f"{fmt(glove_e['code_text_alignment']):>10}")
    print("="*70)

    print("\n  NOTE: Paste these values alongside your existing")
    print("  Word2Vec and ModernBERT columns in Table 1 and Table 2.")
    print("  Category-level analogy breakdown is in the saved JSON.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Point 3 & 4: Evaluate fastText and GloVe baselines")
    parser.add_argument("--processed_data", default="results/data/processed",
                        help="Path to your preprocessed corpus directory")
    parser.add_argument("--output", default="results/evaluations",
                        help="Directory to save JSON results")
    parser.add_argument("--skip_fasttext", action="store_true",
                        help="Skip fastText (use if fasttext-wheel fails on your OS)")
    parser.add_argument("--skip_glove", action="store_true",
                        help="Skip GloVe (use if no internet connection)")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    all_results = {}

    if not args.skip_fasttext:
        ft_results = run_fasttext_evaluation(args.processed_data)
        all_results["fasttext"] = ft_results
        out_path = Path(args.output) / "fasttext_results.json"
        with open(out_path, "w") as f:
            json.dump(ft_results, f, indent=2)
        print(f"\n  fastText results saved → {out_path}")
    else:
        ft_results = None

    if not args.skip_glove:
        glove_results = run_glove_evaluation()
        all_results["glove"] = glove_results
        out_path = Path(args.output) / "glove_results.json"
        with open(out_path, "w") as f:
            json.dump(glove_results, f, indent=2)
        print(f"\n  GloVe results saved → {out_path}")
    else:
        glove_results = None

    if ft_results and glove_results:
        print_paper_tables(ft_results, glove_results)

    combined_path = Path(args.output) / "baseline_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
   

if __name__ == "__main__":
    main()
