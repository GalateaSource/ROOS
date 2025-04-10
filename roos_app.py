# UNIVERSAL ROOS LABORATORY FRAMEWORK (v4.3.2)
# Patch: Fix syntax error in config loader and gracefully handle matplotlib absence

import numpy as np
import sympy as sp
import pandas as pd
import json
import os
import yaml
from typing import Callable, Dict, List, Any, Union
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import datetime
import threading
import time
import hashlib
import networkx as nx  # For graph representation

# Try to import matplotlib but disable rendering if unavailable
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not available. Rendering disabled.")


# OUTPUT ENGINE
class ROOSOutputEngine:
    def __init__(self):
        self.figures = []

    def render(self, label: str, data: List[float], mode: str = "spiral"):
        """Renders data visualizations."""
        if not MATPLOTLIB_AVAILABLE:
            print("[RENDER DISABLED] matplotlib not available.")
            return
        fig, ax = plt.subplots()
        if mode == "spiral":
            theta = np.linspace(0, 4 * np.pi, len(data))
            r = np.array(data)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, label=label)
            ax.set_aspect('equal')
        elif mode == "line":
            ax.plot(data, label=label)
        elif mode == "scatter":
            ax.scatter(range(len(data)), data, label=label)
        ax.set_title(f"{label} Visualization")
        ax.legend()
        self.figures.append(fig)
        plt.show()

    def render_bot_divergence(self, inbot_memory: List[str], outbot_memory: List[str]):
        """Renders the divergence between bot memories over time."""
        if not MATPLOTLIB_AVAILABLE:
            print("[RENDER DISABLED] matplotlib not available.")
            return
        fig, ax = plt.subplots()
        in_lengths = [len(m) for m in inbot_memory]
        out_lengths = [len(m) for m in outbot_memory]
        ax.plot(in_lengths, label="Inbot Memory Length")
        ax.plot(out_lengths, label="Outbot Memory Length")
        ax.set_title("Bot Memory Divergence Over Time")
        ax.set_xlabel("Crawl Iteration")
        ax.set_ylabel("Memory Length")
        ax.legend()
        self.figures.append(fig)
        plt.show()

    def render_similarity_over_time(self, similarity_scores: List[float]):
        """Renders the trend of similarity scores over time."""
        if not MATPLOTLIB_AVAILABLE:
            print("[RENDER DISABLED] matplotlib not available.")
            return
        fig, ax = plt.subplots()
        ax.plot(similarity_scores, label="Divergence Score")
        ax.set_title("Similarity Score Trend")
        ax.set_xlabel("Evaluation Cycle")
        ax.set_ylabel("Similarity Score")
        ax.axhline(y=0.5, color='red', linestyle='--', label="Threshold")
        ax.legend()
        self.figures.append(fig)
        plt.show()

    def export_summary(self, monitor: Any, filepath: str = "roos_summary.json"):
        """Exports a summary of events and anomalies."""
        summary = {
            "events": monitor.events,
            "anomalies": monitor.anomalies
        }
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        return f"Summary exported to {filepath}"


# CONFIG loader with error fix
def load_config(config_path="roos_config.yaml") -> Dict:
    """Loads configuration from a YAML file."""
    default_config = {"divergence_threshold": 0.5, "crawl_interval_sec": 10}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        print("[CONFIG] File not found. Creating default config.")
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        return default_config
    except yaml.YAMLError as e:
        print(f"[CONFIG ERROR] Invalid YAML: {e}")
        return default_config


CONFIG = load_config()


# ============================
# RECURSIVE SEARCH ENGINE FOR SYMBOLIC PATH DISCOVERY
# ============================
class RecursiveSearchEngine:
    """
    Engine for recursively searching symbolic paths.
    """

    def __init__(self, strategy="entropy_harvest"):
        """Initializes the search engine."""
        self.strategy = strategy
        self.log = []

    def search(self, input_term, depth=0, path=None):
        """
        Performs a recursive search for symbolic paths.

        Args:
            input_term (str): The term to start the search from.
            depth (int, optional): The current depth of the search. Defaults to 0.
            path (list, optional): The current search path. Defaults to None.

        Returns:
            list: The discovered symbolic path, or None if no path is found.
        """
        if path is None:
            path = []

        self.log.append((depth, input_term))  # Log current step
        path.append(input_term)

        if depth >= 5 or self.terminate_condition(input_term):  # Stopping condition
            return path

        next_terms = self.expand_term(input_term)  # Strategy-based branching
        for term in next_terms:
            result = self.search(term, depth + 1, path.copy())
            if result:
                return result

        return None

    def expand_term(self, term):
        """
        Expands a term based on the chosen strategy.

        Args:
            term (str): The term to expand.

        Returns:
            list: A list of expanded terms.
        """
        if self.strategy == "entropy_harvest":
            return [f"{term}.{i}" for i in range(2)]  # Generative spread
        elif self.strategy == "heuristic_leap":
            return [f"{term}_alt"]  # Targeted fork
        else:
            return [f"{term}_next"]

    def terminate_condition(self, term):
        """
        Determines when the search should terminate.

        Args:
            term (str): The current term being evaluated.

        Returns:
            bool: True if the search should terminate, False otherwise.
        """
        return "goal" in term  # Placeholder: Replace with a more meaningful condition

    def get_log(self):
        """Returns the search log."""
        return self.log


# ============================
# PRIMES KEY FOR DIVERGENCE INTERPRETATION
# ============================
PRIME_LINES = {
    "Δ₁": {"score": 0.0000, "label": "Collapse to Singularity"},
    "Δ₂": {"score": 0.2838, "label": "Redemptive Recursion"},
    "Δ₃": {"score": 0.3200, "label": "Proto-Unity"},
    "Δ₄": {"score": 0.3600, "label": "Λₑ – Symbolic Tension Attractor"},
    "Δ₅a": {"score": 0.4098, "label": "Spacetime Shift"},
    "Δ₅b": {"score": 0.4387, "label": "Temporal Drift Acceleration"},
    "Δ₆": {"score": 0.5136, "label": "Semantic Fracture / GOM Trigger"},
    "ΔΩ": {"score": 0.5729, "label": "Zaphod Field / Ontological Comedy Collapse"}
}


# ============================
# RELATIONAL EVALUATOR ENGINE — PROTOTYPE
# ============================
class RelationalEvaluator:
    """
    Engine for evaluating relationships between symbolic inputs.
    """

    def __init__(self):
        """Initializes the relational evaluator."""
        self.relations = []  # stores (input_a, input_b, score)

    def evaluate_relation(self, a: str, b: str) -> float:
        """
        Compares two symbolic inputs and returns their relational divergence score.

        Args:
            a (str): The first symbolic input.
            b (str): The second symbolic input.

        Returns:
            float: The relational divergence score.
        """
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([a, b])
        score = 1 - cosine_similarity(tfidf[0], tfidf[1])[0][0]  # divergence score
        self.relations.append((a, b, score))
        return score

    def get_most_stable_relations(self, top_n=5):
        """Gets the most stable (low divergence) relations."""
        return sorted(self.relations, key=lambda x: x[2])[:top_n]  # lowest divergence

    def get_most_unstable_relations(self, top_n=5):
        """Gets the most unstable (high divergence) relations."""
        return sorted(self.relations, key=lambda x: x[2], reverse=True)[:top_n]  # highest divergence


# ============================
# TOMBOT BRAIN TEMPLATE
# ============================
class Tombot:
    """
    Agent that explores and analyzes symbolic relationships.
    """

    def __init__(self, name, polarity="coherent", search_engine=None, evaluator=None):
        """
        Initializes a Tombot.

        Args:
            name (str): The name of the Tombot.
            polarity (str, optional): The polarity of the Tombot ('coherent' or 'divergent'). Defaults to "coherent".
            search_engine (RecursiveSearchEngine, optional): The search engine to use. Defaults to None.
            evaluator (RelationalEvaluator, optional): The relational evaluator to use. Defaults to None.
        """
        self.name = name
        self.polarity = polarity  # 'coherent' or 'divergent'
        self.memory = []
        self.graph = nx.Graph()  # Internal graph to represent discovered relationships
        self.search_engine = search_engine
        self.evaluator = evaluator

    def think(self, input_term):
        """
        Processes an input term, explores its symbolic connections, and updates the internal graph.

        Args:
            input_term (str): The term to process.

        Returns:
            list: The symbolic path discovered during the thinking process.
        """
        path = self.search_engine.search(input_term)
        if not path:
            return []  # No path found

        self.memory.extend(path)  # Store the explored path
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            score = self.evaluator.evaluate_relation(a, b)  # Evaluate relation between consecutive terms
            self.graph.add_edge(a, b, weight=score)  # Add the relation to the graph
        return path

    def get_graph(self):
        """Returns the Tombot's internal graph."""
        return self.graph


# ============================
# MINIMAL BOOTSTRAP RUN
# ============================
if __name__ == "__main__":
    print("\nROOS Minimal Test Run")

    # Simulated monitor and output engine
    monitor = type("ROOSMonitor", (), {"events": [], "anomalies": []})()
    output_engine = ROOSOutputEngine()

    # Simulated bot memory
    inbot_memory = ["This is a test", "Another symbolic message"]
    outbot_memory = ["This is a test", "Another symbolic message", "Extra"]

    # Dummy bots
    inbot = type("Bot", (), {"memory": [(datetime.datetime.now(), m) for m in inbot_memory]})()
    outbot = type("Bot", (), {"memory": [(datetime.datetime.now(), m) for m in outbot_memory]})()

    # Dummy evaluator
    class Evaluator:
        def __init__(self, inbot, outbot, monitor):
            self.inbot = inbot
            self.outbot = outbot
            self.monitor = monitor
            self.divergence_threshold = CONFIG.get("divergence_threshold", 0.5)
            self.similarity_scores = []

        def evaluate_divergence(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            corpus = [m for _, m in self.inbot.memory + self.outbot.memory]
            tfidf_matrix = TfidfVectorizer().fit_transform(corpus)
            similarity_score = cosine_similarity(tfidf_matrix).mean()
            self.similarity_scores.append(similarity_score)
            return f"Similarity Score: {similarity_score:.4f}"

    evaluator = Evaluator(inbot, outbot, monitor)
    print(evaluator.evaluate_divergence())
