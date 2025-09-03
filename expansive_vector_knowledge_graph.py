# suhpai-model/expansive_vector_knowledge_graph.py

import requests
from sentence_transformers import SentenceTransformer
import networkx as nx
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

from supreme_karaka_parser import Karaka

class ExpansiveVectorKnowledgeGraph:
    """
    Lightweight vector-augmented KG using NetworkX + embeddings.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight ~80MB
        self.embeddings = {}  # entity: vector
        self.events = []

        # Populate initial knowledge
        self._populate_initial_knowledge()
        self._bootstrap_from_wikidata()  # New: Auto-bootstrap

    def _populate_initial_knowledge(self):
        initial_facts = [
            {"Action": "hit", Karaka.KARTA: "boy", Karaka.KARMA: "ball", Karaka.KARANA: "bat", Karaka.ADHIKARANA: "park"},
            {"Action": "sent", Karaka.KARTA: "John", Karaka.KARMA: "letter", Karaka.SAMPRADANA: "Mary", Karaka.APADANA: "Paris"},
            {"Action": "lives", Karaka.KARTA: "Mary", Karaka.ADHIKARANA: "London"},
            {"Action": "works", Karaka.KARTA: "David", Karaka.ADHIKARANA: "Paris"},
            {"Action": "opened", Karaka.KARTA: "key", Karaka.KARMA: "door"},
            {"Action": "drove", Karaka.KARTA: "Alice", Karaka.KARMA: "car", Karaka.ADHIKARANA: "market"}
        ]
        for fact in initial_facts:
            self.add_fact(fact)

    def _bootstrap_from_wikidata(self):
        """Bootstrap with Wikidata subgraphs (per CEUR-WS techniques)."""
        seeds = ["Q5", "Q82794"]  # Human, Location; expand as needed
        for seed in seeds:
            query = f"""
            SELECT ?item ?itemLabel ?property ?value ?valueLabel WHERE {{
              ?item wdt:P31 wd:{seed}.
              ?item ?property ?value.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }} LIMIT 100
            """
            url = "https://query.wikidata.org/sparql"
            response = requests.get(url, params={'format': 'json', 'query': query})
            if response.status_code == 200:
                results = response.json()['results']['bindings']
                for res in results:
                    # Map to Kāraka: e.g., item as KARTA, value as ADHIKARANA
                    fact = {"Action": "is", Karaka.KARTA: res['itemLabel']['value'], Karaka.ADHIKARANA: res.get('valueLabel', {}).get('value', '')}
                    self.add_fact(fact)
        # Prune low-relevance (analogical pruning: remove nodes with degree <2)
        to_remove = [n for n in self.graph if self.graph.degree(n) < 2]
        self.graph.remove_nodes_from(to_remove)

    def add_fact(self, karaka_graph: Dict[Any, str]):
        """Adds fact, updates graph and embeddings."""
        self.events.append(karaka_graph)
        action = karaka_graph.get("Action")
        agent = karaka_graph.get(Karaka.KARTA)
        if agent and action:
            self.graph.add_edge(agent, action, relation="performs")
            # Add other relations...
        
        # Embed the entire graph as string
        fact_str = " ".join([f"{k}: {v}" for k, v in karaka_graph.items()])
        emb = self.embedder.encode(fact_str)
        self.embeddings[fact_str] = emb
        
        # Update entity props (e.g., location)
        location = karaka_graph.get(Karaka.ADHIKARANA)
        if agent and location:
            if agent.lower() not in self.graph.nodes:
                self.graph.add_node(agent.lower())
            self.graph.nodes[agent.lower()]['location'] = location
            # Embed location for fuzzy
            loc_emb = self.embedder.encode(location)
            self.embeddings[agent.lower()] = loc_emb

    def query_direct_fact(self, query_graph: Dict[Any, str], target_karaka: Karaka) -> Optional[str]:
        """Direct lookup with fuzzy fallback."""
        query_str = " ".join([f"{k}: {v}" for k, v in query_graph.items()])
        query_emb = self.embedder.encode(query_str)
        
        best_match = None
        best_score = -1
        for fact_str, fact_emb in self.embeddings.items():
            if " ".join(fact_str.split()[:len(query_str.split())]) == query_str:  # Exact prefix
                fact = self.events[list(self.embeddings.keys()).index(fact_str)]
                return fact.get(target_karaka)
            score = cosine_similarity([query_emb], [fact_emb])[0][0]
            if score > best_score and score > 0.8:  # Threshold for fuzzy
                best_score = score
                fact = self.events[list(self.embeddings.keys()).index(fact_str)]
                best_match = fact.get(target_karaka)
        return best_match

    def query_inference(self, query_type: str, params: Dict[str, str]) -> List[str]:
        """Multi-hop with vectors."""
        if query_type == "FIND_PEOPLE_IN_SAME_LOCATION":
            person = params.get("person", "").lower()
            if person not in self.graph.nodes or 'location' not in self.graph.nodes[person]:
                return []
            
            target_loc = self.graph.nodes[person]['location']
            target_emb = self.embeddings.get(person, self.embedder.encode(target_loc))
            
            similar_people = []
            for node in self.graph.nodes:
                if node != person and 'location' in self.graph.nodes[node]:
                    node_emb = self.embeddings.get(node)
                    if node_emb is not None:
                        score = cosine_similarity([target_emb], [node_emb])[0][0]
                        if score > 0.75:  # Fuzzy location match
                            similar_people.append(node.capitalize())
            return similar_people
        
        return []

    def update_embeddings(self, new_fact_str: str):
        """Adaptive learning: Fine-tune embeddings (simple average update)."""
        new_emb = self.embedder.encode(new_fact_str)
        for existing_str, emb in list(self.embeddings.items()):
            if cosine_similarity([new_emb], [emb])[0][0] > 0.5:  # Related
                self.embeddings[existing_str] = (emb + new_emb) / 2  # Average for adaptation
