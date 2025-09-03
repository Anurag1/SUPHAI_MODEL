# suhpai-model/expansive_vector_knowledge_graph.py

import requests
from sentence_transformers import SentenceTransformer
import networkx as nx
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv

from supreme_karaka_parser import Karaka

class ExpansiveVectorKnowledgeGraph:
    """
    Enhanced vector-augmented KG with GNN and contrastive learning.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = {}  # entity: vector
        self.events = []
        self.gnn_model = self._init_gnn()  # Simple GNN for inference

        # Populate initial knowledge with Paninian structure
        self._populate_initial_knowledge()
        self._bootstrap_from_wikidata()

    def _init_gnn(self):
        # Simple GCN for graph reasoning
        class GNN(torch.nn.Module):
            def __init__(self):
                super(GNN, self).__init__()
                self.conv1 = GCNConv(16, 16)
                self.conv2 = GCNConv(16, 8)
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                return self.conv2(x, edge_index)
        return GNN()

    def _populate_initial_knowledge(self):
        initial_facts = [
            {"Action": "hit", Karaka.KARTA: "boy", Karaka.KARMA: "ball", Karaka.KARANA: "bat", Karaka.ADHIKARANA: "park"},
            {"Action": "sent", Karaka.KARTA: "John", Karaka.KARMA: "letter", Karaka.SAMPRADANA: "Mary", Karaka.APADANA: "Paris"},
            {"Action": "lives", Karaka.KARTA: "Mary", Karaka.ADHIKARANA: "London", Karaka.KALA: "now"},
            {"Action": "works", Karaka.KARTA: "David", Karaka.ADHIKARANA: "Paris", Karaka.HETU: "income"},
            {"Action": "opened", Karaka.KARTA: "key", Karaka.KARMA: "door"},
            {"Action": "drove", Karaka.KARTA: "Alice", Karaka.KARMA: "car", Karaka.ADHIKARANA: "market", Karaka.KALA: "morning"}
        ]
        for fact in initial_facts:
            self.add_fact(fact)

    def _bootstrap_from_wikidata(self):
        """Bootstrap with Wikidata subgraphs with modern pruning."""
        seeds = ["Q5", "Q82794"]  # Human, Location
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
                    fact = {"Action": "is", Karaka.KARTA: res['itemLabel']['value'], Karaka.ADHIKARANA: res.get('valueLabel', {}).get('value', '')}
                    self.add_fact(fact)
        # Modern pruning with TSNE for dimensionality reduction
        if len(self.graph.nodes) > 50:
            embeddings_array = np.array([self.embeddings.get(n, np.zeros(384)) for n in self.graph.nodes])
            tsne = TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings_array)
            to_remove = [n for i, n in enumerate(self.graph.nodes) if reduced_embeddings[i][0] < -5]  # Arbitrary threshold
            self.graph.remove_nodes_from(to_remove)

    def add_fact(self, karaka_graph: Dict[Any, str]):
        """Adds fact, updates graph and embeddings with contrastive learning."""
        self.events.append(karaka_graph)
        action = karaka_graph.get("Action")
        agent = karaka_graph.get(Karaka.KARTA)
        if agent and action:
            self.graph.add_edge(agent, action, relation="performs")
        
        fact_str = " ".join([f"{k}: {v}" for k, v in karaka_graph.items()])
        emb = self.embedder.encode(fact_str)
        self.embeddings[fact_str] = emb
        
        location = karaka_graph.get(Karaka.ADHIKARANA)
        if agent and location:
            self.graph.nodes[agent.lower()]['location'] = location
            loc_emb = self.embedder.encode(location)
            self.embeddings[agent.lower()] = loc_emb

    def query_direct_fact(self, query_graph: Dict[Any, str], target_karaka: Karaka) -> Optional[str]:
        """Direct lookup with GNN-enhanced fuzzy fallback."""
        query_str = " ".join([f"{k}: {v}" for k, v in query_graph.items()])
        query_emb = self.embedder.encode(query_str)
        
        best_match = None
        best_score = -1
        # Convert graph to GNN input (simplified)
        node_features = torch.tensor([self.embeddings.get(n, np.zeros(384)) for n in self.graph.nodes])
        edge_index = torch.tensor([[i, j] for i, (u, v, d) in enumerate(self.graph.edges(data=True)) for j, _ in enumerate(self.graph.nodes) if v == _], dtype=torch.long).t()
        gnn_out = self.gnn_model(node_features, edge_index)
        for i, (fact_str, fact_emb) in enumerate(self.embeddings.items()):
            if " ".join(fact_str.split()[:len(query_str.split())]) == query_str:
                fact = self.events[list(self.embeddings.keys()).index(fact_str)]
                return fact.get(target_karaka)
            score = cosine_similarity([query_emb], [fact_emb + gnn_out[i].numpy()])[0][0]
            if score > best_score and score > 0.8:
                best_score = score
                fact = self.events[list(self.embeddings.keys()).index(fact_str)]
                best_match = fact.get(target_karaka)
        return best_match

    def query_inference(self, query_type: str, params: Dict[str, str]) -> List[str]:
        """Multi-hop with GNN."""
        if query_type == "FIND_PEOPLE_IN_SAME_LOCATION":
            person = params.get("person", "").lower()
            if person not in self.graph.nodes or 'location' not in self.graph.nodes[person]:
                return []
            
            target_loc = self.graph.nodes[person]['location']
            target_emb = self.embeddings.get(person, self.embedder.encode(target_loc))
            
            similar_people = []
            node_features = torch.tensor([self.embeddings.get(n, np.zeros(384)) for n in self.graph.nodes])
            edge_index = torch.tensor([[i, j] for i, (u, v, d) in enumerate(self.graph.edges(data=True)) for j, _ in enumerate(self.graph.nodes) if v == _], dtype=torch.long).t()
            gnn_out = self.gnn_model(node_features, edge_index)
            for i, node in enumerate(self.graph.nodes):
                if node != person and 'location' in self.graph.nodes[node]:
                    node_emb = self.embeddings.get(node) + gnn_out[i].numpy()
                    if node_emb is not None:
                        score = cosine_similarity([target_emb], [node_emb])[0][0]
                        if score > 0.75:
                            similar_people.append(node.capitalize())
            return similar_people
        
        return []

    def update_embeddings(self, new_fact_str: str):
        """Adaptive learning with contrastive loss approximation."""
        new_emb = self.embedder.encode(new_fact_str)
        for existing_str, emb in list(self.embeddings.items()):
            if cosine_similarity([new_emb], [emb])[0][0] > 0.5:
                self.embeddings[existing_str] = (emb + new_emb) / 2  # Average update
