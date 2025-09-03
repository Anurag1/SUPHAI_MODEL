# suhpai-model/supreme_context_manager.py

from sentence_transformers import SentenceTransformer
from transformers import pipeline, PeftModel, PeftConfig
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

class SupremeAnuvrittiContextManager:
    """
    Enhanced with contrastive learning for robust coref.
    """
    def __init__(self):
        self.context_frame = {
            "last_male_agent": None,
            "last_female_agent": None,
            "last_inanimate_object": None,
            "last_action": None,
        }
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.history_embeddings = []  # List of (text, emb) for history
        self.contrastive_loss = self._init_contrastive_loss()

        # LLM for coref augmentation
        self.coref_llm = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

    def _init_contrastive_loss(self):
        # Simple contrastive loss function for embedding refinement
        def contrastive_loss(emb1, emb2, label):
            margin = 1.0
            dist = np.sum(np.square(emb1 - emb2))
            return torch.max(torch.tensor(0.0), margin - dist) if label == 0 else torch.max(torch.tensor(0.0), dist - margin)
        return contrastive_loss

    def update_context(self, karaka_graph: Dict[Any, str]):
        """Updates context and embeds with contrastive learning."""
        agent = karaka_graph.get("KARTA")
        patient = karaka_graph.get("KARMA")
        
        if agent:
            emb = self.embedder.encode(agent)
            if agent.lower() in ["john", "david", "boy"]:
                self.context_frame["last_male_agent"] = (agent, emb)
            elif agent.lower() in ["mary", "alice", "girl"]:
                self.context_frame["last_female_agent"] = (agent, emb)
            else:
                self.context_frame["last_inanimate_object"] = (agent, emb)
        
        if patient:
            self.context_frame["last_inanimate_object"] = (patient, self.embedder.encode(patient))
        
        if karaka_graph.get("Action"):
            self.context_frame["last_action"] = karaka_graph["Action"]
        
        graph_str = " ".join([f"{k}: {v}" for k, v in karaka_graph.items()])
        new_emb = self.embedder.encode(graph_str)
        self.history_embeddings.append((graph_str, new_emb))
        # Refine with contrastive loss (simplified)
        if len(self.history_embeddings) > 1:
            prev_emb = self.history_embeddings[-2][1]
            label = 1 if cosine_similarity([new_emb], [prev_emb])[0][0] > 0.6 else 0
            loss = self.contrastive_loss(torch.tensor(new_emb), torch.tensor(prev_emb), label)
            if loss > 0:
                self.history_embeddings[-1] = (graph_str, new_emb - loss.numpy() * 0.01)  # Small adjustment

    def resolve_pronouns(self, text: str) -> str:
        """Resolves using rules + contrastive embedding similarity."""
        resolved_text = text
        pron_emb = self.embedder.encode(text)
        
        candidates = {
            ("he", "him"): self.context_frame.get("last_male_agent"),
            ("she", "her"): self.context_frame.get("last_female_agent"),
            ("it", "that"): self.context_frame.get("last_inanimate_object")
        }
        
        for prons, candidate in candidates.items():
            if candidate:
                cand_name, cand_emb = candidate
                score = cosine_similarity([pron_emb], [cand_emb])[0][0]
                if score > 0.7:
                    for pron in prons:
                        resolved_text = resolved_text.replace(f" {pron} ", f" {cand_name} ")
        
        if "he" in resolved_text or "she" in resolved_text or "it" in resolved_text:
            best_match = max(self.history_embeddings, key=lambda x: cosine_similarity([pron_emb], [x[1]])[0][0], default=None)
            if best_match and cosine_similarity([pron_emb], [best_match[1]])[0][0] > 0.75:
                resolved_text = resolved_text.replace("he", best_match[0].split()[0])
                resolved_text = resolved_text.replace("she", best_match[0].split()[0])
                resolved_text = resolved_text.replace("it", best_match[0].split()[0])
        
        return resolved_text
