# suhpai-model/supreme_context_manager.py

from sentence_transformers import SentenceTransformer
from transformers import pipeline, PeftModel, PeftConfig
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SupremeAnuvrittiContextManager:
    """
    Enhanced with vector similarity for robust coref.
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
        # LLM for coref augmentation (per OpenReview 2025)
        self.coref_llm = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")
        # LoRA for fine-tuning
        peft_config = PeftConfig.from_pretrained("path/to/lora_adapter")  # Assume prepped
        self.coref_llm.model = PeftModel.from_pretrained(self.coref_llm.model, "path/to/lora_adapter")

    def update_context(self, karaka_graph: Dict[Any, str]):
        """Updates context and embeds."""
        agent = karaka_graph.get("KARTA")  # Simplified key access
        patient = karaka_graph.get("KARMA")
        
        if agent:
            emb = self.embedder.encode(agent)
            # Heuristic + embed
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
        
        # Add to history
        graph_str = " ".join([f"{k}: {v}" for k, v in karaka_graph.items()])
        self.history_embeddings.append((graph_str, self.embedder.encode(graph_str)))
        # Fine-tune LoRA on new interaction (simple update)
        # Example: torch.optim for one step on graph_str as input/target

    def resolve_pronouns(self, text: str) -> str:
        """Resolves using rules + vector similarity to history."""
        resolved_text = text
        pron_emb = self.embedder.encode(text)  # Embed whole for context
        
        # Enhanced replacement with similarity check
        candidates = {
            ("he", "him"): self.context_frame.get("last_male_agent"),
            ("she", "her"): self.context_frame.get("last_female_agent"),
            ("it", "that"): self.context_frame.get("last_inanimate_object")
        }
        
        for prons, candidate in candidates.items():
            if candidate:
                cand_name, cand_emb = candidate
                score = cosine_similarity([pron_emb], [cand_emb])[0][0]
                if score > 0.6:  # Threshold
                    for pron in prons:
                        resolved_text = resolved_text.replace(f" {pron} ", f" {cand_name} ")
        
        # Fallback to history search for ambiguity
        if "he" in resolved_text or "she" in resolved_text or "it" in resolved_text:
            best_match = max(self.history_embeddings, key=lambda x: cosine_similarity([pron_emb], [x[1]])[0][0], default=None)
            if best_match and cosine_similarity([pron_emb], [best_match[1]])[0][0] > 0.7:
                resolved_text = resolved_text.replace("he", best_match[0].split()[0])  # Rough extract
        
        # If ambiguity (score <0.8), use LLM
        if score < 0.8:
            prompt = f"Resolve coref in: {text} with history: {self.history_embeddings[-1][0] if self.history_embeddings else ''}"
            resolved = self.coref_llm(prompt, max_new_tokens=20)[0]['generated_text']
            resolved_text = resolved
        
        return resolved_text
