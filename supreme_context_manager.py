# suhpai-model/suhpai_core.py

from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline
import asyncio
import platform

from supreme_karaka_parser import SupremeKarakaParser, Karaka
from expansive_vector_knowledge_graph import ExpansiveVectorKnowledgeGraph
from supreme_context_manager import SupremeAnuvrittiContextManager

class SUHPaiCore:
    def __init__(self, parser: SupremeKarakaParser):
        self.parser = parser
        self.evkg = ExpansiveVectorKnowledgeGraph()
        self.context = SupremeAnuvrittiContextManager()
        self.generator = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        self.question_word_to_karaka = {
            "who": Karaka.KARTA, "what": Karaka.KARMA, "where": Karaka.ADHIKARANA,
            "when": Karaka.KALA, "with what": Karaka.KARANA, "by whom": Karaka.KARTA,
            "to whom": Karaka.SAMPRADANA, "from where": Karaka.APADANA, "why": Karaka.HETU
        }
        self.hypothesis_task = None

    async def _generate_hypothesis(self):
        """Generate advanced hypotheses with Paninian and modern reasoning."""
        while True:
            if not self.evkg.events:
                await asyncio.sleep(1)
                continue

            fact_batch = self.evkg.events[-min(5, len(self.evkg.events)):]  # Batch of last 5 facts
            for fact in fact_batch:
                action = fact.get("Action")
                agent = fact.get(Karaka.KARTA)
                location = fact.get(Karaka.ADHIKARANA)
                time = fact.get(Karaka.KALA)
                purpose = fact.get(Karaka.HETU)

                # Paninian Hypothesis 1: Transitive Karaka Inference
                if agent and location:
                    similar_agents = self.evkg.query_inference("FIND_PEOPLE_IN_SAME_LOCATION", {"person": agent})
                    for similar_agent in similar_agents:
                        hypothesis = {
                            "Action": "is",
                            Karaka.KARTA: similar_agent,
                            Karaka.ADHIKARANA: location,
                            Karaka.KALA: time or "unknown"
                        }
                        if await self._validate_hypothesis(hypothesis):
                            self.evkg.add_fact(hypothesis)
                            print(f"Retained Hypothesis: {similar_agent} is at {location} at {time or 'unknown'}")
                        else:
                            print(f"Discarded Hypothesis: {similar_agent} is at {location} at {time or 'unknown'}")

                # Paninian Hypothesis 2: Action-Purpose Correlation
                if action and purpose and agent:
                    hypothesis = {
                        "Action": action,
                        Karaka.KARTA: agent,
                        Karaka.HETU: purpose,
                        Karaka.KARMA: fact.get(Karaka.KARMA, "unknown")
                    }
                    if await self._validate_hypothesis(hypothesis):
                        self.evkg.add_fact(hypothesis)
                        print(f"Retained Hypothesis: {agent} {action} for {purpose}")
                    else:
                        print(f"Discarded Hypothesis: {agent} {action} for {purpose}")

                # Modern Hypothesis 3: Temporal Pattern Inference
                if time and agent and len([f for f in self.evkg.events if f.get(Karaka.KALA) == time]) > 1:
                    hypothesis = {
                        "Action": "repeats",
                        Karaka.KARTA: agent,
                        Karaka.KALA: time
                    }
                    if await self._validate_hypothesis(hypothesis):
                        self.evkg.add_fact(hypothesis)
                        print(f"Retained Hypothesis: {agent} repeats at {time}")
                    else:
                        print(f"Discarded Hypothesis: {agent} repeats at {time}")

            await asyncio.sleep(2)  # Optimized interval

    async def _validate_hypothesis(self, hypothesis: Dict[Any, str]) -> bool:
        """Advanced validation with Paninian rules and GNN."""
        action = hypothesis.get("Action")
        agent = hypothesis.get(Karaka.KARTA)
        location = hypothesis.get(Karaka.ADHIKARANA)
        time = hypothesis.get(Karaka.KALA)
        purpose = hypothesis.get(Karaka.HETU)

        if not action or not agent:
            return False

        # Paninian Rule Check: Karaka consistency
        if location and any(f.get(Karaka.ADHIKARANA) and f.get(Karaka.ADHIKARANA) != location for f in self.evkg.events if f.get(Karaka.KARTA) == agent):
            return False
        if time and any(f.get(Karaka.KALA) and f.get(Karaka.KALA) != time for f in self.evkg.events if f.get(Karaka.KARTA) == agent and f.get("Action") == action):
            return False
        if purpose and not any(f.get(Karaka.HETU) == purpose for f in self.evkg.events if f.get(Karaka.KARTA) == agent):
            return False

        # GNN-enhanced embedding check
        hyp_str = " ".join([f"{k}: {v}" for k, v in hypothesis.items()])
        hyp_emb = self.evkg.embedder.encode(hyp_str)
        node_features = torch.tensor([self.evkg.embeddings.get(n, np.zeros(384)) for n in self.evkg.graph.nodes])
        edge_index = torch.tensor([[i, j] for i, (u, v, d) in enumerate(self.evkg.graph.edges(data=True)) for j, _ in enumerate(self.evkg.graph.nodes) if v == _], dtype=torch.long).t()
        gnn_out = self.evkg.gnn_model(node_features, edge_index)
        max_sim = max([cosine_similarity([hyp_emb], [emb + gnn_out[i].numpy()])[0][0] for i, (fact_str, emb) in enumerate(self.evkg.history_embeddings)], default=0)
        return max_sim > 0.8  # Tighter threshold

    def _identify_query_karakas(self, query_graph: Dict[Any, str]) -> Tuple[Dict[Any, str], Optional[Karaka]]:
        """Identify query karakas with Paninian extensions."""
        cleaned_query = {"Action": query_graph.get("Action")}
        target_karaka = None
        for role, value in query_graph.items():
            if role == "Action": continue
            is_question = any(q_word in value.lower() for q_word in self.question_word_to_karaka)
            if is_question:
                for q_word, karaka_type in self.question_word_to_karaka.items():
                    if q_word in value.lower():
                        target_karaka = karaka_type
                        break
            else:
                cleaned_query[role] = value
        return cleaned_query, target_karaka
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Local generation with Paninian context."""
        response = self.generator(prompt, max_new_tokens=60, temperature=0.3, do_sample=False)
        return response[0]['generated_text'].strip()

    def process_query(self, query_text: str) -> str:
        # Resolve with supreme context
        resolved_query = self.context.resolve_pronouns(query_text)
        
        # Parse with advanced rules
        query_graphs = self.parser.parse(resolved_query)
        if not query_graphs:
            return self._generate_llm_response(f"User: '{query_text}'. Couldn't parse structure. Rephrase please.")

        final_response = ""
        for query_graph in query_graphs:
            cleaned_query, target_karaka = self._identify_query_karakas(query_graph)
            
            if not target_karaka:
                self.evkg.add_fact(query_graph)
                self.context.update_context(query_graph)
                self.evkg.update_embeddings(" ".join([f"{k}: {v}" for k, v in query_graph.items()]))
                final_response += "Noted and adapted. "
                continue

            if "same city" in resolved_query.lower() or "near" in resolved_query.lower():
                person = cleaned_query.get(Karaka.KARTA)
                results = self.evkg.query_inference("FIND_PEOPLE_IN_SAME_LOCATION", {"person": person})
                if results:
                    final_response += f"{', '.join(results)} are in/near the same location as {person}. "
                else:
                    final_response += f"No info on others near {person}. "
                continue

            answer = self.evkg.query_direct_fact(cleaned_query, target_karaka)
            if answer:
                prompt = f"""
                Helpful AI. Fact: Action '{cleaned_query.get('Action', '')}', Agent '{cleaned_query.get(Karaka.KARTA, 'unknown')}', Object '{cleaned_query.get(Karaka.KARMA, 'unknown')}'.
                User wants '{target_karaka}' which is '{answer}'.
                Short, direct sentence with Paninian clarity.
                """
                response_part = self._generate_llm_response(prompt)
                final_response += response_part + " "
                
                full_fact = cleaned_query.copy()
                full_fact[target_karaka] = answer
                self.context.update_context(full_fact)
                self.evkg.update_embeddings(" ".join([f"{k}: {v}" for k, v in full_fact.items()]))
            else:
                final_response += "No matching info found. "

        return final_response.strip()

    async def run_hypothesis_engine(self):
        """Run the hypothesis engine in parallel with optimization."""
        if self.hypothesis_task is None or self.hypothesis_task.done():
            self.hypothesis_task = asyncio.create_task(self._generate_hypothesis())
        if platform.system() == "Emscripten":
            asyncio.ensure_future(self._generate_hypothesis())
        else:
            if __name__ == "__main__":
                await self.hypothesis_task
