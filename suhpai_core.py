# suhpai-model/suhpai_core.py

from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline

from supreme_karaka_parser import SupremeKarakaParser, Karaka
from expansive_vector_knowledge_graph import ExpansiveVectorKnowledgeGraph
from supreme_context_manager import SupremeAnuvrittiContextManager

class SUHPaiCore:
    def __init__(self, parser: SupremeKarakaParser):
        self.parser = parser
        self.evkg = ExpansiveVectorKnowledgeGraph()
        self.context = SupremeAnuvrittiContextManager()
        # Local lightweight LLM for generation (quantized Phi-3-mini)
        self.generator = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        # Quantize via Optimum (in init or separate script)
        # from optimum.onnxruntime import ORTModelForCausalLM
        # self.generator.model = ORTModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", export=True)

        self.question_word_to_karaka = {
            "who": Karaka.KARTA, "what": Karaka.KARMA, "where": Karaka.ADHIKARANA,
            "when": Karaka.ADHIKARANA, "with what": Karaka.KARANA, "by whom": Karaka.KARTA,
            "to whom": Karaka.SAMPRADANA, "from where": Karaka.APADANA, "why": "Purpose"  # Extended
        }

    def _identify_query_karakas(self, query_graph: Dict[Any, str]) -> Tuple[Dict[Any, str], Optional[Karaka]]:
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
        """Local generation, constrained."""
        response = self.generator(prompt, max_new_tokens=60, temperature=0.3, do_sample=False)
        return response[0]['generated_text'].strip()

    def process_query(self, query_text: str) -> str:
        # 1. Resolve with supreme context (multilingual handling)
        resolved_query = self.context.resolve_pronouns(query_text)
        
        # 2. Parse (multilingual)
        query_graphs = self.parser.parse(resolved_query)
        if not query_graphs:
            return self._generate_llm_response(f"User: '{query_text}'. Couldn't parse structure. Rephrase please.")

        final_response = ""
        for query_graph in query_graphs:
            cleaned_query, target_karaka = self._identify_query_karakas(query_graph)
            
            if not target_karaka:
                self.evkg.add_fact(query_graph)
                self.context.update_context(query_graph)
                self.evkg.update_embeddings(" ".join([f"{k}: {v}" for k, v in query_graph.items()]))  # Adaptive
                final_response += "Noted and adapted. "
                continue

            # 3. Inference with fuzzy/multi-hop
            if "same city" in resolved_query.lower() or "near" in resolved_query.lower():
                person = cleaned_query.get(Karaka.KARTA)
                results = self.evkg.query_inference("FIND_PEOPLE_IN_SAME_LOCATION", {"person": person})
                if results:
                    final_response += f"{', '.join(results)} are in/near the same location as {person}. "
                else:
                    final_response += f"No info on others near {person}. "
                continue

            # 4. Direct/fuzzy lookup
            answer = self.evkg.query_direct_fact(cleaned_query, target_karaka)
            if answer:
                prompt = f"""
                Helpful AI. Fact: Action '{cleaned_query.get('Action', '')}', Agent '{cleaned_query.get(Karaka.KARTA, 'unknown')}', Object '{cleaned_query.get(Karaka.KARMA, 'unknown')}'.
                User wants '{target_karaka}' which is '{answer}'.
                Short, direct sentence.
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
