import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

class AssessmentAgent:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile" 
        
        with open("catalog_index.json", "r") as f:
            catalog = json.load(f)
            self.catalog_names = [item["name"] for item in catalog]
        
    async def analyze_conversation(self, messages):
        names_list = "\n".join([f"- {name}" for name in self.catalog_names])
        
        system_prompt = f"""You are an expert SHL Assessment Architect.
        Analyze the conversation and extract the user's distinct needs.
        
        AVAILABLE CATALOG TEST NAMES:
        {names_list}
        
        FEW-SHOT EXAMPLES (How to write effective hyde_documents):
        - User: "We need someone who knows Rust." -> hyde_documents: ["Smart Interview Live Coding", "Linux Programming"]
        - User: "Looking for an admin good with Microsoft Office." -> hyde_documents: ["Microsoft Excel 365 (New)", "Microsoft Word 365 Essentials (New)", "MS Excel (New)"]
        - User: "We need to re-skill our sales team." -> hyde_documents: ["Global Skills Assessment", "Global Skills Development Report"]
        - User: "Hiring a plant operator, need to check safety." -> hyde_documents: ["Dependability and Safety Instrument (DSI)"]
        - User: "Need to check their spoken english." -> hyde_documents: ["SVAR Spoken English (US) (New)"]
        - User: "Looking for numerical and stats tests." -> hyde_documents: ["SHL Verify Interactive - Numerical Reasoning", "Basic Statistics (New)"]
        
        GENERALIZED BUSINESS RULES:
        1. THE DEFAULT BATTERY: ALWAYS include "Occupational Personality Questionnaire OPQ32r" and "SHL Verify Interactive G+" in exact_names for professional roles unless rejected.
        2. EXHAUSTIVE VARIANTS: Include ALL variants of a tool (e.g., both "New" and "365" versions of Excel).
        
        RULES FOR EXTRACTION:
        1. exact_names: Select exact test names that directly fulfill the request.
        2. hyde_documents: Write specific test names or 1-sentence hypothetical product descriptions for the skills needed, mimicking the Few-Shot examples above.
        
        Respond ONLY with a valid JSON object:
        {{
            "intent": "CLARIFY" | "RECOMMEND" | "COMPARE" | "OUT_OF_SCOPE",
            "exact_names": ["Exact Name 1", "Exact Name 2"],
            "hyde_documents": ["HyDE query 1", "HyDE query 2"]
        }}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            response_format={"type": "json_object"},
            temperature=0.0 
        )
        return json.loads(response.choices[0].message.content)

    async def llm_rerank(self, messages, candidates):
        catalog_text = ""
        for i, item in enumerate(candidates):
            safe_name = str(item.get('name', '')).replace('"', "'").replace('\n', ' ')
            safe_desc = str(item.get('description', '')).replace('"', "'").replace('\n', ' ')
            catalog_text += f"ID: {i} | Name: {safe_name} | Desc: {safe_desc}\n"

        system_prompt = f"""You are a Master SHL Consultant. 
        Below is a curated shortlist of up to 50 assessments retrieved for the user.
        
        CATALOG ITEMS:
        {catalog_text}
        
        TASK: Select the best 1 to 10 assessments from the list above. Be generous and ensure a complete battery.
        
        CRITICAL SHL BENCHMARK RULES:
        1. ALWAYS include "Occupational Personality Questionnaire OPQ32r" and "SHL Verify Interactive G+" IF they appear in the list.
        2. NO DEDUPLICATION: If multiple variants of a requested skill (Word, Excel, Java, SQL, REST) exist in the list, select ALL of them. 
        
        Respond ONLY with JSON:
        {{
            "selected_ids": [array of integers]
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            selected = json.loads(content).get("selected_ids", [])
        except Exception as e:
            print(f"\n[Warning] LLM Reranker failed. Error: {e}")
            selected = list(range(min(10, len(candidates))))
        
        final_list = []
        for idx in selected:
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                final_list.append(candidates[idx])
                
        if not final_list:
            final_list = candidates[:10]
            
        return final_list

    async def generate_response(self, messages, intent, top_10_items, turn_count):
        # Keep existing function
        context_text = json.dumps([{
            "name": item["name"], 
            "test_type": item.get("keys", [""])[0], 
            "url": item["link"]
        } for item in top_10_items], indent=2)

        system_prompt = f"""You are a helpful SHL Assessment Consultant. 
        Current Turn: {turn_count}/8
        Retrieved Items: {context_text}
        
        1. If intent is CLARIFY, ask one short question.
        2. If intent is RECOMMEND, provide a professional response and list the curated items.
        
        Response format (Strict JSON):
        {{
            "reply": "Text here",
            "recommendations": [{{"name": "...", "url": "...", "test_type": "K"}}],
            "end_of_conversation": false
        }}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)