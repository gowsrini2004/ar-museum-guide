"""
Knowledge grounding system using RAG
Ensures LLM only uses curator-verified information
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class KnowledgeGrounder:
    """RAG-based knowledge grounding to prevent hallucinations"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_api_key_here":
            self.client = OpenAI(api_key=api_key)
            self.use_llm = True
        else:
            self.client = None
            self.use_llm = False
            print("⚠️  OpenAI API key not found. Using template responses.")
    
    def generate_explanation(self, artifact, user_interest="general overview"):
        """
        Generate grounded explanation from curator-verified knowledge
        
        Args:
            artifact: Artifact dict with knowledge entries
            user_interest: What aspect user is interested in
        
        Returns:
            Grounded explanation with source attribution
        """
        # Extract curator-verified knowledge
        knowledge_entries = artifact.get("knowledge", [])
        
        if not knowledge_entries:
            return {
                "explanation": "No verified information available for this artifact.",
                "sources": [],
                "grounded": True
            }
        
        # Construct context from verified sources
        context = self._build_context(knowledge_entries)
        
        if self.use_llm:
            explanation = self._generate_with_llm(artifact, context, user_interest)
        else:
            explanation = self._generate_template(artifact, knowledge_entries)
        
        # Extract sources
        sources = [entry["source"] for entry in knowledge_entries]
        
        return {
            "explanation": explanation,
            "sources": sources,
            "grounded": True,
            "num_sources": len(sources)
        }
    
    def _build_context(self, knowledge_entries):
        """Build context string from knowledge entries"""
        context_parts = []
        for entry in knowledge_entries:
            context_parts.append(
                f"[{entry['type'].upper()}] (Source: {entry['source']})\n{entry['content']}"
            )
        return "\n\n".join(context_parts)
    
    def _generate_with_llm(self, artifact, context, user_interest):
        """Generate explanation using LLM with strict grounding"""
        
        system_prompt = """You are a museum guide assistant. Your role is to explain artifacts based ONLY on the provided verified information from museum curators.

CRITICAL RULES:
1. Use ONLY information from the provided context below
2. If information is not in the context, say "This information is not currently available"
3. Do not make assumptions or add external knowledge
4. Keep explanations concise (3-4 sentences for AR display)
5. Use accessible language for general museum visitors
6. Cite sources naturally in your explanation

Remember: Accuracy is more important than completeness. Never hallucinate."""

        user_prompt = f"""Artifact: {artifact['name']}
Category: {artifact['category']}
Period: {artifact['period']}
Origin: {artifact['origin']}

User is interested in: {user_interest}

VERIFIED CURATOR INFORMATION:
{context}

Generate a brief, engaging explanation suitable for an AR museum guide display."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Low temperature for factual accuracy
                max_tokens=250
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return self._generate_template(artifact, artifact["knowledge"])
    
    def _generate_template(self, artifact, knowledge_entries):
        """Fallback template-based generation"""
        parts = [f"This is a {artifact['name']} from {artifact['origin']}, dating to {artifact['period']}."]
        
        for entry in knowledge_entries[:2]:  # Use first 2 entries
            parts.append(entry['content'])
        
        return " ".join(parts)


if __name__ == "__main__":
    # Test the grounder
    import json
    
    with open("data/sample_artifacts.json", 'r') as f:
        artifacts = json.load(f)
    
    grounder = KnowledgeGrounder()
    result = grounder.generate_explanation(artifacts[0])
    
    print("Explanation:")
    print(result["explanation"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  - {source}")
