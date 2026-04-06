from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever
from google import genai
import os
import json

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def search(self, query: str, top_k: int = 10) -> list[str]:
        # We grab more results than needed (3x) to give the reranker 
        # a good pool to filter from.
        search_results = self.datastore.search(query, top_k=top_k * 3)
        
        if not search_results:
            return []
            
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(self, query: str, search_results: list[str], top_k: int = 10) -> list[str]:
        """
        Uses Gemini 2.0 Flash as a cross-encoder to rerank documents.
        """
        # Format the documents into a numbered list for the prompt
        docs_text = "\n".join([f"ID {i}: {doc}" for i, doc in enumerate(search_results)])
        
        prompt = f"""
        Evaluate the following documents and determine their relevance to the user's query.
        
        Query: {query}
        
        Documents:
        {docs_text}
        
        Task: Return a JSON list of the Document IDs, ordered from most relevant to least relevant. 
        Only include the IDs of the top {top_k} most relevant documents.
        Format: [ID_NUMBER, ID_NUMBER, ...]
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json', # Ensures we get valid JSON back
                }
            )
            
            # Parse the JSON list of indices
            result_indices = json.loads(response.text)
            
            # Handle potential edge cases where Gemini might return strings or nesting
            if isinstance(result_indices, dict):
                result_indices = list(result_indices.values())[0]

            print(f"✅ Gemini Reranked Indices: {result_indices}")
            
            # Return the original content in the new order
            return [search_results[int(i)] for i in result_indices if int(i) < len(search_results)]
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}. Falling back to original search order.")
            return search_results[:top_k]