import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")


groq_client = Groq(api_key=GROQ_API_KEY)


faqs_path = Path(__file__).parent / "resource" / "faq_data.csv"


chroma_client = chromadb.Client()
collection_name_faq = 'faqs'


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)



def ingest_faq_data(path):
    """Ingest FAQs from CSV into ChromaDB."""
    existing_collections = [c.name for c in chroma_client.list_collections()]

    if collection_name_faq not in existing_collections:
        collection = chroma_client.get_or_create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )

        df = pd.read_csv(path)
        docs = df['question'].tolist()
        metadata = [{"answer": ans} for ans in df['answer'].tolist()]
        ids = [f"id_{i}" for i in range(len(docs))]

        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print(f"✅ FAQ data successfully ingested into collection '{collection_name_faq}'")
    else:
        print(f"⚠️ Collection '{collection_name_faq}' already exists")


def get_relevant_qa(query):
    """Retrieve top relevant FAQs from ChromaDB."""
    collection = chroma_client.get_collection(name=collection_name_faq)
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result


def generate_answer(query, context):
    """Generate answer using Groq based on provided context."""
    prompt = f"""
Given the question and context below, generate the answer based on the context only.
If you don't find the answer inside the context, say "I don't know". Do not make things up.

QUESTION: {query}

CONTEXT: {context}
"""
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL
    )
    return chat_completion.choices[0].message.content



def faq_chain(query):
    """Get answer to a query using ChromaDB + Groq."""
    result = get_relevant_qa(query)
    # Combine answers from metadata into readable context
    context = "\n".join(r.get("answer", "") for r in result["metadatas"][0])
    answer = generate_answer(query, context)
    return answer



if __name__ == "__main__":
    ingest_faq_data(faqs_path)

    query = "What's your policy on defective products?"
    answer = faq_chain(query)
    print("\n🔹 Answer:\n", answer)
