"""
glossary_lookup tool: RAG-based Singlish term retrieval for the Interpretation Agent.

Pipeline: embed query (Titan via Bedrock) -> cosine similarity search (Chroma, in-memory)
-> return top-k matched glossary entries as a formatted string for prompt injection.

Swap DUMMY_GLOSSARY for real curated entries once ready -- schema stays the same.
"""

import json
import boto3
import chromadb

from configs.settings import AWS_REGION



DUMMY_GLOSSARY = [
    {
        "id": "g001",
        "term": "paiseh",
        "language": "Hokkien",
        "meaning": "feeling shy, embarrassed, or reluctant to impose on others",
        "variants": ["pai seh", "pai-seh", "paisay"],
        "category": "caregiving_emotion",
        "example": "wa paiseh to ask for help"
    },
    {
        "id": "g002",
        "term": "buay tahan",
        "language": "Hokkien/Malay",
        "meaning": "cannot tolerate / cannot stand something anymore",
        "variants": ["bo tahan", "buay tahan liao"],
        "category": "caregiving_state",
        "example": "the pain until buay tahan already"
    },
    {
        "id": "g003",
        "term": "makan",
        "language": "Malay",
        "meaning": "to eat, or a meal",
        "variants": ["makan already", "jiak"],
        "category": "daily_living",
        "example": "have you makan yet?"
    },
    {
        "id": "g004",
        "term": "song bo",
        "language": "Hokkien",
        "meaning": "feeling good / comfortable, or asking if something feels good (often used post-massage or after relief from pain)",
        "variants": ["song", "song boh"],
        "category": "caregiving_state",
        "example": "after apply the cream, song bo?"
    },
    {
        "id": "g005",
        "term": "toa peh kong",
        "language": "Hokkien",
        "meaning": "colloquial/idiomatic phrase used when someone is being overly cautious or fussy; context-dependent",
        "variants": [],
        "category": "general",
        "example": "why you so toa peh kong about this"
    },
]


class TitanEmbedder:
    def __init__(self, region: str = AWS_REGION, model_id: str = "amazon.titan-embed-text-v2:0"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["embedding"]


_embedder = TitanEmbedder()
_chroma_client = chromadb.Client()  # in-memory, no persistence
_collection = _chroma_client.get_or_create_collection(name="singlish_glossary")


def _entry_to_embed_text(entry: dict) -> str:
    """Concatenate term + meaning + example for richer embedding signal."""
    parts = [entry["term"], entry["meaning"]]
    if entry.get("example"):
        parts.append(entry["example"])
    return ". ".join(parts)


def index_glossary(glossary: list[dict] = DUMMY_GLOSSARY) -> None:
    """Embed and store all glossary entries. Call once at startup."""
    ids = [entry["id"] for entry in glossary]
    documents = [_entry_to_embed_text(entry) for entry in glossary]
    embeddings = [_embedder.embed(doc) for doc in documents]
    metadatas = [
        {
            "term": entry["term"],
            "meaning": entry["meaning"],
            "language": entry["language"],
            "category": entry.get("category", ""),
            "example": entry.get("example", ""),
        }
        for entry in glossary
    ]

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )



def glossary_lookup(query: str, top_k: int = 3) -> str:
    """
    RAG lookup: embed the query, retrieve top-k most similar glossary entries
    by cosine similarity, return as a formatted string for prompt injection.
    """
    query_embedding = _embedder.embed(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    metadatas = results.get("metadatas", [[]])[0]

    if not metadatas:
        return "No relevant glossary terms found."

    lines = []
    for m in metadatas:
        line = f"- {m['term']} ({m['language']}): {m['meaning']}"
        if m.get("example"):
            line += f" [e.g. \"{m['example']}\"]"
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    index_glossary()
    print(glossary_lookup("ah ma paiseh to ask for help lah"))