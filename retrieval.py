from docs import DOCS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def build_store():
    embeddings = get_embeddings()

    texts = []
    ids = []
    metadata = []

    for doc in DOCS:
        texts.append(doc["text"])
        ids.append(doc["id"])
        metadata.append({"doc_id": doc["id"], "source_type": doc["source_type"]})

    store = Chroma.from_texts(
        texts = texts,
        embedding = embeddings,
        metadatas = metadata,
        ids = ids,
        collection_name = "docs",
        collection_metadata = {"hnsw:space": "cosine"}
    )
    return store

def search(store, query, k = 3):
    results = store.similarity_search_with_relevance_scores(query, k = k)
    output = []

    for doc, score in results:
        output.append((doc.page_content, doc.metadata["doc_id"], score))
    return output

test_queries = [
    ("When was the invoice payment received?", "B"),
    ("How many paid leave days do employees get each year?", "H"),
    ("Tell me about a coffee maker with a programmable timer", "L"),
    ("How do I reset my company password?", "P"),
    ("How long does the battery on the wireless earbuds last?", "M"),
]

def run_tests(store):
    print("query | expected | got | score | result")
    passed = 0

    for query, expected in test_queries:
        results = search(store, query, k = 3)
        top_doc = results[0][1]
        top_score = results[0][2]

        if top_doc == expected:
            status = "PASS"
            passed = passed + 1
        else:
            status = "FAIL"

        print(query, "|", expected, "|", top_doc, "|", round(top_score, 3), "|", status)
    print("\n", passed, "/", len(test_queries), "passed")

if __name__ == "__main__":
    store = build_store()
    run_tests(store)

    print("\nsearch: when was the invoice paid")
    for text, doc_id, score in search(store, "when was the invoice paid"):
        print(doc_id, "|", round(score, 3), "|", text[:60])
