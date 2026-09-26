Extraction :
    This script extracts useful information from different documents using an LLM.


Confidence :
    This script calculates the confidence of the extracted information using an LLM.


Strategy :
    This script chooses the best extraction method for each document using deterministic, heuristic, and semantic extraction.


Contradiction :
    This script checks when two documents disagree on the same fact and keeps both values instead of picking one.


Retrieval :
    This script stores documents in a vector database and finds the most relevant one for a given question.


Answer :
    This script answers a question using retrieved documents and names the exact doc_id it used as a citation.


Evidence :
    This script checks if the retrieved documents are relevant enough before answering, and says "insufficient evidence" instead of guessing when they are not.


Multihop :
    This script answers a question that needs facts from two different documents and checks that both documents are cited.