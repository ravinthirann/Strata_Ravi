from pydantic import BaseModel
from extraction import get_model
from retrieval import build_store, search

class Answer(BaseModel):
    answer: str
    citations: list[str]

def ask(model, store, question, k = 3):
    chunks = search(store, question, k = k)

    context = ""
    for text, doc_id, score in chunks:
        context = context + "[" + doc_id + "] " + text + "\n"

    prompt = """Answer the question using ONLY the context below.In citations, list the doc_id of every chunk you actually used.If you didn't use a chunk, don't cite it.Context:""" + context + """Question: """ + question

    struct_model = model.with_structured_output(Answer)
    result = struct_model.invoke(prompt)

    retrieved_ids = []
    for text, doc_id, score in chunks:
        retrieved_ids.append(doc_id)

    bad_citations = []
    for c in result.citations:
        if c not in retrieved_ids:
            bad_citations.append(c)
    return result, retrieved_ids, bad_citations

test_questions = [
    ("What was paid on the invoice and when?", "B"),
    ("How many paid leave days do employees get each year?", "H"),
    ("What does the coffee maker come with?", "L"),
    ("How do I reset my company password?", "P"),
    ("How long does the earbuds battery last?", "M"),
]

def run_tests(model, store):
    print("question | answer | citations | expected | result")

    for question, expected in test_questions:
        result, retrieved_ids, bad_citations = ask(model, store, question)

        if expected in result.citations and len(bad_citations) == 0:
            status = "PASS"
        else:
            status = "FAIL"

        print(question, "|", result.answer, "|", result.citations, "|", expected, "|", status)

        if len(bad_citations) > 0:
            print("  bad citation, model cited a doc it was never shown:", bad_citations)

if __name__ == "__main__":
    model = get_model()
    store = build_store()
    run_tests(model, store)
