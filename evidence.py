from extraction import get_model
from retrieval import build_store, search
from answer import ask

good_questions = [
    "What was paid on the invoice and when?",
    "How many paid leave days do employees get each year?",
    "How do I reset my company password?",
]

bad_questions = [
    "What's the company's parental leave policy?",
    "What is the capital of Australia?",
    "How do I file a tax return in Germany?",
]

def top_score(store, question, k = 3):
    chunks = search(store, question, k = k)

    if len(chunks) == 0:
        return 0.0

    scores = []
    for text, doc_id, score in chunks:
        scores.append(score)
    return max(scores)

def calibrate(store):
    print("good questions:")
    for question in good_questions:
        score = top_score(store, question)
        print(" ", question, "|", round(score, 3))

    print("\nbad questions:")
    for question in bad_questions:
        score = top_score(store, question)
        print(" ", question, "|", round(score, 3))

threshold = 0.5

def ask_safe(model, store, question, k = 3):
    score = top_score(store, question, k = k)

    if score < threshold:
        print("  score", round(score, 3), "is below", threshold, "- not calling the model")
        return {"answer": "insufficient evidence", "citations": []}

    reply, ids, wrong = ask(model, store, question, k = k)
    return {"answer": reply.answer, "citations": reply.citations}

if __name__ == "__main__":
    model = get_model()
    store = build_store()

    calibrate(store)

    print("\ntesting the gate:")
    print("\nbad question:")
    print(ask_safe(model, store, "What's the company's parental leave policy?"))
    print("\ngood question through the same gate:")
    print(ask_safe(model, store, "How do I reset my company password?"))
