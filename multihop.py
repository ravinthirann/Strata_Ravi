from extraction import get_model
from retrieval import build_store, search
from answer import ask

question = "Was the refund processed before or after the invoice was paid?"

if __name__ == "__main__":
    model = get_model()
    store = build_store()

    print("what got retrieved for this question:")
    chunks = search(store, question, k = 5)
    for text, doc_id, score in chunks:
        print(" ", doc_id, "|", round(score, 3), "|", text[:60])

    reply, ids, wrong = ask(model, store, question, k = 5)

    print("\nanswer:", reply.answer)
    print("citations:", reply.citations)

    if "A" in reply.citations and "B" in reply.citations:
        status = "PASS"
    else:
        status = "FAIL"

    print("result:", status)
