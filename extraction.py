from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq
from docs import DOCS

load_dotenv()

class Observation(BaseModel):
    type: str
    extract_value: str
    derivation_method: str
    confidence: float
    evidence_span: str

class ObservationList(BaseModel):
    observations: List[Observation]

def get_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

extract_prompt = """Extract every meaningful fact from the document below.Look for things like dates, amounts, names, IDs, and statuses.
For each fact, fill in:
- type: a short label for what kind of fact it is
- extract_value: the cleaned-up value
- derivation_method: just put "llm_semantic"
- confidence: your own rough 0 to 1 guess
- evidence_span: this MUST be an exact, verbatim substring copied from the document text below. Do not reformat it and do not paraphrase it.
Document:
{text}
"""

def normalize(s):
    return s.strip().lower()

def valid_evidence(evi_span, src_txt):
    if evi_span in src_txt:
        return True
    else:
        return normalize(evi_span) in normalize(src_txt)

def extract_obs(model, doc):
    struct_model = model.with_structured_output(
        ObservationList
    )

    prompt = extract_prompt.format(text=doc["text"])

    try:
        answer = struct_model.invoke(prompt)
        return answer.observations
    except:
        return []

def run_all(model=None):
    if model is None:
        model = get_model()
    results = {}

    for doc in DOCS:
        obs = extract_obs(model, doc)
        results[doc["id"]] = obs

    return results

def print_table(results):
    print("\nDoc ID | Type | Value | Evidence Span | Status")
    print("-" * 85)

    for doc in DOCS:
        doc_id = doc["id"]

        for obs in results[doc_id]:
            passed = valid_evidence(
                obs.evidence_span,
                doc["text"]
            )

            if passed:
                status = "PASS"
            else:
                status = "FAIL"

            print(doc_id,"|", obs.type,"|", obs.extract_value, "|", obs.evidence_span,"|", status)

if __name__ == "__main__":
    model = get_model()
    r = run_all(model)
    print_table(r)