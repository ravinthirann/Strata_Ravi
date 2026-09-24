from docs import DOCS
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq

load_dotenv()

class observation(BaseModel):
    type : str
    extracted_value : str
    derivation_method : str
    confidence : float
    evidence_span : str

class observationList(BaseModel):
    observations : list[observation]

def get_model():
    return ChatGroq(
        model = "openai/gpt-oss-120b",
        temperature = 0
    )

extract_prompt = """Extract every meaningful fact (dates, amounts, names, IDs, statuses))
- type : what type of infromation
- extracted_value : cleaned value
- derivation_method : how it was derived
- confidence : 0 to 1
- evidence_span : must be an exact text, verbatim substring copied from the text below do not reformat or paraphrase it.
Document : {text}"""

def valid_evidence(evidence_span, source_text):
    if evidence_span in source_text:
        return True
    else:
        return False

def extract_obs(model, doc):
    struct_model = model.with_structured_output(observationList)
    prompt = extract_prompt.format(text = doc["text"])
    answer = struct_model.invoke(prompt)
    return answer.observations

def run_all(model):
    results = {}
    for doc in DOCS:
        obs = extract_obs(model, doc)
        results[doc["id"]] = obs
    return results

def print_table(results):
    print("\n Doc ID | Type | Evidence Span | Status")

    for doc in DOCS :
        doc_id = doc["id"]

        for obs in results[doc_id]:
            passed = valid_evidence(obs.evidence_span, doc["text"])

            if passed:
                status = "PASS"
            else:
                status = "FAIL"

            print(doc_id,"|", obs.type,"|", obs.extracted_value,"|", obs.evidence_span,"|", status)

if __name__ == "__main__":
    model = get_model()
    results = run_all(model)
    print_table(results)