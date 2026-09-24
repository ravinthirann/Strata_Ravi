import re
from docs import DOCS
from dotenv import load_dotenv
from extraction import get_model, run_all

load_dotenv()

patterns = {
    "date" : r"\d{4}-\d{2}-\d{2}",
    "amount" : r"\$[\d]+\.\d{2}",
    "invoice_id" : r"[A-Z]+-?\d+"
}

source_trust = {
    "invoice_pdf" : 0.95,
    "email" : 0.7,
    "support_ticket" : 0.75,
    "crm_note" : 0.6,
    "log" : 0.9,
    "meeting_note" : 0.5
}

def structural_clarity(evidence_span, obs_type):
    obs_type = obs_type.lower()
    evidence = evidence_span.strip()

    if "date" in obs_type:
        pattern = patterns["date"]
    elif "amount" in obs_type:
        pattern = patterns["amount"]
    elif "invoice_id" in obs_type:
        pattern = patterns["invoice_id"]
    else:
        return 0.5

    if re.fullmatch(pattern, evidence):
        return 1.0
    elif re.search(pattern, evidence):
        return 0.5
    else:
        return 0.0

def find_context(evidence_span, text):
    sentence = re.split(r"(?<=[.!?])\s+",text)
    for s in sentence:
        if evidence_span in s:
            return s
    return text

semantic_prompt = """On a scale of 0 to 1, how unambiguous is this value given the context? Reply with just the number.
Value: {value}
Context: {context}"""

def semantic_clarity(model, evidence_span, text):
    context = find_context(evidence_span, text)
    prompt = semantic_prompt.format(value = evidence_span, context = context)
    response = model.invoke(prompt)
    return float(response.content.strip())

def src_trust(source_type):
    return source_trust.get(source_type, 0.5)

def compute_confidence(structural, semantic, trust):
    confidence = (
        0.4 * structural +
        0.3 * semantic +
        0.3 * trust
    )
    return round(confidence, 3)

def score_obs(model, doc, observation):
    structural = structural_clarity(observation.evidence_span, observation.type)
    semantic = semantic_clarity(model, observation.evidence_span, doc["text"])
    trust = src_trust(doc["source_type"])

    confidence = compute_confidence(structural, semantic, trust)

    return{
        "confidence" : confidence,
        "factors" : {
            "structural_clarity" : structural,
            "semantic_clarity" : semantic,
            "source_trust" : trust
        }
    }

if __name__ == "__main__":
    model = get_model()
    results = run_all(model)

    for doc in DOCS:
        print("\n---Docs", doc["id"], "---")
        for obs in results[doc["id"]]:
            score = score_obs(model, doc, obs)
            print(obs.type,"|", obs.extracted_value,"|", score)