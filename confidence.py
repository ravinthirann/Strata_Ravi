import re
from dotenv import load_dotenv
from extraction import get_model, run_all
from docs import DOCS

load_dotenv()

PATTERNS = {
    "date": r"\d{4}-\d{2}-\d{2}",
    "amount": r"\$[\d,]+\.\d{2}",
    "id": r"[A-Z]+-?\d+"
}

SOURCE_TRUST = {
    "invoice_pdf": 0.95,
    "log": 0.9,
    "support_ticket": 0.75,
    "email": 0.7,
    "crm_note": 0.6,
    "meeting_note": 0.5
}

semantic_prompt = """On a scale of 0 to 1, how unambiguous is this value given the context? Reply with just the number. Value: {value} Context: {context}"""

def structural_clarity(evidence_span, obs_type):
    obs_type = obs_type.lower()
    if "date" in obs_type:
        pattern = PATTERNS["date"]
    elif "amount" in obs_type or "currency" in obs_type:
        pattern = PATTERNS["amount"]
    elif "ticket" in obs_type or "invoice" in obs_type or "id" in obs_type:
        pattern = PATTERNS["id"]
    else:
        return 0.5

    if re.fullmatch(pattern, evidence_span.strip()):
        return 1.0
    elif re.search(pattern, evidence_span):
        return 0.5
    else:
        return 0.0

def find_context(evidence_span, text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        if evidence_span in sentence:
            return sentence
    return text

def semantic_clarity(model, evidence_span, text):
    context = find_context(evidence_span, text)

    prompt = semantic_prompt.format(value=evidence_span, context=context)
    response = model.invoke(prompt)

    try:
        return float(response.content.strip())
    except:
        return 0.5

def source_trust(source_type):
    return SOURCE_TRUST.get(source_type, 0.5)

def compute_confidence(structural, semantic, trust):
    confidence = (
        0.4 * structural +
        0.3 * semantic +
        0.3 * trust
    )
    return round(confidence, 3)

def score_observation(model, doc, obs):
    structural = structural_clarity(
        obs.evidence_span,
        obs.type
    )

    semantic = semantic_clarity(
        model,
        obs.evidence_span,
        doc["text"]
    )

    trust = source_trust(doc["source_type"])

    confidence = compute_confidence(
        structural,
        semantic,
        trust
    )

    return {
        "confidence": confidence,
        "factors": {
            "structural_clarity": structural,
            "semantic_clarity": semantic,
            "source_trust": trust
        }
    }

if __name__ == "__main__":
    model = get_model()
    results = run_all(model)

    for doc in DOCS:
        print("\n--- Doc", doc["id"], "---")

        for obs in results[doc["id"]]:
            score = score_observation(model, doc, obs)
            print( obs.type, "|", obs.extract_value, "|", score)