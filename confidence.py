import re
from dotenv import load_dotenv
from extraction import get_model, run_all
from docs import DOCS

load_dotenv()

patterns = {
    "date": r"\d{4}-\d{2}-\d{2}",
    "amount": r"\$[\d,]+\.\d{2}",
    "invoice_id": r"[A-Z]+-?\d+",
    "ticket_id": r"#\d+",
    "phone_number": r"\+\d[\d-]{6,}"
}

src_trust = {
    "invoice_pdf": 0.95,
    "log": 0.9,
    "support_ticket": 0.75,
    "email": 0.7,
    "crm_note": 0.6,
    "meeting_note": 0.5
}

semantic_prompt = """On a scale of 0 to 1, how unambiguous is this value given the context? Reply with just the number. Value: {value} Context: {context}"""

def struct_clarity(evidence_span, obs_type):
    obs_type = obs_type.lower()
    evidence = evidence_span.strip()

    if "date" in obs_type:
        p = patterns["date"]
    elif "amount" in obs_type or "currency" in obs_type:
        p = patterns["amount"]
    elif "invoice" in obs_type:
        p = patterns["invoice_id"]
    elif "ticket" in obs_type:
        p = patterns["ticket_id"]
    elif "phone" in obs_type:
        p = patterns["phone_number"]
    else:
        return 0.5

    if re.fullmatch(p, evidence):
        return 1.0
    elif re.search(p, evidence):
        return 0.5
    else:
        return 0.0

def find_context(evidence_span, text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for s in sentences:
        if evidence_span in s:
            return s
    return text

def semantic_clarity(model, evidence_span, text):
    context = find_context(evidence_span, text)

    prompt = semantic_prompt.format(value=evidence_span, context=context)
    response = model.invoke(prompt)

    try:
        return float(response.content.strip())
    except (ValueError, AttributeError):
        return 0.5

def source_trust(source_type):
    return src_trust.get(source_type, 0.5)

def compute_confidence(structural, semantic, trust):
    confidence = (
        0.4 * structural +
        0.3 * semantic +
        0.3 * trust
    )
    return round(confidence, 3)

def score_obs(model, doc, obs):
    structural = struct_clarity(
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
            "struct_clarity": structural,
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
            score = score_obs(model, doc, obs)
            print( obs.type, "|", obs.extract_value, "|", score)