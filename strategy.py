import re
from dotenv import load_dotenv
from extraction import get_model, extract_obs
from docs import DOCS

load_dotenv()

def extract_regex(text, field_type):
    patterns = {
        "date": r"\d{4}-\d{2}-\d{2}",
        "amount": r"\$[\d,]+\.\d{2}",
        "id": r"(?:INV-\d+|#\d+)",
        "phone": r"\+\d[\d-]{6,}",
    }
    ptn = patterns.get(field_type)

    if not ptn:
        return None

    match = re.search(ptn, text)

    if match:
        return match.group()
    return None

def llm_extract(model, text, field_type):
    prompt = f"""Extract the {field_type} from the text below.Return only the value.If there is no {field_type}, return "none".Text:{text}"""
    response = model.invoke(prompt)
    value = response.content.strip()

    if value.lower() == "none":
        return None
    return value

def extract_semantic(model, text):
    document = {
        "id": "unknown",
        "source_type": "unknown",
        "text": text
    }
    return extract_obs(model, document)

def strategy():
    model = get_model()
    results = []

    for doc in DOCS:
        text = doc["text"]
        for field_type in ["date", "amount", "id", "phone"]:
            value = extract_regex(text, field_type)
            if value:
                results.append({
                    "doc_id": doc["id"],
                    "field": field_type,
                    "value": value,
                    "strategy_use": "deterministic"
                })

        for field_type in ["priority"]:
            value = llm_extract(model, text, field_type)
            if value:
                results.append({
                    "doc_id": doc["id"],
                    "field": field_type,
                    "value": value,
                    "strategy_use": "heuristic"
                })

        if doc["id"] in ["C", "D"]:
            observations = extract_semantic(model, text)
            for obs in observations:
                results.append({
                    "doc_id": doc["id"],
                    "field": obs.type,
                    "value": obs.extract_value,
                    "strategy_use": "semantic"
                })
    return results

def print_results(results):
    print("doc_id | field | value | strategy_used")
    for result in results:
        print(result["doc_id"], "|", result["field"], "|", result["value"], "|", result["strategy_use"])

if __name__ == "__main__":
    results = strategy()
    print_results(results)