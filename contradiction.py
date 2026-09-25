import re
from extraction import get_model, run_all
from docs import DOCS

name_pattern = r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"

def get_entity_name(text):
    match = re.search(name_pattern, text)
    if match:
        return match.group()
    return "unknown"

def create_records(results):
    records = []
    for doc in DOCS:
        doc_id = doc["id"]
        entity = get_entity_name(doc["text"])

        for observation in results.get(doc_id, []):
            record = {
                "id": f"{doc_id}_{len(records)}",
                "source": doc_id,
                "entity": entity,
                "type": observation.type,
                "value": observation.extracted_value,
                "status": "ok",
                "conflicts_with": []
            }
            records.append(record)
    return records

def find_contradictions(records):
    groups = {}
    for record in records:
        key = (record["entity"], record["type"])
        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    for group in groups.values():
        values = []

        for record in group:
            if record["value"] not in values:
                values.append(record["value"])

        if len(values) > 1:
            for record in group:
                record["status"] = "contradictory"
                for other in group:
                    if other != record:
                        record["conflicts_with"].append(other["id"])
    return records

def lookup(records, entity, type):
    results = []

    for record in records:
        if record["entity"] == entity and record["type"] == type:
            results.append(record)
    return results

if __name__ == "__main__":
    model = get_model()
    results = run_all(model)
    records = create_records(results)
    records = find_contradictions(records)

    results = lookup(records, "Raj Malhotra", "phone_number")
    print("Raj Malhotra phone numbers:")
    for record in results:
        print(
            record["value"],
            record["source"],
            record["status"],
            record["conflicts_with"]
        )