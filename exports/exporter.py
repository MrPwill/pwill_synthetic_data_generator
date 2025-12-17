import pandas as pd
import json
from typing import List, Union
from core.schemas import GenerationResult, GeneratedSample

def export_to_csv(result: GenerationResult) -> str:
    data = []
    for s in result.samples:
        if isinstance(s.content, dict):
             data.append(s.content)
        elif isinstance(s.content, str):
             data.append({"content": s.content})
        else:
             data.append({"content": str(s.content)})
             
    df = pd.DataFrame(data)
    filename = f"export_{result.request_id}.csv"
    df.to_csv(filename, index=False)
    return filename

def export_to_json(result: GenerationResult) -> str:
    data = [s.dict() for s in result.samples]
    filename = f"export_{result.request_id}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return filename

def export_to_jsonl(result: GenerationResult) -> str:
    filename = f"export_{result.request_id}.jsonl"
    with open(filename, "w") as f:
        for s in result.samples:
            json.dump(s.dict(), f, default=str)
            f.write("\n")
    return filename
