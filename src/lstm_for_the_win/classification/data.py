"""Review loading, normalization, validation, and deterministic validation splitting."""
from __future__ import annotations
import csv, random, re, unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

VALID_SENTIMENTS={"positive","neutral","negative"}; VALID_TOPICS={"smartphone","television","refrigerator","washing_machine"}
VALID_LEVELS={"limited","informal","standard","advanced","technical"}; VALID_LENGTHS={"short","medium","long"}; VALID_TASKS={"sentiment","topic"}
SLANG={"ngl","tbh","idk","imo","kinda","lowkey","fr","lol","wtf","gonna"}

@dataclass(frozen=True)
class ReviewRecord:
    ID:int; text:str; sentiment:str; topic:str; linguistic_level:str; flagprofanity:int; input_timestamp:str
    hasemoji:int=0; hasspellingerror:int=0; hasslang:int=0; length_class:str="short"; mixed_sentiment:int=0
    goldtest:int=0; source:str="incoming"; training_generation:int|None=None

def _hasemoji(text:str)->bool: return any(ord(c)>=0x1F000 for c in text)
def _length(text:str)->str:
    n=len(text.split()); return "short" if n<14 else "medium" if n<30 else "long"
def _hasslang(text:str)->bool: return bool(set(re.findall(r"[a-z']+",text.lower())) & SLANG)

def clean_text(text:str)->str:
    text=unicodedata.normalize("NFKD",str(text)); text="".join(c for c in text if not unicodedata.combining(c)).lower()
    text=re.sub(r"https?://\S+|www\.\S+"," ",text); text=re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"," ",text); text=re.sub(r"[@#]\w+"," ",text)
    return re.sub(r"\s+"," ","".join(c if c.isalpha() or c.isspace() or c=="'" or ord(c)>=0x1F000 else " " for c in text)).strip()

def _bin(value:str|None,field:str,row:int,default:int=0)->int:
    if value in (None,""): return default
    if value not in {"0","1"}: raise ValueError(f"Row {row} {field} must be 0 or 1.")
    return int(value)

def _ts(value:str,row:int)->None:
    try: parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as error: raise ValueError(f"Row {row} contains an invalid input_timestamp.") from error
    if parsed.tzinfo is None: raise ValueError(f"Row {row} input_timestamp must include a timezone.")

def _meta(row:dict[str,str],raw:str,n:int)->dict[str,int|str]:
    return {"hasemoji":_bin(row.get("hasemoji"),"hasemoji",n,int(_hasemoji(raw))),"hasspellingerror":_bin(row.get("hasspellingerror"),"hasspellingerror",n),"hasslang":_bin(row.get("hasslang"),"hasslang",n,int(_hasslang(raw))),"length_class":row.get("length_class","").strip() or _length(raw),"mixed_sentiment":_bin(row.get("mixed_sentiment"),"mixed_sentiment",n)}

def _common(record:ReviewRecord,n:int)->None:
    if record.ID<1 or not record.text: raise ValueError(f"Row {n} contains an invalid ID or empty text.")
    if record.sentiment not in VALID_SENTIMENTS or record.topic not in VALID_TOPICS or record.linguistic_level not in VALID_LEVELS or record.length_class not in VALID_LENGTHS: raise ValueError(f"Row {n} contains invalid review metadata.")
    _ts(record.input_timestamp,n)

def _sequence(records:Sequence[ReviewRecord],name:str)->None:
    if not records: raise ValueError(f"{name} is empty.")
    ids=[r.ID for r in records]; texts=[r.text for r in records]
    if ids!=sorted(ids) or len(ids)!=len(set(ids)): raise ValueError(f"{name} IDs must be unique and monotonically increasing.")
    if len(texts)!=len(set(texts)): raise ValueError(f"{name} text must be unique.")

def load_train(path:str|Path)->list[ReviewRecord]:
    p=Path(path); required={"ID","text","sentiment","topic","linguistic_level","flagprofanity","source","training_generation","input_timestamp"}
    if not p.is_file(): raise FileNotFoundError(f"Training dataset not found: {p}")
    out=[]
    with p.open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames): raise ValueError("train.csv does not use the expected schema.")
        for n,row in enumerate(reader,start=2):
            raw=row["text"]; source=row["source"].strip()
            if source not in {"initial","goldtest"}: raise ValueError(f"Row {n} source must be initial or goldtest.")
            try: rid=int(row["ID"]); generation=int(row["training_generation"])
            except (TypeError,ValueError) as error: raise ValueError(f"Row {n} contains an invalid numeric field.") from error
            record=ReviewRecord(rid,clean_text(raw),row["sentiment"].strip(),row["topic"].strip(),row["linguistic_level"].strip(),_bin(row["flagprofanity"],"flagprofanity",n),row["input_timestamp"].strip(),source=source,training_generation=generation,**_meta(row,raw,n)); _common(record,n); out.append(record)
    _sequence(out,"train.csv"); return out

def load_incoming(path:str|Path)->list[ReviewRecord]:
    p=Path(path); required={"ID","text","expected_sentiment","expected_topic","linguistic_level","flagprofanity","goldtest","input_timestamp"}
    if not p.is_file(): raise FileNotFoundError(f"Incoming dataset not found: {p}")
    out=[]
    with p.open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames): raise ValueError("incoming.csv does not use the expected schema.")
        for n,row in enumerate(reader,start=2):
            raw=row["text"]
            try: rid=int(row["ID"])
            except (TypeError,ValueError) as error: raise ValueError(f"Row {n} contains an invalid ID.") from error
            record=ReviewRecord(rid,clean_text(raw),row["expected_sentiment"].strip(),row["expected_topic"].strip(),row["linguistic_level"].strip(),_bin(row["flagprofanity"],"flagprofanity",n),row["input_timestamp"].strip(),goldtest=_bin(row["goldtest"],"goldtest",n),**_meta(row,raw,n)); _common(record,n); out.append(record)
    _sequence(out,"incoming.csv"); return out

def label_for(record:ReviewRecord,task:str)->str:
    if task not in VALID_TASKS: raise ValueError("task must be sentiment or topic.")
    return record.sentiment if task=="sentiment" else record.topic

def stratified_validation_split(records:Iterable[ReviewRecord],task:str,validation_fraction:float,seed:int)->tuple[list[ReviewRecord],list[ReviewRecord]]:
    if not 0.0<validation_fraction<1.0: raise ValueError("validation_fraction must be strictly between 0 and 1.")
    groups=defaultdict(list)
    for record in records: groups[label_for(record,task)].append(record)
    if len(groups)<2: raise ValueError("At least two labels are required.")
    fit=[]; validation=[]
    for i,label in enumerate(sorted(groups)):
        group=list(groups[label])
        if len(group)<3: raise ValueError(f"Label {label} requires at least three training examples.")
        random.Random(seed+i*10_007).shuffle(group); size=min(max(1,int(round(len(group)*validation_fraction))),len(group)-2)
        validation.extend(group[:size]); fit.extend(group[size:])
    fit.sort(key=lambda r:r.ID); validation.sort(key=lambda r:r.ID); return fit,validation
