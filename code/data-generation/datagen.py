"""
datagen_llm.py

Async SOW generator that mixes template-based clauses with LLM-generated paragraphs
using your custom LLM router (SimpleLLMCaller).

Usage:
    python datagen_llm.py

Requirements:
    pip install aiohttp faker
    Place SimpleLLMCaller class in same file or import it.
"""

import os
import csv
import random
import asyncio
import json
import datetime
from faker import Faker
from typing import Dict, Any
from code.helpers.llm import SimpleLLMCaller
from tqdm.asyncio import tqdm

# === CONFIGURABLE PARAMETERS ===
OUT_DIR = "./generated_data"
os.makedirs(OUT_DIR, exist_ok=True)

VENDOR_NAME = "Sprinklr Pvt. Ltd."
NUM_DOCS = 1200               # total docs to generate
USE_LLM_PROB = 0.8             # probability a clause will use LLM (0.0 - 1.0)
MODEL_NAME = "gpt-4.1-mini"    # model name to pass to router
MAX_CONCURRENT_CALLS = 16      # concurrency of LLM calls
LLM_RETRY = 3                 # how many retries SimpleLLMCaller will perform (passed to its constructor)
CACHE_PATH = "./llm_cache.json"  # simple local cache to avoid re-calls

# you must have SimpleLLMCaller available in this module (paste earlier class or import it)
# from simple_llm import SimpleLLMCaller

fake = Faker()

# === Clause pools (use your expanded version) ===
CLAUSE_POOL = {
    # (shortened here - use the full pool you already prepared)
    "Scope of Work": [
        "development of web applications",
        "data analytics dashboards",
        "mobile app design and delivery",
        "AI-based chatbot implementation",
        "maintenance and bug fixes",
        "cloud infrastructure setup",
        "data pipeline development",
        "backend microservice architecture",
        "API integration and documentation",
        "DevOps automation and CI/CD setup",
        "UI/UX design and usability testing",
        "custom plugin and extension development",
        "data migration and ETL operations",
        "third-party API evaluation and integration",
        "performance optimization and scalability improvements"
    ],
    "Deliverables": [
        "final project documentation",
        "source code and build scripts",
        "trained AI models and data artifacts",
        "UAT deployment and demo video",
        "compliance audit reports",
        "user manuals and admin guides",
        "deployment scripts and environment configs",
        "API reference documentation",
        "test cases and QA reports",
        "project status and progress reports",
        "design mockups and prototypes",
        "post-deployment performance metrics"
    ],
    # ... include the rest of your clause pool here ...
}

OUTLIER_CLAUSES = [
    "Client retains full intellectual property rights even before payment.",
    "Vendor may subcontract without prior written approval.",
    "Payments are deferred until completion of all deliverables.",
    "Either party may terminate without cause upon 3 days notice.",
    "A confidentiality breach does not constitute material breach under this agreement.",
    "Vendor is entitled to use client data for internal AI training purposes.",
    "All disputes to be resolved solely in vendor’s jurisdiction.",
    "Client agrees to bear all costs arising out of vendor’s tax obligations.",
    "Payment delays beyond 90 days will not attract penalties.",
    "Vendor’s liability shall not exceed INR 10,000 regardless of contract value."
]

FLUFF_TEMPLATES = [
    "It is hereby agreed that {}.",
    "Parties mutually consent to the inclusion of {}.",
    "This clause covers {} in detail.",
    "In essence, {} shall be undertaken as part of the project scope.",
    "Under this section, the Vendor commits to {}.",
    "For the avoidance of doubt, this includes {}."
]

NOISE_LINES = [
    "### END OF DOCUMENT ###",
    "Scanned Copy - Subject to Verification",
    "Attachment: Appendix-A (missing)",
    "Ref ID: {}",
    "Digitally Signed - Validity Pending",
    "For Internal Review Only",
    "Confidential Draft - Not for Distribution",
    "Page {} of {}",
    "Timestamp: {}",
    "Verification Code: {}"
]

TONES = ["formal", "technical", "legal", "conversational", "neutral", "corporate", "contractual", "explanatory"]

# === Utilities (template-based helpers) ===
def add_linguistic_noise(text: str) -> str:
    if random.random() < 0.15:
        text = text.upper()
    if random.random() < 0.10:
        text = text.replace(".", "")
    if random.random() < 0.10:
        text = " ".join(text.split())
    return text

def gen_clause_text_template(subtheme: str, tone: str) -> str:
    base = random.choice(FLUFF_TEMPLATES).format(subtheme)
    if tone == "technical":
        base += " Technical details will be documented later."
    elif tone == "legal":
        base = "WHEREAS, " + base
    elif tone == "conversational":
        base = "Basically, " + base.lower()
    return add_linguistic_noise(base)

def compose_section_template(clause: str, tone: str) -> str:
    num_subthemes = random.randint(1, 3)
    subthemes = random.sample(CLAUSE_POOL[clause], k=min(num_subthemes, len(CLAUSE_POOL[clause])))
    combined = " and ".join(subthemes)
    section_parts = []
    for _ in range(random.randint(1, 3)):
        part = gen_clause_text_template(combined, tone)
        if random.random() < 0.20:
            part += " " + random.choice(OUTLIER_CLAUSES)
        if random.random() < 0.15:
            part += f" Effective date: {fake.date_between('-3y', 'today')}."
        section_parts.append(part)
    # emphatically include heading occasionally
    if random.random() < 0.6:
        return f"{clause}\n" + "\n".join(section_parts)
    else:
        # inline header
        return clause + " includes " + " ".join(section_parts)

# === LLM + caching helpers ===
def load_cache(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}

def save_cache(path: str, cache: Dict[str, str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def build_prompt(clause: str, subthemes: str, tone: str, client: str) -> str:
    """
    Construct a concise, deterministic prompt for the LLM.
    Keep it deterministic so caching works well.
    """
    prompt = (
        f"You are an expert contract writer. Produce 2-4 concise sentences suitable for a "
        f"Statement of Work (SOW) under the heading '{clause}', covering: {subthemes}. "
        f"Write in a {tone} tone, suitable for a commercial SOW for client '{client}'. "
        "Avoid excessive legalese; include any relevant numbers or dates if appropriate. "
        "Style rules:\n"
        "- Avoid starting every sentence with 'The Contractor', 'The Vendor', or 'The Client'.\n"
        "-You may use transitions such as “Additionally,”, “To ensure,”, “During this phase,”, etc., but only occasionally.\n"
        "- Keep it professional and contract-appropriate.\n"
        "- Do not repeat boilerplate.\n"
        "- Return plain text only, no lists or bullet points."
    )
    return prompt

# === Main async generation logic ===
async def generate_documents(use_llm_prob: float = USE_LLM_PROB,
                             num_docs: int = NUM_DOCS,
                             model_name: str = MODEL_NAME,
                             max_concurrency: int = MAX_CONCURRENT_CALLS):
    # load cache
    cache = load_cache(CACHE_PATH)

    # create LLM caller (ensure SimpleLLMCaller is defined/imported)
    llm_caller = SimpleLLMCaller(max_retries=LLM_RETRY, semaphore=max_concurrency)

    # concurrency semaphore for LLM requests at the generator level
    llm_semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_single_doc(i: int):
        client_name = fake.company()
        tone = random.choice(TONES)
        n_clauses = random.randint(6, 12)
        chosen = random.choices(list(CLAUSE_POOL.keys()), k=n_clauses)
        lines = [f"Statement of Work (SOW)\nBetween {client_name} and {VENDOR_NAME}\n\n"]
        lines.append(f"Date of Agreement: {fake.date_between('-3y', 'today')}\nEffective Until: {fake.date_between('today', '+2y')}\n\n")

        doc_record = {
            "file": f"sow_{i:04d}.txt",
            "client": client_name,
            "tone": tone,
            "doc_id": f"SOW-{i:04d}"
        }

        # For each clause, decide whether to use LLM or template
        llm_tasks = []
        clause_results = []

        for clause in chosen:
            # select subthemes to pass into the prompt (1-3)
            num_subthemes = random.randint(1, 3)
            subthemes = random.sample(CLAUSE_POOL[clause], k=min(num_subthemes, len(CLAUSE_POOL[clause])))
            subthemes_str = " and ".join(subthemes)

            use_llm = random.random() < use_llm_prob

            if use_llm:
                prompt = build_prompt(clause, subthemes_str, tone, client_name)

                # include generation params (match sample request shape)
                generation_params = {
                    "temperature": 1.0,
                    "client_identifier": "backend-core-devs",
                    "provider": "OPEN_AI",
                    "max_tokens": 512
                }

                # include generation params in cache key so different params don't collide
                cache_key = f"{clause}|||{subthemes_str}|||{tone}|||{client_name}|||{json.dumps(generation_params, sort_keys=True)}"
                if cache_key in cache:
                    # cached result
                    clause_results.append((clause, cache[cache_key], True))
                else:
                    # create an async task to call LLM (pass generation_params through)
                    async def call_and_store(key, pr, gparams):
                        async with llm_semaphore:
                            try:
                                text = await llm_caller.call_and_get_text(pr, model_name, generation_params=gparams)
                                if text and isinstance(text, str) and text.strip():
                                    cache[key] = text.strip()
                                    return (clause, text.strip(), True)
                                else:
                                    # fallback to template if empty
                                    return (clause, compose_section_template(clause, tone), False)
                            except Exception:
                                # fallback to template on errors
                                return (clause, compose_section_template(clause, tone), False)
                    llm_tasks.append((cache_key, prompt, generation_params, call_and_store))
            else:
                # immediate template
                clause_results.append((clause, compose_section_template(clause, tone), False))

        # run all LLM tasks in parallel (bounded by llm_semaphore)
        if llm_tasks:
            # build coroutine list (pass the generation_params into each call)
            coros = [task_info[3](task_info[0], task_info[1], task_info[2]) for task_info in llm_tasks]
            completed = await asyncio.gather(*coros)
            clause_results.extend(completed)

        # now assemble sections and write file
        # shuffle order a bit to increase variance
        random.shuffle(clause_results)
        for clause_name, text, used_llm in clause_results:
            # sometimes repeat a clause to simulate reappearing sections
            repeat = 1
            for _ in range(repeat):
                lines.append(text + "\n\n")
            doc_record[clause_name.replace(" ", "_").lower()] = 1 if text else 0
            doc_record[f"{clause_name.replace(' ', '_').lower()}_llm"] = int(used_llm)

        # occasional noise
        if random.random() < 0.2:
            noise = random.choice(NOISE_LINES)
            placeholders = noise.count("{}")
            if placeholders == 0:
                lines.append(noise + "\n")
            elif placeholders == 1:
                lines.append(noise.format(fake.uuid4()) + "\n")
            elif placeholders == 2:
                lines.append(noise.format(random.randint(1, 20), random.randint(20, 99)) + "\n")
            else:
                lines.append(noise.format(datetime.datetime.now().isoformat()) + "\n")

        rep = fake.name()
        address = fake.address().replace("\n", ", ")
        lines.append(f"Signed by: {rep}, {client_name}\nAddress: {address}\n")

        path = f"{OUT_DIR}/sow_{i:04d}.txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        return doc_record

    # generate docs in batches
    results = []
    # use a pool of tasks but limit overall concurrency to avoid too many concurrent file/LLM ops
    sem = asyncio.Semaphore(max_concurrency * 2)

    async def worker(idx):
        async with sem:
            return await generate_single_doc(idx)

    tasks = [worker(i) for i in range(500,NUM_DOCS)]
    with tqdm(total=len(tasks), desc="Generating SOWs", ncols=100) as pbar:
        for chunk_start in range(0, len(tasks), max_concurrency):
            chunk = tasks[chunk_start:chunk_start + max_concurrency]
            completed_chunk = await asyncio.gather(*chunk)
            results.extend(completed_chunk)
            # persist cache periodically
            save_cache(CACHE_PATH, cache)
            pbar.update(len(completed_chunk))

    # write manifest
    manifest_path = f"{OUT_DIR}/manifest_llm.csv"
    keys = sorted(set().union(*[r.keys() for r in results]))
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    # final cache save
    save_cache(CACHE_PATH, cache)
    print(f"Generated {len(results)} documents into {OUT_DIR}/  (manifest: {manifest_path})")


# === Entrypoint ===
if __name__ == "__main__":
    # run the async generator
    asyncio.run(generate_documents(USE_LLM_PROB, NUM_DOCS, MODEL_NAME, MAX_CONCURRENT_CALLS))
