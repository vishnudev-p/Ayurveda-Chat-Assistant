# ==========================================
# Run this cell in Jupyter Notebook
# RAG Backend with Ngrok Tunnel
# ==========================================

import nest_asyncio
import threading
import time
from pyngrok import ngrok
import uvicorn

# Apply nest_asyncio to allow uvicorn to run in Jupyter
nest_asyncio.apply()

# ==========================================
# CONFIGURATION
# ==========================================
NGROK_AUTH_TOKEN = ""  # Replace with your ngrok token
PORT = 3038

# ==========================================
# Setup Ngrok
# ==========================================
print("🔧 Setting up Ngrok...")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Kill any existing ngrok tunnels
try:
    ngrok.kill()
except:
    pass

# ==========================================
# Import your RAG backend
# ==========================================
print("📦 Loading RAG backend modules...")

import os, re, time, torch, subprocess, sys, pkg_resources
from typing import List, Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from pymongo import MongoClient
from datetime import datetime
import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment / Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
installed = {pkg.key for pkg in pkg_resources.working_set}
if "bson" in installed:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "bson", "-y"], check=False)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Config
MONGO_URI = ""

# Source Data Configuration
SOURCE_DB = ""
CONTENT_COLLECTION = ""
DICTIONARY_COLLECTION = ""

# Vector Storage Configuration
VECTOR_DB = ""
VECTOR_COLLECTION = ""

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

PT_MAX_AGE_SECS = 3 * 24 * 60 * 60
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CHUNKS = 3
MAX_CHARS_PER_CHUNK = 2500
MAX_CONTEXT_CHARS = 8000

MIN_COSINE_SIM = 0.30
MIN_CROSS_ENCODER = 0.25

# Load Models
logger.info("Loading Embedding & Reranker models...")
embedder = SentenceTransformer(EMBED_MODEL, device=device)
reranker = CrossEncoder(RERANK_MODEL, device=device)
logger.info(f"Models ready on {device.upper()}")

logger.info("Loading generation model from Hugging Face...")
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = ""

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, token=os.environ["HF_TOKEN"])
if gen_tokenizer.pad_token is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID,
    token=os.environ["HF_TOKEN"],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=gen_tokenizer
)
logger.info("Generation model ready.")

# Helpers
ASCII_WORDS = re.compile(r"[A-Za-z]")
TAGS_RE = re.compile(r"<[^>]+>")

def html_to_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = TAGS_RE.sub(" ", s)
    return (s.replace("&nbsp;", " ").replace("&amp;", "&")
             .replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&#39;", "'"))

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html_to_text(text)
    lines = [ln.strip() for ln in text.splitlines() if ASCII_WORDS.search(ln) or re.search(r'[\u0900-\u097F]', ln)]
    text = " ".join(lines)
    # Modified regex to keep Devanagari characters along with English
    text = re.sub(r"[^\w\s.,;:()\-/%\u0900-\u097F।॥]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def prepare_bm25(chunks: List[str]):
    tokenized = [[w.lower() for w in re.findall(r"[a-z0-9]+", c.lower())] for c in chunks]
    return tokenized, BM25Okapi(tokenized)

def safe_float(x) -> float:
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return 0.0

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def read_file(path):
    try:
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if path.lower().endswith(".pdf"):
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            return "\n".join(pages)
        if path.lower().endswith(".docx"):
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        if path.lower().endswith((".json", ".csv")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
    return ""

# ==========================================
# Vector Database Utilities  (REPLACED / UPGRADED)
# ==========================================

def _content_doc_from_src(doc: Dict) -> Dict:
    """Extract a single cleaned textbook chunk doc with exact required fields."""
    textbook = doc.get("textBook name") or doc.get("textbook_name") or doc.get("textbook") or "Unknown"
    topic = doc.get("topic name") or doc.get("topic_name") or doc.get("topic") or "Unknown"
    subject = doc.get("subject name") or doc.get("subject_name") or doc.get("subject") or "Unknown"
    content = doc.get("Content description") or doc.get("content_description") or doc.get("content") or ""

    textbook = clean_text(textbook)
    topic = clean_text(topic)
    subject = clean_text(subject)
    content_clean = clean_text(content)

    # skip if meaningless
    if not content_clean or len(content_clean.split()) < 5:
        return None

    return {
        "textBook name": textbook,
        "topic name": topic,
        "subject name": subject,
        "Content description": content_clean,
        "source_id": str(doc.get("_id", "")),
    }

def _dict_doc_from_src(doc: Dict) -> Dict:
    """Extract a single cleaned dictionary chunk doc with exact required fields."""
    word = (doc.get("word") or "").strip()
    desc = (doc.get("word_description") or "").strip()

    word = clean_text(word)
    desc = clean_text(desc)

    if not word or not desc or len(desc.split()) < 3:
        return None

    return {
        "word": word,
        "word_description": desc,
        "source_id": str(doc.get("_id", "")),
    }

def fetch_text_from_mongodb() -> List[Dict]:
    """
    Fetch and transform textbook documents from SOURCE_DB into the EXACT structure:
    { "textBook name", "topic name", "subject name", "Content description" }
    """
    content_docs = []
    try:
        logger.info(f"Connecting to MongoDB for textbook data from {SOURCE_DB}...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        
        db = client[SOURCE_DB]

        all_collections = db.list_collection_names()

        if not all_collections:
            logger.warning("No collections found in source database.")
            return content_docs

        if CONTENT_COLLECTION not in all_collections and "content" not in all_collections:
            logger.warning(f"Collection '{CONTENT_COLLECTION}' not found in {SOURCE_DB}. Available: {all_collections}")

        for coll_name in all_collections:
            if coll_name.lower() not in [CONTENT_COLLECTION.lower(), "content", "content1"]:
                continue

            coll = db[coll_name]
            count = coll.count_documents({})
            logger.info(f"Processing collection '{coll_name}' with {count} documents")

            for doc in coll.find({}):
                out = _content_doc_from_src(doc)
                if out:
                    content_docs.append(out)

        client.close()
        logger.info(f"✅ Fetched and cleaned {len(content_docs)} textbook docs")        
    except Exception as e:
        logger.error(f"MongoDB fetch error (text content): {e}")
        import traceback
        logger.error(traceback.format_exc())

    return content_docs

    
    return dictionary_texts

def fetch_dictionary_from_mongodb() -> List[Dict]:
    """
    Fetch and transform dictionary documents from SOURCE_DB into the EXACT structure:
    { "word", "word_description" }
    """
    dict_docs = []

    try:
        logger.info(f"Connecting to MongoDB for dictionary data from {SOURCE_DB}...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[SOURCE_DB]


        all_collections = db.list_collection_names()
        if DICTIONARY_COLLECTION not in all_collections:
            logger.warning(f"Dictionary collection '{DICTIONARY_COLLECTION}' not found. Available: {all_collections}")
            return dict_docs

        coll = db[DICTIONARY_COLLECTION]
        count = coll.count_documents({})
        logger.info(f"Processing dictionary collection '{DICTIONARY_COLLECTION}' with {count} entries")

        for doc in coll.find({}):
            out = _dict_doc_from_src(doc)
            if out:
                dict_docs.append(out)

        client.close()
        logger.info(f"✅ Fetched and cleaned {len(dict_docs)} dictionary docs")


    except Exception as e:
        logger.error(f"Dictionary fetch error: {e}")
        import traceback
    return dict_docs


def embed_in_batches(texts: List[str], batch_size=64) -> torch.Tensor:
    all_embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embs = embedder.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embs.append(embs.cpu())
        if (i // batch_size) % 10 == 0:
            logger.info(f"Embedded {i + len(batch)} / {total}")

    return torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, embedder.get_sentence_embedding_dimension()))


def save_to_vector_db(docs_with_embeddings: List[Dict], metadata: dict) -> int:
    """
    Save BOTH textbook and dictionary docs to MongoDB vector collection in VECTOR_DB,
    each doc already has 'embedding' and exact fields (no markers).
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[VECTOR_DB]
        coll = db[VECTOR_COLLECTION]

        coll.delete_many({})
        if docs_with_embeddings:
            coll.insert_many(docs_with_embeddings)
            logger.info(f"✅ Saved {len(docs_with_embeddings)} vector docs to {VECTOR_DB}.{VECTOR_COLLECTION}")

        meta_coll = db["vector_metadata"]
        meta_coll.delete_many({})
        meta_coll.insert_one(metadata)

        client.close()
        return len(docs_with_embeddings)

    except Exception as e:
        logger.error(f"⚠️ Error saving to vector DB: {e}")
        return 0


def load_from_vector_db():
    """
    Load vectors from Mongo and prepare in-memory search structures.
    We ALSO build in-memory 'chunk' strings with markers so /debug/chunks works same as before.
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[VECTOR_DB]
        coll = db[VECTOR_COLLECTION]
        meta_coll = db["vector_metadata"]

        docs = list(coll.find({}))
        meta_doc = meta_coll.find_one({}) or {}

        if not docs:
            client.close()
            return None

        chunks: List[str] = []
        embeddings: List[List[float]] = []
        meta_list: List[Dict] = []

        content_count = 0
        dict_count = 0
        upload_count = meta_doc.get("num_uploads", 0) or 0  # preserved from metadata if any

        for d in docs:
            emb = d.get("embedding")
            if not emb:
                continue

            # textbook doc
            if (
                "textBook name" in d
                and "topic name" in d
                and "subject name" in d
                and "Content description" in d
            ):
                content_count += 1
                text = (
                    "[TEXTBOOK CONTENT]\n"
                    f"Book: {d.get('textBook name', 'Unknown')}\n"
                    f"Topic: {d.get('topic name', 'Unknown')}\n"
                    f"Subject: {d.get('subject name', 'Unknown')}\n\n"
                    f"{d.get('Content description', '')}"
                )
                chunks.append(text)
                embeddings.append(emb)
                meta_list.append({
                    "type": "content",
                    "textBook name": d.get("textBook name", ""),
                    "topic name": d.get("topic name", ""),
                    "subject name": d.get("subject name", ""),
                    "Content description": d.get("Content description", ""),
                    "source_id": d.get("source_id", "")
                })

            # dictionary doc
            elif "word" in d and "word_description" in d:
                dict_count += 1
                text = (
                    "[DICTIONARY TERM]\n"
                    f"Word: {d.get('word', '')}\n\n"
                    f"Definition: {d.get('word_description', '')}"
                )
                chunks.append(text)
                embeddings.append(emb)
                meta_list.append({
                    "type": "dictionary",
                    "word": d.get("word", ""),
                    "word_description": d.get("word_description", ""),
                    "source_id": d.get("source_id", "")
                })

        client.close()

        logger.info(f"✅ Loaded {len(chunks)} chunks from vector DB ({VECTOR_DB}.{VECTOR_COLLECTION})")

        emb_tensor = (
            torch.tensor(embeddings, dtype=torch.float32)
            if embeddings
            else torch.empty((0, embedder.get_sentence_embedding_dimension()))
        )
        tokenized, bm25 = prepare_bm25(chunks)

        # merge/override metadata with live counts
        meta_doc.update({
            "num_content": int(content_count),
            "num_dictionary": int(dict_count),
            "num_uploads": int(upload_count),
        })

        return {
            "chunks": chunks,
            "embeddings": emb_tensor,
            "bm25_tokens": tokenized,
            "_bm25": bm25,
            "meta": meta_doc
        }

    except Exception as e:
        logger.error(f"⚠️ Error loading from vector DB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def build_vector_db() -> int:
    """
    Unified builder:

    - Pulls textbook docs and dictionary docs from SOURCE_DB
    - Optionally includes uploads (your prior behavior preserved)
    - Embeds ALL using your chosen embedder
    - Saves to VECTOR_DB with exact required document shapes + 'embedding'
    """

    logger.info("Rebuilding vector database...")
    
    # 1) Source DB docs
    content_docs = fetch_text_from_mongodb()  # list of dicts
    dict_docs = fetch_dictionary_from_mongodb()  # list of dicts

    logger.info(f"MongoDB content docs: {len(content_docs)}")
    logger.info(f"Dictionary docs: {len(dict_docs)}")
    
     # 2) Uploads (kept for parity, but uploads are NOT stored as vector docs with fields — they influence context only if desired)
    upload_texts = []
    for fn in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, fn)
        if not os.path.isfile(path):
            continue
        if not fn.lower().endswith((".txt", ".pdf", ".docx", ".json", ".csv")):
            continue
        raw = read_file(path)
        cleaned = clean_text(raw)
        if cleaned:
            upload_texts.append(f"[UPLOAD:{fn}] {cleaned}")
            logger.info(f"Cached from uploads: {fn}")

    # 3) Prepare texts for embedding

    # For content docs, create the string used for embedding (but storage stays fielded JSON)
    content_texts = [
        f"{d['textBook name']} - {d['topic name']} - {d['subject name']}: {d['Content description']}"
        for d in content_docs
    ]

    dict_texts = [
        f"{d['word']}: {d['word_description']}"
        for d in dict_docs
    ]

    total_texts = content_texts + dict_texts

    logger.info(f"Total vectorizable texts: {len(total_texts)}")
    logger.info(f"  - Content: {len(content_texts)}")
    logger.info(f"  - Dictionary: {len(dict_texts)}")
    logger.info(f"  - Uploads (not embedded into Mongo vectors): {len(upload_texts)}")

    if not total_texts:

            logger.warning("No text to build vector database!")
            return 0

    # 4) Embed all vectors
    logger.info(f"Embedding {len(total_texts)} items with {EMBED_MODEL} ...")
    all_embs = embed_in_batches(total_texts)
    if all_embs.shape[0] != len(total_texts):
        logger.error("Embedding count mismatch!")
        return 0
    # 5) Attach embeddings to docs (EXACT field structure) and save
    docs_with_embeddings: List[Dict] = []

    # content docs first
    for i, d in enumerate(content_docs):
        emb = all_embs[i].tolist()
        doc = {
            "textBook name": d["textBook name"],
            "topic name": d["topic name"],
            "subject name": d["subject name"],
            "Content description": d["Content description"],
            "embedding": emb,
            "source_id": d.get("source_id", ""),
            "created_at": datetime.utcnow()
        }
        docs_with_embeddings.append(doc)

    # dictionary docs next
    offset = len(content_docs)
    for j, d in enumerate(dict_docs):
        emb = all_embs[offset + j].tolist()
        doc = {
            "word": d["word"],
            "word_description": d["word_description"],
            "embedding": emb,
            "source_id": d.get("source_id", ""),
            "created_at": datetime.utcnow()
        }
        docs_with_embeddings.append(doc)

    metadata = {
        "built_at": time.time(),
        "files": os.listdir(UPLOAD_DIR),
        "embed_model": EMBED_MODEL,
        "embedding_dim": int(all_embs.shape[1]) if all_embs.numel() else embedder.get_sentence_embedding_dimension(),
        "source_type": "mongodb(content1,dictionary)+uploads(separate)",
        "num_chunks": len(docs_with_embeddings),
        "num_content": len(content_docs),
        "num_dictionary": len(dict_docs),
        "num_uploads": len(upload_texts)
    }

    count = save_to_vector_db(docs_with_embeddings, metadata)
    logger.info(f"[OK] Saved vector store with {count} docs to MongoDB ({VECTOR_DB}.{VECTOR_COLLECTION})")
    return count

def load_db():
    db = load_from_vector_db()
    if db is None:
        logger.info("Vector store not found in MongoDB, building...")
        n = build_vector_db()
        if n == 0:
            logger.error("Failed to build vector store - no data found")
            return None
        db = load_from_vector_db()

    if db:
        logger.info(f"Loaded vector store with {len(db['chunks'])} chunks")
    return db

def ensure_fresh_db(force_if_mismatch: bool = True):
    global DB
    need = False
    if DB is None:
        logger.warning("DB is None, needs rebuild")
        need = True
    else:
        meta = DB.get("meta", {})
        if (time.time() - meta.get("built_at", 0)) > PT_MAX_AGE_SECS:
            logger.info("DB too old, needs rebuild")
            need = True
        if force_if_mismatch and meta.get("embed_model") != EMBED_MODEL:
            logger.info("Detected embed model change, needs rebuild")
            need = True
    
    if need:
        n = build_vector_db()
        DB = load_from_vector_db() if n > 0 else None
        

DB = load_db()

def extract_sanskrit_text(text: str) -> str:
    """Extract Sanskrit/Devanagari script from text"""
    if not text:
        return ""
    
    # First, try to find continuous blocks of Devanagari text
    # Pattern: sequences of Devanagari chars with spaces, punctuation
    devanagari_blocks = re.findall(r'[\u0900-\u097F।॥\s]+', text)
    
    sanskrit_parts = []
    for block in devanagari_blocks:
        # Clean and validate the block
        block = block.strip()
        # Must have at least 10 Devanagari characters to be considered a verse
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', block))
        if devanagari_count >= 10:
            sanskrit_parts.append(block)
    
    if sanskrit_parts:
        # Join all parts and format properly
        sanskrit = ' '.join(sanskrit_parts)
        # Try to detect verse breaks (।) and add line breaks
        sanskrit = re.sub(r'।\s*', '।\n', sanskrit)
        # Clean up excessive whitespace
        sanskrit = re.sub(r' +', ' ', sanskrit)
        sanskrit = re.sub(r'\n\s*\n+', '\n', sanskrit)
        return sanskrit.strip()
    
    return ""

def generate_text_local(prompt, max_new_tokens=2500, temperature=0.3):
    """
    Generate text with proper completion handling.
    Uses iterative generation if response is incomplete.
    """
    try:
        system_prompt = (
            "You are a scholarly Ayurvedic assistant specializing in classical texts like Charaka Saṃhitā. "
            "Provide detailed, well-structured responses with proper explanations and summaries. "
            "CRITICAL: You MUST complete your response with a full '## Summary' section. Never stop mid-sentence.\n\n"
        )
        input_text = system_prompt + prompt
        
        # First generation attempt
        outputs = generator(
            input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=gen_tokenizer.eos_token_id,
            eos_token_id=gen_tokenizer.eos_token_id,
            return_full_text=False
        )
        
        generated = (outputs[0].get("generated_text") or "").strip()
        finish_reason = outputs[0].get('finish_reason', 'unknown')
        
        logger.info(f"Initial generation: {len(generated)} chars, finish_reason: {finish_reason}")
        
        # Check if Summary section exists and response seems complete
        has_summary = "## Summary" in generated
        seems_incomplete = (
            finish_reason == "length" or 
            not has_summary or
            (has_summary and len(generated.split("## Summary")[1].strip()) < 100)
        )
        
        # If incomplete, try to continue generation
        if seems_incomplete:
            logger.warning("Response incomplete - attempting continuation...")
            
            # Create continuation prompt
            continuation_prompt = f"""You are continuing an Ayurvedic response. Here's what was written so far:

{generated}

{"CRITICAL: The response above is missing a proper Summary section." if not has_summary else "CRITICAL: The Summary section above is incomplete."}

Your task: {"Write a complete '## Summary' section" if not has_summary else "Complete the Summary section"} that:
- Synthesizes the main points in 5-7 well-developed sentences
- Emphasizes the practical and conceptual significance
- Relates back to the broader Ayurvedic framework
- Provides closure to the entire discussion

{"## Summary" if not has_summary else ""}

Write the {"complete Summary section" if not has_summary else "continuation"} now:"""

            continuation_outputs = generator(
                continuation_prompt,
                max_new_tokens=800,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=gen_tokenizer.eos_token_id,
                eos_token_id=gen_tokenizer.eos_token_id,
                return_full_text=False
            )
            
            continuation = (continuation_outputs[0].get("generated_text") or "").strip()
            
            if continuation:
                logger.info(f"Generated continuation: {len(continuation)} chars")
                
                if not has_summary:
                    # Add the complete Summary section
                    if not continuation.startswith("## Summary"):
                        continuation = "## Summary\n\n" + continuation
                    generated = generated + "\n\n" + continuation
                else:
                    # Complete the existing Summary section
                    # Split at Summary, keep everything before, append new content
                    parts = generated.split("## Summary")
                    if len(parts) == 2:
                        before_summary = parts[0]
                        incomplete_summary = parts[1].strip()
                        # Merge incomplete with continuation
                        generated = before_summary + "## Summary\n\n" + incomplete_summary + " " + continuation
                
                logger.info("✓ Response completed with full Summary section")
            else:
                logger.warning("Continuation generation failed")
        
        logger.info(f"Final generated text length: {len(generated)} chars")
        return generated
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""

def get_ood_response():
    return {
        "response": "I don't have information about that in the provided texts.",
        "debug": {
            "status": "out_of_domain",
            "reason": "Question not related to knowledge base content"
        }
    }

# FastAPI App
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    try:
        ensure_fresh_db()
        body = await request.json()
        question = (body.get("question") or "").strip()
        
        logger.info(f"Received question: {question}")
        
        if not question:
            return JSONResponse({"response": "Empty question."}, status_code=400)

        global DB
        if DB is None:
            logger.error("Knowledge base is empty")
            return JSONResponse({
                "response": "Knowledge base is empty. Please check MongoDB connection and rebuild.",
                "debug": "DB is None"
            }, status_code=400)

        chunks, embs, bm25 = DB["chunks"], DB["embeddings"], DB["_bm25"]
        logger.info(f"Working with {len(chunks)} chunks")

        q_emb = embedder.encode([question], convert_to_tensor=True)
        q_emb_dim = int(q_emb.shape[-1])
        store_dim = int(embs.shape[1]) if embs.ndim == 2 else 0

        if store_dim != q_emb_dim:
            logger.warning(f"Dimension mismatch: query={q_emb_dim}, store={store_dim}")
            n = build_vector_db()
            if n == 0:
                return JSONResponse({"response": "Rebuild failed."}, status_code=500)
            DB = load_from_vector_db()
            chunks, embs, bm25 = DB["chunks"], DB["embeddings"], DB["_bm25"]

        sem = util.cos_sim(q_emb, embs.to(q_emb.device)).squeeze(0).cpu().numpy() if len(embs) else []
        token_q = [w for w in re.findall(r"[a-z0-9]+", question.lower())]
        bm = bm25.get_scores(token_q) if bm25 else []

        import numpy as np
        def norm(a):
            a = np.array(a, dtype="float32")
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(a)

        hybrid = (0.6 * norm(sem) + 0.4 * norm(bm)if len(sem) and len(bm)else norm(sem)if len(sem)else norm(bm))

        top_idx = np.argsort(-hybrid)[:min(30, len(chunks))] if len(hybrid) else []

        if not len(top_idx):
            logger.warning("No candidates after hybrid scoring")
            return JSONResponse(get_ood_response())

        logger.info(f"Top semantic scores: {sem[top_idx[:5]] if len(sem) else []}")
        logger.info(f"Top BM25 scores: {bm[top_idx[:5]] if len(bm) else []}")

        candidates = [(question, chunks[i]) for i in top_idx]
        ce_scores = reranker.predict(candidates)
        reranked = sorted(zip(top_idx, ce_scores), key=lambda x: -x[1])[:min(6, len(top_idx))]

        if not reranked:
            logger.warning("No reranked results")
            return JSONResponse(get_ood_response())

        best_match_idx, best_ce_score_raw = reranked[0]
        best_ce_score = safe_float(best_ce_score_raw)

        best_chunk_emb = embedder.encode([chunks[best_match_idx]], convert_to_tensor=True)
        cos_tensor = util.cos_sim(q_emb, best_chunk_emb)
        sim_score = safe_float(cos_tensor)

        logger.info(f"Best match - Similarity: {sim_score:.3f}, CrossEncoder: {best_ce_score:.3f}")
        logger.info(f"Thresholds - MIN_SIM: {MIN_COSINE_SIM}, MIN_CE: {MIN_CROSS_ENCODER}")

        if sim_score < MIN_COSINE_SIM or best_ce_score < MIN_CROSS_ENCODER:
            logger.warning(
                f"REJECTED: Question doesn't match knowledge base. "
                f"Sim={sim_score:.3f} (need {MIN_COSINE_SIM}), "
                f"CE={best_ce_score:.3f} (need {MIN_CROSS_ENCODER})"
            )
            response = get_ood_response()
            response["debug"]["similarity_score"] = float(sim_score)
            response["debug"]["cross_encoder_score"] = float(best_ce_score)
            response["debug"]["thresholds"] = {
                "min_similarity": MIN_COSINE_SIM,
                "min_cross_encoder": MIN_CROSS_ENCODER
            }
            return JSONResponse(response)

        logger.info(f"✓ PASSED relevance check. Proceeding to generation.")
        logger.info(f"Best matching chunk preview: {chunks[best_match_idx][:200]}...")

        final_chunks = [chunks[i][:1800] for i, _ in reranked[:MAX_CHUNKS]]
        context = "\n\n---\n\n".join(final_chunks)[:MAX_CONTEXT_CHARS]
        logger.info(f"Context length: {len(context)} chars from {len(final_chunks)} chunks")

        # Extract Sanskrit text from context
        sanskrit_texts = []
        for chunk in final_chunks:
            sanskrit = extract_sanskrit_text(chunk)
            if sanskrit:
                sanskrit_texts.append(sanskrit)
        
        combined_sanskrit = "\n\n".join(sanskrit_texts) if sanskrit_texts else ""
        logger.info(f"Extracted Sanskrit text: {len(combined_sanskrit)} chars")

        prompt = f"""You are an expert Ayurvedic scholar. Using ONLY the context provided below, answer the question in a well-structured, detailed manner using proper Markdown formatting.

**CONTEXT:**
{context}

**QUESTION:** {question}

**IMPORTANT SANSKRIT HANDLING:**
{"- Sanskrit verses found in the context MUST be included in a separate '## Sanskrit Script' section at the very beginning of your response (after any brief opening line but before Introduction)." if combined_sanskrit else ""}
{"- Use the EXACT Sanskrit text provided below without any modification:" if combined_sanskrit else ""}
{combined_sanskrit if combined_sanskrit else ""}

**INSTRUCTIONS:**
1. {"If Sanskrit text exists, start with: '## Sanskrit Script' section containing the exact Sanskrit verses." if combined_sanskrit else ""}

2. Then provide a brief introductory paragraph (titled '## Introduction') that contextualizes the topic within Ayurvedic literature.

3. Then create a "## Detailed Explanation" section with:
   - Clear definition with Sanskrit terminology (if mentioned in context)
   - Key characteristics or principles
   - Specific examples or classifications from the texts
   - Therapeutic or clinical perspectives (if applicable)
   - Use **bold** for emphasis on important terms
   - Use proper markdown headers (##, ###) for sub-sections

4. If there are key characteristics, examples, or points to list, create a section like:
   ### Key Characteristics
   - Point 1 with explanation
   - Point 2 with explanation

5. End with a "## Summary" section that:
   - Synthesizes the main points in 4-6 sentences
   - Emphasizes the practical or conceptual significance
   - Relates back to the broader Ayurvedic framework

**IMPORTANT:**
- Use ONLY information explicitly stated in the context above
- Include Sanskrit terms ONLY if they appear in the context (format them in italics like *term*)
- **CRITICAL: Do NOT include any Devanagari/Sanskrit script in your response except in the Sanskrit Script section at the top**
- Do NOT quote or reproduce Sanskrit verses anywhere in the Introduction, Detailed Explanation, or Summary sections
- Only use transliterated Sanskrit terms (in italics) or English translations in the body of your response
- If the context doesn't contain enough information to answer comprehensively, respond with: "I don't have information about that in the provided texts."
- Use proper Markdown formatting throughout (headers, bold, italics, lists)
- Use ## for main sections (Detailed Explanation, Summary)
- Use ### for sub-sections
- Use **bold** for important Sanskrit terms or key concepts
- Use *italics* for transliterated Sanskrit terms in running text
- Use - or * for bullet points

**MARKDOWN FORMAT EXAMPLE:**

{"## Sanskrit Script" if combined_sanskrit else ""}
{combined_sanskrit if combined_sanskrit else ""}

{"## Introduction" if combined_sanskrit else "[Opening contextual paragraph with **important terms** in bold and *Sanskrit terms* in italics]"}

{"""[Brief contextual paragraph - NO Sanskrit script here, only transliterated terms like *Apasmara* or English translations]

## Detailed Explanation""" if combined_sanskrit else """## Detailed Explanation"""}

[Well-structured explanation - ONLY use transliterated Sanskrit terms in italics, NOT Devanagari script]

The Sanskrit term *Upadrava* literally means "that which follows."

### Key Characteristics

- **Occurs in the later phase** – It develops after the main disease has manifested
- **Depends on the main disease** – The same pathogenic factors trigger complications  
- **Varies in intensity** – It can be mild or severe

### Examples in Practice

[More detailed content - reference concepts from the verses using English or transliterated terms ONLY]

## Summary

[4-6 sentence synthesis - NO Sanskrit script, only English and transliterated terms in italics]

**REMEMBER: Sanskrit/Devanagari script appears ONLY in the "## Sanskrit Script" section. Everywhere else, use transliterated terms (like *jala*, *agni*) or English translations.**

Now write your response using proper Markdown formatting:"""

        logger.info("Generating structured answer...")
        answer = generate_text_local(prompt, max_new_tokens=2500, temperature=0.3)
        
        if not answer or len(answer.strip()) < 20:
            logger.warning("Generation produced empty or too short response")
            return JSONResponse(get_ood_response())

        # Validate that answer is complete
        if "## Summary" not in answer:
            logger.error("CRITICAL: Generated response missing Summary section even after retry")
            # Try one final emergency summary generation
            emergency_prompt = f"""Based on this incomplete Ayurvedic response, write ONLY a Summary section (4-6 sentences) that concludes the discussion:

{answer}

## Summary

"""
            try:
                emergency_outputs = generator(
                    emergency_prompt,
                    max_new_tokens=400,
                    temperature=0.3,
                    do_sample=False,  # Deterministic for emergency
                    pad_token_id=gen_tokenizer.eos_token_id,
                    return_full_text=False
                )
                emergency_summary = (emergency_outputs[0].get("generated_text") or "").strip()
                if emergency_summary:
                    answer = answer + "\n\n## Summary\n\n" + emergency_summary
                    logger.info("✓ Emergency summary added")
            except Exception as e:
                logger.error(f"Emergency summary generation failed: {e}")

        answer = answer.strip()
        
        # Clean up any unwanted prefixes
        answer = re.sub(r'^(Answer:|Response:|Write your response now:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # If Sanskrit was found but model didn't include it, prepend it
        if combined_sanskrit and "## Sanskrit Script" not in answer:
            answer = f"## Sanskrit Script\n\n{combined_sanskrit}\n\n{answer}"
        
        # POST-PROCESSING: Remove Devanagari script from everywhere EXCEPT the Sanskrit Script section
        if "## Sanskrit Script" in answer:
            # Split the response into sections
            parts = answer.split("## Sanskrit Script", 1)
            if len(parts) == 2:
                before_sanskrit = parts[0]
                after_sanskrit_section = parts[1]
                
                # Find where the Sanskrit Script section ends (next ## header)
                next_section_match = re.search(r'\n##\s+', after_sanskrit_section)
                if next_section_match:
                    sanskrit_section = after_sanskrit_section[:next_section_match.start()]
                    rest_of_response = after_sanskrit_section[next_section_match.start():]
                else:
                    # No next section found, keep as is
                    sanskrit_section = after_sanskrit_section
                    rest_of_response = ""
                
                # Remove ALL Devanagari from the rest of the response
                if rest_of_response:
                    # Remove lines containing Devanagari (including blockquotes)
                    lines = rest_of_response.split('\n')
                    cleaned_lines = []
                    skip_next = False
                    
                    for i, line in enumerate(lines):
                        # Check if line contains Devanagari
                        if re.search(r'[\u0900-\u097F]', line):
                            skip_next = True  # Skip the next empty line too
                            continue
                        # Skip empty lines after Devanagari lines
                        if skip_next and line.strip() == '':
                            skip_next = False
                            continue
                        skip_next = False
                        cleaned_lines.append(line)
                    
                    rest_of_response = '\n'.join(cleaned_lines)
                    # Clean up excessive blank lines
                    rest_of_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', rest_of_response)
                
                # Reconstruct the answer
                answer = before_sanskrit + "## Sanskrit Script" + sanskrit_section + rest_of_response
                answer = answer.strip()
                
                # Final validation
                if "## Summary" not in answer:
                    logger.warning("Response still incomplete after all attempts - Summary section missing")

        return JSONResponse({
            "response": answer,
            "debug": {
                "status": "success",
                "similarity_score": float(sim_score),
                "cross_encoder_score": float(best_ce_score),
                "num_chunks_used": int(len(final_chunks)),
                "context_length": int(len(context)),
                "sanskrit_found": bool(combined_sanskrit),
                "sanskrit_length": int(len(combined_sanskrit)),
                "passed_thresholds": True
            }
        })

    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({
            "response": "An error occurred processing your request.",
            "error": str(e)
        }, status_code=500)

@app.get("/status")
async def status():
    if DB is None:
        return JSONResponse({
            "status": "empty",
            "message": "Vector database not initialized. Check MongoDB connection."
        })
    
    meta = DB.get("meta", {})
    age = int(time.time() - meta.get("built_at", 0))
    
    return JSONResponse({
        "status": "ready",
        "num_chunks": int(len(DB["chunks"])),
        "num_content": int(meta.get("num_content", 0)),
        "num_dictionary": int(meta.get("num_dictionary", 0)),
        "num_uploads": int(meta.get("num_uploads", 0)),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.get("built_at", 0))),
        "seconds_since_build": int(age),
        "files": meta.get("files", []),
        "embed_model": meta.get("embed_model", "unknown"),
        "embedding_dim": int(meta.get("embedding_dim", 0)) if meta.get("embedding_dim") else None,
        "source_type": meta.get("source_type", "unknown"),
        "storage": "MongoDB",
        "source_db": SOURCE_DB,
        "vector_db": VECTOR_DB,
        "thresholds": {
            "min_cosine_similarity": MIN_COSINE_SIM,
            "min_cross_encoder": MIN_CROSS_ENCODER
        }
    })

@app.post("/rebuild")
async def rebuild():
    try:
        n = build_vector_db()
        if n == 0:
            return JSONResponse({
                "status": "failed", 
                "reason": "no text found",
                "message": "Check MongoDB connection and data"
            }, status_code=400)
        
        global DB
        DB = load_from_vector_db()

        
        meta = DB.get("meta", {})
        return JSONResponse({
            "status": "rebuilt",
            "num_chunks": int(len(DB["chunks"])) if DB else 0,
            "num_content": int(meta.get("num_content", 0)),
            "num_dictionary": int(meta.get("num_dictionary", 0)),
            "num_uploads": int(meta.get("num_uploads", 0)),
            "storage": "MongoDB",
            "source_db": SOURCE_DB,
            "vector_db": VECTOR_DB,
            "message": f"Successfully rebuilt with {n} chunks in MongoDB ({VECTOR_DB}.{VECTOR_COLLECTION})"
        })
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

@app.get("/debug/chunks")
async def debug_chunks():
    if DB is None:
        return JSONResponse({"error": "DB not loaded"})
    
    chunks = DB["chunks"]
    
    # Separate chunks by type
    content_chunks = [c for c in chunks if "[TEXTBOOK CONTENT]" in c]
    dict_chunks = [c for c in chunks if "[DICTIONARY TERM]" in c]
    upload_chunks = [c for c in chunks if "[UPLOAD:" in c]
    
    return JSONResponse({
        "total_chunks": int(len(chunks)),
        "content_chunks": int(len(content_chunks)),
        "dictionary_chunks": int(len(dict_chunks)),
        "upload_chunks": int(len(upload_chunks)),
        "sample_content": content_chunks[0][:300] if content_chunks else "No content chunks",
        "sample_dictionary": dict_chunks[0][:300] if dict_chunks else "No dictionary chunks",
        "chunk_lengths": [int(len(c)) for c in chunks[:10]],
        "storage": "MongoDB"
    })

@app.post("/test_mongo")
async def test_mongo():
    try:
        texts = fetch_text_from_mongodb()
        dict_texts = fetch_dictionary_from_mongodb()
        
        return JSONResponse({
            "status": "success",
            "content_documents": int(len(texts)),
            "dictionary_entries": int(len(dict_texts)),
            "sample_content": (
                f"[TEXTBOOK CONTENT]\nBook: {texts[0]['textBook name']}\nTopic: {texts[0]['topic name']}\nSubject: {texts[0]['subject name']}\n\n{texts[0]['Content description'][:420]}..."
                if texts else "No content data"
            ),
            "sample_dictionary": (
                f"[DICTIONARY TERM]\nWord: {dict_texts[0]['word']}\n\nDefinition: {dict_texts[0]['word_description'][:500]}..."
                if dict_texts else "No dictionary data"
            )
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

# ==========================================
# Add CORS Middleware for Postman/Browser
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "RAG Backend API is running!",
        "status": "online",
        "storage": "MongoDB Vector Storage",
        "source_database": f"{SOURCE_DB} (content1, dictionary)",
        "vector_database": f"{VECTOR_DB} (embeddings)",
        "endpoints": {
            "POST /generate": "Ask questions",
            "GET /status": "Check system status",
            "POST /rebuild": "Rebuild vector database",
            "GET /debug/chunks": "View sample chunks",
            "POST /test_mongo": "Test MongoDB connection"
        }
    }

# ==========================================
# Start Server with Ngrok
# ==========================================
print("🚀 Starting FastAPI server...")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("⏳ Waiting for server to initialize...")
time.sleep(8)

print("🌐 Creating Ngrok tunnel...")
try:
    tunnel = ngrok.connect(PORT, bind_tls=True)
    public_url = str(tunnel.public_url)
    
    print("\n✅ Models ready on CUDA" if device == "cuda" else "\n✅ Models ready on CPU")
    print("✅ MongoDB Vector Storage connected")
    print(f"✅ Public URL: {public_url}")
    print(f"🔗 POST /generate → {public_url}/generate")
    print(f"ℹ️ GET /status → {public_url}/status")
    print(f"🧰 POST /rebuild → {public_url}/rebuild")
    print(f"🪪 GET /debug/chunks → {public_url}/debug/chunks")
    print(f"🧪 POST /test_mongo → {public_url}/test_mongo")
    
    print("\n🟢 Server is running... (Ctrl+C to stop)")
    while True:
        time.sleep(10)
        
except Exception as e:
    print(f"\n❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    
except KeyboardInterrupt:
    print("\n🛑 Shutting down server...")
    ngrok.kill()
    print("✅ Server stopped.")
