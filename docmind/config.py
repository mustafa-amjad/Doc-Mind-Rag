"""Central settings. Every other file reads from here, so you change things in ONE place."""
import os  # lets us read environment variables
from pathlib import Path  # builds file paths that work on Windows and Mac

from dotenv import load_dotenv  # reads the .env file

load_dotenv()  # load .env into environment variables (does nothing if no .env)

# ---- keys and model names ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # your Groq key, empty string if missing
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # the "kid" who writes answers
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # text -> 384 numbers
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # the "strict teacher"

# ---- chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # max characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # characters repeated between neighbour chunks
MIN_CHUNK_CHARS = 30  # throw away tiny junk chunks (page numbers, etc.)
SECTION_CHARS = 3000  # DOCX/TXT/HTML are grouped into sections this big (section number = 'page')

# ---- retrieval ----
RETRIEVE_K = 10  # how many candidates each search (vector, keyword) brings back
FINAL_K = 4  # how many chunks we finally give to the LLM
RRF_K = 60  # RRF smoothing constant (60 is the standard value from the paper)

# ---- FAISS index choice ----
FLAT_LIMIT = 2000  # fewer chunks than this -> exact Flat search, otherwise HNSW
HNSW_M = 32  # neighbours per node in the HNSW graph
HNSW_EF_CONSTRUCTION = 200  # build quality (higher = better graph, slower build)
HNSW_EF_SEARCH = 64  # search quality (higher = more accurate, slower)

# ---- storage (accounts, saved chats, saved indexes) ----
ROOT_DIR = Path(__file__).resolve().parent.parent  # project root folder
DATA_DIR = ROOT_DIR / "data"  # everything we save lives here
DB_PATH = DATA_DIR / "docmind.db"  # SQLite file: users, chats, messages
INDEX_DIR = DATA_DIR / "indexes"  # one folder per chat: chunks + vectors

# ---- chat memory ----
HISTORY_TURNS = 6  # how many recent messages the AI "remembers" in a chat

# ---- login sessions (stay logged in after refresh) ----
COOKIE_NAME = "docmind_session"  # name of the browser cookie
SESSION_DAYS = 30  # how long a login stays valid

# ---- web links ----
URL_TIMEOUT = 10  # seconds to wait for a website
MAX_URL_BYTES = 10 * 1024 * 1024  # 10 MB download limit
MAX_REDIRECTS = 3  # follow at most 3 redirects

# ---- OCR (scanned PDFs) ----
OCR_SCALE = 2.0  # render pages at 2x zoom before reading (sharper text)
MAX_OCR_PAGES = 50  # per file, so one huge scan cannot freeze the server
