import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# 先做 chunks 并进行 embedding
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "RAG_Corpus.txt"
CHROMA_PATH = BASE_DIR / "chroma_db"

embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


def chunk_by_paragraphs(text: str, min_chars=50, max_chars=800):
    parts = re.split(r"\n\s*\n", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p or len(p) < min_chars:
            continue

        # 过长：为了“最简单最小”先做硬切（后续你可改成按句切）
        if len(p) > max_chars:
            for i in range(0, len(p), max_chars):
                sub = p[i:i+max_chars].strip()
                if sub:
                    chunks.append(sub)
        else:
            chunks.append(p)
    return chunks



# 再创建 chroma 向量库
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
# 创建该 chromadb 中的一个命名空间 collection
collection = client.get_or_create_collection(name="rag_paragraph_chunks")


# 对 chunks 向量化并且存到 chroma_db 中的 collection 中
def add_chunks_to_chroma(chunks, metadatas=None):
    if not chunks:
        return

    embeddings = embed_model.encode(
       chunks,
       normalize_embeddings=True,
       show_progress_bar=True
    )

    if metadatas is None:
        metadatas = [{"source": "RAG_Corpus.txt", "chunk_index": i} for i in range(len(chunks))]
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# 读取语料，切 chunk，并写入 ChromaDB
def build_vector_store(corpus_path=CORPUS_PATH):
    with open(corpus_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = chunk_by_paragraphs(text)
    add_chunks_to_chroma(chunks)
    return chunks


def is_vector_store_empty():
    return collection.count() == 0


# 定义检索函数：先把 query embedding，再去 collection 中检索 top-k 个最相似 chunks
def retrieve_chunks(query: str, k=5):
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    return res
