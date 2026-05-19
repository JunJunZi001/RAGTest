import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from RAG import build_vector_store, is_vector_store_empty, retrieve_chunks


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """你是一个 RAG 知识问答助手。
你的任务是根据检索到的资料回答用户问题。
要求：
1. 优先使用检索资料中的信息，不要凭空编造。
2. 如果资料没有覆盖，要明确说“资料中没有提到”。
3. 回答要适合面试复盘，尽量解释流程和原因。
"""

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
    )
    return tokenizer, model


def format_retrieved_context(retrieval_result):
    docs = retrieval_result["documents"][0]
    metadatas = retrieval_result["metadatas"][0]
    distances = retrieval_result["distances"][0]

    context_blocks = []
    for i, (doc, metadata, distance) in enumerate(zip(docs, metadatas, distances), start=1):
        chunk_index = metadata.get("chunk_index", "unknown")
        context_blocks.append(
            f"[资料{i}] chunk_index={chunk_index}, distance={distance:.4f}\n{doc}"
        )

    return "\n\n".join(context_blocks)


def build_messages(query, retrieved_context):
    user_prompt = f"""请根据下面的检索资料回答问题。

检索资料：
{retrieved_context}

问题：
{query}

请给出最终回答："""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(query, k=3):
    retrieval_result = retrieve_chunks(query, k=k)
    retrieved_context = format_retrieved_context(retrieval_result)

    tokenizer, model = load_llm()
    messages = build_messages(query, retrieved_context)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([prompt_text], return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
    )
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return answer, retrieved_context


if __name__ == "__main__":
    if is_vector_store_empty():
        chunks = build_vector_store()
        print(f"ChromaDB 为空，已完成建库，chunks: {len(chunks)}")
    else:
        print("ChromaDB 已有数据，跳过建库。")

    query = "overlap（重叠）在 chunking 中的目的是什么？"
    answer, retrieved_context = generate_answer(query)

    print("用户问题：")
    print(query)
    print("\n召回的 chunks：")
    print(retrieved_context)
    print("\n模型回答：")
    print(answer)
