import os
import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from tqdm import tqdm
from data_loader import load_corpus, load_train_data
import re
import numpy as np
from rank_bm25 import BM25Okapi
import string

# 1. Định nghĩa hàm Legal Chunking
def chunk_document(text, chunk_size=1200, overlap=200, max_chunks=150):
    lines = text.split("\n", 1)
    title = lines[0] if lines else ""
    body = lines[1] if len(lines) > 1 else text
    if len(body) <= chunk_size: return [f"{title}\n{body}"]
    chunks = []
    start = 0
    while start < len(body) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunks.append(f"{title}\n{body[start:end]}")
        start += (chunk_size - overlap)
    return chunks

def legal_chunk_document(text, max_chunk_chars=1200, overlap=200, max_chunks=150):
    lines = text.split("\n", 1)
    title = lines[0] if lines else ""
    body = lines[1] if len(lines) > 1 else text
    parts = re.split(r'(?=Điều\s+\d+)', body)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2: return chunk_document(text, chunk_size=max_chunk_chars, overlap=overlap, max_chunks=max_chunks)
    chunks = []
    buffer = ""
    for part in parts:
        if buffer and len(buffer) + len(part) <= max_chunk_chars:
            buffer = buffer + "\n" + part
        else:
            if buffer: chunks.append(f"{title}\n{buffer}")
            if len(part) > max_chunk_chars:
                start = 0
                while start < len(part) and len(chunks) < max_chunks:
                    end = start + max_chunk_chars
                    chunks.append(f"{title}\n{part[start:end]}")
                    start += max_chunk_chars - overlap
                buffer = ""
            else:
                buffer = part
        if len(chunks) >= max_chunks: break
    if buffer and len(chunks) < max_chunks: chunks.append(f"{title}\n{buffer}")
    return chunks[:max_chunks] if chunks else [text[:max_chunk_chars]]

def get_best_chunk_bm25(question, text, max_chunk_chars=600):
    """Tìm Chunk tốt nhất trong tài liệu chứa câu trả lời cho câu hỏi bằng BM25"""
    chunks = legal_chunk_document(text, max_chunk_chars=max_chunk_chars, overlap=100, max_chunks=150)
    if len(chunks) == 1:
        return chunks[0]
    
    # Chuẩn bị cho BM25
    tokenized_chunks = [ch.lower().translate(str.maketrans('', '', string.punctuation)).split() for ch in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    tokenized_query = question.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25.get_scores(tokenized_query)
    
    # Lấy chunk có điểm BM25 cao nhất với câu hỏi
    best_idx = np.argmax(scores)
    return chunks[best_idx]

OUTPUT_MODEL_DIR = "fine_tuned_vietnamese_bi_encoder"
BASE_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
EPOCHS = 5  # 5 Epochs (Hard Negatives mới từ Top 50 sẽ tạo sự khác biệt)
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 256  # RoBERTa chỉ hỗ trợ tối đa 256 positional embeddings
LR = 1.5e-5

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*65)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN LEGAL BI-ENCODER SUPER (5 EPOCHS) TRÊN: {device.upper()}")
    print("="*65)

    # 1. Nạp dữ liệu
    corpus = load_corpus()
    train_data = load_train_data()
    print(f"Đã nạp {len(corpus)} văn bản và {len(train_data)} câu hỏi train.")

    # Thử nạp Hard Negatives
    hard_negatives = {}
    if os.path.exists("hard_negatives.pkl"):
        import pickle
        import random
        with open("hard_negatives.pkl", "rb") as f:
            hard_negatives = pickle.load(f)
        print(f"🔥 Đã nạp Hard Negatives cho {len(hard_negatives)} câu hỏi.")

    # 2. Xây dựng tập huấn luyện: Ghép cặp câu hỏi với TẤT CẢ văn bản đáp án đúng
    train_examples = []
    print("Đang chuẩn bị các cặp (Câu hỏi, TẤT CẢ văn bản đúng 1.200 ký tự)...")
    
    multi_count = 0
    triplet_count = 0
    for qid, item in train_data.items():
        question = item.get("question", "").strip()
        answers = item.get("answer", [])
        if not question or not answers:
            continue
        
        # Tạo cặp train cho MỌI văn bản đáp án đúng, không chỉ answers[0]
        for ans_id in answers:
            target_doc_id = str(ans_id)
            if target_doc_id in corpus:
                # [PASSAGE-LEVEL FINE-TUNING]: Tìm ĐÚNG CHUNK chứa câu trả lời thay vì cắt 1200 ký tự đầu
                doc_text = get_best_chunk_bm25(question, corpus[target_doc_id], max_chunk_chars=600)
                
                # Nạp thêm Hard Negative nếu có
                hn_text = None
                if hard_negatives and qid in hard_negatives and hard_negatives[qid]:
                    # Chọn ngẫu nhiên 1 hard negative
                    hn_id = random.choice(hard_negatives[qid])
                    if hn_id in corpus:
                        hn_text = get_best_chunk_bm25(question, corpus[hn_id], max_chunk_chars=600)
                
                if hn_text:
                    train_examples.append(InputExample(texts=[question, doc_text, hn_text]))
                    triplet_count += 1
                else:
                    train_examples.append(InputExample(texts=[question, doc_text]))
        
        if len(answers) > 1:
            multi_count += 1

    print(f"✅ Đã tạo {len(train_examples)} mẫu huấn luyện (gồm {triplet_count} mẫu có Hard Negatives).")

    # 3. Tạo DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True
    )

    # 4. Khởi tạo mô hình BGE-M3
    print(f"⚡ Đang tải mô hình gốc '{BASE_MODEL_NAME}'...")
    model = SentenceTransformer(
        BASE_MODEL_NAME,
        device=device,
        model_kwargs={"use_safetensors": True}
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    # 5. Hàm mất mát MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 6. Bắt đầu Train với Mixed Precision (AMP)
    print(f"🔥 Đang huấn luyện trong {EPOCHS} Epochs (Batch size: {BATCH_SIZE}, Max Length: {MAX_SEQ_LENGTH})...")
    
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        show_progress_bar=True,
        output_path=OUTPUT_MODEL_DIR,
        use_amp=True
    )

    print("\n" + "="*60)
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT! Mô hình đã được lưu tại thư mục: '{OUTPUT_MODEL_DIR}'")
    print("="*60)

if __name__ == "__main__":
    train()
