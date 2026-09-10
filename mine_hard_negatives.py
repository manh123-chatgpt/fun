import os
import json
import pickle
from tqdm import tqdm
from data_loader import load_corpus, load_train_data
from bm25_retriever import BM25Searcher
from dense_retriever import DenseSearcher
from hybrid_retriever import HybridSearcher

def mine_multi_stage_hard_negatives(
    output_path="hard_negatives.pkl",
    top_bm25_k=30,
    top_dense_k=30,
    top_hybrid_k=30
):
    print("=" * 65)
    print("⛏️ BẮT ĐẦU KHAI THÁC HARD NEGATIVES ĐA NGUỒN (MULTI-STAGE)")
    print("=" * 65)
    
    corpus = load_corpus()
    train_data = load_train_data()
    
    print("📦 Đang khởi tạo các công cụ tìm kiếm...")
    bm25_searcher = BM25Searcher(corpus)
    dense_searcher = DenseSearcher(corpus)
    hybrid_searcher = HybridSearcher(corpus)
    
    hard_negatives = {}
    stats = {"bm25_only": 0, "dense_only": 0, "both": 0, "total_triplets": 0}
    
    for qid, item in tqdm(train_data.items(), desc="Đang khai thác Hard Negatives"):
        question = item.get("question", "").strip()
        answers = set([str(a) for a in item.get("answer", [])])
        
        if not question or not answers:
            continue
            
        # 1. Khai thác từ BM25 (Mẫu âm khó do trùng lặp từ khóa / lexical overlap)
        bm25_docs = bm25_searcher.search(question, top_k=top_bm25_k)
        bm25_hn = [str(d) for d in bm25_docs if str(d) not in answers]
        
        # 2. Khai thác từ Dense (Mẫu âm khó do gần nghĩa khái niệm / semantic similarity)
        dense_docs = dense_searcher.search(question, top_k=top_dense_k)
        dense_hn = [str(d) for d in dense_docs if str(d) not in answers]
        
        # 3. Khai thác từ Hybrid
        hybrid_docs = hybrid_searcher.search(question, top_k=top_hybrid_k)
        hybrid_hn = [str(d) for d in hybrid_docs if str(d) not in answers]
        
        # Hợp nhất và loại bỏ trùng lặp nhưng giữ thứ tự ưu tiên
        combined_hn = []
        seen = set()
        
        # Ưu tiên các hard negative xuất hiện trong cả BM25 và Dense (rất khó)
        intersect = set(bm25_hn[:10]) & set(dense_hn[:10])
        for doc_id in intersect:
            if doc_id not in seen:
                combined_hn.append(doc_id)
                seen.add(doc_id)
                stats["both"] += 1
                
        # Thêm các top negative từ hybrid và dense
        for doc_id in hybrid_hn[:15] + dense_hn[:15] + bm25_hn[:15]:
            if doc_id not in seen:
                combined_hn.append(doc_id)
                seen.add(doc_id)
                
        if combined_hn:
            hard_negatives[str(qid)] = combined_hn
            stats["total_triplets"] += len(combined_hn)
            
    with open(output_path, "wb") as f:
        pickle.dump(hard_negatives, f)
        
    print("\n" + "=" * 65)
    print(f"✅ Đã khai thác thành công Hard Negatives cho {len(hard_negatives)} câu hỏi!")
    print(f"📊 Tổng số mẫu âm khó: {stats['total_triplets']}")
    print(f"🎯 Mẫu âm siêu khó (gặp ở cả BM25 & Dense): {stats['both']}")
    print(f"💾 Đã lưu kết quả vào: '{output_path}'")
    print("=" * 65)

if __name__ == "__main__":
    mine_multi_stage_hard_negatives()

