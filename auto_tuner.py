import os
import json
import numpy as np
from tqdm import tqdm
from data_loader import load_corpus, load_train_data
from bm25_retriever import BM25Searcher
from dense_retriever import DenseSearcher, FineTunedDenseSearcher, E5LargeSearcher
from hybrid_retriever import expand_legal_query, extract_law_numbers
from evaluator import compute_metrics

def run_auto_tuner():
    print("="*70)
    print("🚀 AUTO-TUNER V2: K-FOLD CROSS-VALIDATION (ROBUST PARAMETER SEARCH)")
    print("="*70)

    corpus = load_corpus()
    train_data = load_train_data()
    
    all_items = list(train_data.items())
    print(f"✅ Đã nạp {len(corpus)} văn bản và {len(all_items)} câu hỏi.")

    # Khởi tạo các Searcher
    print("\n⚡ Đang nạp các Searcher Engines...")
    bm25 = BM25Searcher(corpus, use_cache=True)
    dense_bgem3 = DenseSearcher(corpus, use_cache=True)
    dense_finetuned = FineTunedDenseSearcher(corpus, use_cache=True)
    e5_searcher = E5LargeSearcher(corpus, use_cache=True)

    # Bản đồ số hiệu văn bản
    doc_law_numbers = {}
    for doc_id in corpus.keys():
        title_line = corpus[doc_id].split("\n")[0]
        laws = extract_law_numbers(title_line)
        if laws:
            doc_law_numbers[doc_id] = laws

    # Precompute candidates cho TOÀN BỘ 7000 câu bằng BATCHING (Cực Nhanh!)
    print("\n⚡ Đang trích xuất Top 150 ứng viên cho toàn bộ 7.000 câu hỏi (Batch Inference)...")
    
    # 1. Trích xuất tất cả câu hỏi và Query Expansion
    all_qids = []
    all_expanded_queries = []
    all_e5_queries = []
    exact_matched_dict = {}
    truths_dict = {}
    
    for qid, item in all_items:
        question = item["question"]
        all_qids.append(qid)
        truths_dict[qid] = item["answer"]
        
        query_laws = extract_law_numbers(question)
        exact_matched = []
        if query_laws:
            for doc_id, doc_laws in doc_law_numbers.items():
                for ql in query_laws:
                    if any(ql in dl or dl in ql for dl in doc_laws):
                        exact_matched.append(doc_id)
                        break
        exact_matched_dict[qid] = exact_matched
        
        expanded = expand_legal_query(question)
        all_expanded_queries.append(expanded)
        all_e5_queries.append(f"query: {expanded}")

    # 2. Batch Encoding (Tận dụng tối đa GPU)
    print("   👉 Đang encode bằng BGE-M3...")
    bgem3_q_emb = dense_bgem3.model.encode(all_expanded_queries, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    
    print("   👉 Đang encode bằng RoBERTa...")
    finetuned_q_emb = dense_finetuned.model.encode(all_expanded_queries, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    
    print("   👉 Đang encode bằng E5-Large...")
    e5_q_emb = e5_searcher.model.encode(all_e5_queries, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    # 3. Tính toán Top 150 bằng Matrix Multiplication (CPU/RAM xử lý cực nhanh)
    print("   👉 Đang tính toán ma trận điểm số...")
    precomputed = {}
    
    # Chạy qua từng câu hỏi để lấy top 150
    for i, qid in enumerate(tqdm(all_qids, desc="Ranking 150 Candidates")):
        expanded = all_expanded_queries[i]
        
        # Lấy BM25 (chạy đơn vì nó đã dùng thuật toán siêu tốc O(1) rồi)
        bm25_top = bm25.search(expanded, top_k=150)
        
        # Tính BGE-M3
        bgem3_scores = np.dot(dense_bgem3.corpus_embeddings, bgem3_q_emb[i])
        bgem3_max = {}
        for doc_id, score in zip(dense_bgem3.chunk_doc_ids, bgem3_scores):
            if doc_id not in bgem3_max or score > bgem3_max[doc_id]: bgem3_max[doc_id] = score
        bgem3_top = [doc_id for doc_id, _ in sorted(bgem3_max.items(), key=lambda x: x[1], reverse=True)[:150]]

        # Tính RoBERTa
        finetuned_scores = np.dot(dense_finetuned.corpus_embeddings, finetuned_q_emb[i])
        finetuned_max = {}
        for doc_id, score in zip(dense_finetuned.chunk_doc_ids, finetuned_scores):
            if doc_id not in finetuned_max or score > finetuned_max[doc_id]: finetuned_max[doc_id] = score
        finetuned_top = [doc_id for doc_id, _ in sorted(finetuned_max.items(), key=lambda x: x[1], reverse=True)[:150]]

        # Tính E5
        e5_scores = np.dot(e5_searcher.corpus_embeddings, e5_q_emb[i])
        e5_max = {}
        for doc_id, score in zip(e5_searcher.chunk_doc_ids, e5_scores):
            if doc_id not in e5_max or score > e5_max[doc_id]: e5_max[doc_id] = score
        e5_top = [doc_id for doc_id, _ in sorted(e5_max.items(), key=lambda x: x[1], reverse=True)[:150]]

        precomputed[qid] = {
            "bm25": bm25_top,
            "bgem3": bgem3_top,
            "finetuned": finetuned_top,
            "e5": e5_top,
            "exact_matched": exact_matched_dict[qid],
            "truth": truths_dict[qid]
        }

    # 5-Fold Cross-Validation Grid Search
    print("\n🔥 Đang chạy 5-Fold Cross-Validation Grid Search (Full Chunking)...")
    
    n = len(all_items)
    fold_size = n // 5
    qid_list = [qid for qid, _ in all_items]
    
    rrf_k_list = [3, 5, 10, 15]
    w_bm25_list = [0.8, 1.0, 1.2, 1.5]
    w_bgem3_list = [0.5, 0.8, 1.0]
    w_finetuned_list = [1.2, 1.5, 1.8, 2.0, 2.5]
    w_e5_list = [0.5, 0.8, 1.0]
    candidate_k_list = [100, 130, 150]

    best_avg_recall = 0.0
    best_config = {}
    total_configs = len(rrf_k_list) * len(w_bm25_list) * len(w_bgem3_list) * len(w_finetuned_list) * len(w_e5_list) * len(candidate_k_list)
    print(f"Tổng số cấu hình cần quét: {total_configs}")

    config_count = 0
    for rrf_k in rrf_k_list:
        for w_bm25 in w_bm25_list:
            for w_bgem3 in w_bgem3_list:
                for w_finetuned in w_finetuned_list:
                    for w_e5 in w_e5_list:
                        for cand_k in candidate_k_list:
                            config_count += 1
                            
                            # Tính Recall trên mỗi fold
                            fold_recalls = []
                            for fold_idx in range(5):
                                start = fold_idx * fold_size
                                end = start + fold_size if fold_idx < 4 else n
                                val_qids = set(qid_list[start:end])
                                
                                predictions = {}
                                val_truth = {}
                                
                                for qid in val_qids:
                                    cand = precomputed[qid]
                                    val_truth[qid] = cand["truth"]
                                    
                                    scores = {}
                                    for rank, doc_id in enumerate(cand["bm25"][:cand_k]):
                                        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * w_bm25
                                    for rank, doc_id in enumerate(cand["bgem3"][:cand_k]):
                                        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * w_bgem3
                                    for rank, doc_id in enumerate(cand["finetuned"][:cand_k]):
                                        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * w_finetuned
                                    for rank, doc_id in enumerate(cand["e5"][:cand_k]):
                                        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1)) * w_e5
                                    for doc_id in cand["exact_matched"]:
                                        scores[doc_id] = scores.get(doc_id, 0.0) + 5.0
    
                                    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                                    predictions[qid] = [doc_id for doc_id, _ in sorted_docs[:5]]

                            metrics = compute_metrics(predictions, val_truth, k=5)
                            fold_recalls.append(metrics["Recall"])

                        avg_recall = np.mean(fold_recalls)
                        
                        if avg_recall > best_avg_recall:
                            best_avg_recall = avg_recall
                            best_config = {
                                "rrf_k": rrf_k,
                                "w_bm25": w_bm25,
                                "w_bgem3": w_bgem3,
                                "w_finetuned": w_finetuned,
                                "w_e5": w_e5,
                                "candidate_k": cand_k
                            }
                            fold_str = " | ".join([f"F{i+1}={r*100:.1f}%" for i, r in enumerate(fold_recalls)])
                            print(f"🌟 [{config_count}/{total_configs}] Avg Recall = {avg_recall*100:.2f}% [{fold_str}] -> {best_config}")

    print("\n" + "="*70)
    print("🏆 BỘ THAM SỐ VÔ ĐỊCH (5-FOLD CROSS-VALIDATED):")
    print(f"👉 Average Recall@5 : {best_avg_recall * 100:.2f}%")
    for k, v in best_config.items():
        print(f"   - {k}: {v}")
    print("="*70)

if __name__ == "__main__":
    run_auto_tuner()
