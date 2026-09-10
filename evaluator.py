import numpy as np
import json

def compute_metrics(predictions, ground_truth, k=5):
    """
    Tính Recall và Precision theo đúng luật cuộc thi UIT Data Challenge 2026.
    predictions: dict { "question_id": ["doc_id_1", "doc_id_2", ...] } hoặc { "question_id": {"answer": [...]} }
    ground_truth: dict { "question_id": ["true_doc_id_1", ...] }
    k: số lượng văn bản tối đa được phép trả về (mặc định là 5).
    """
    recalls = []
    precisions = []

    for qid, true_docs in ground_truth.items():
        # Lấy danh sách dự đoán
        pred_entry = predictions.get(qid, [])
        if isinstance(pred_entry, dict) and "answer" in pred_entry:
            pred_docs = pred_entry["answer"] or []
        elif isinstance(pred_entry, list):
            pred_docs = pred_entry
        else:
            pred_docs = []

        # Ép kiểu sang chuỗi và loại bỏ trùng lặp
        pred_docs = [str(d) for d in pred_docs]
        true_docs = [str(d) for d in true_docs]

        # Ràng buộc của BTC: Phải từ 1 đến tối đa k (5) văn bản
        num_preds = len(pred_docs)
        if num_preds == 0 or num_preds > k:
            recalls.append(0.0)
            precisions.append(0.0)
            continue

        true_set = set(true_docs)
        pred_set = set(pred_docs)
        hits = len(true_set & pred_set)

        # Tính Recall & Precision cho câu hỏi này
        rec = hits / len(true_set) if len(true_set) > 0 else 0.0
        prec = hits / num_preds if num_preds > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0

    return {
        "Recall": mean_recall,
        "Precision": mean_precision
    }

if __name__ == "__main__":
    # Chạy thử nghiệm hàm chấm điểm với ví dụ nhỏ
    dummy_truth = {
        "q1": ["doc_A"],
        "q2": ["doc_B", "doc_C"],
        "q3": ["doc_D"]
    }
    dummy_pred = {
        "q1": ["doc_A", "doc_X"],        # Đúng 1/1 -> Recall = 1.0, Precision = 0.5
        "q2": ["doc_B"],                 # Đúng 1/2 -> Recall = 0.5, Precision = 1.0
        "q3": ["1", "2", "3", "4", "5", "6"] # Vi phạm > 5 IDs -> Recall = 0.0, Precision = 0.0
    }
    scores = compute_metrics(dummy_pred, dummy_truth)
    print("Test thử bộ chấm điểm:")
    print(f"Recall trung bình: {scores['Recall']:.4f}")
    print(f"Precision trung bình: {scores['Precision']:.4f}")
