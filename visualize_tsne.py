import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_loader import load_corpus, load_train_data

def categorize_legal_domain(text):
    """
    Phân loại sơ bộ chủ đề pháp lý dựa vào từ khóa để tô màu trực quan trên biểu đồ t-SNE
    """
    text_lower = text.lower()
    if any(k in text_lower for k in ["lao động", "lương", "sa thải", "nghỉ việc", "bảo hiểm", "hđlđ", "công đoàn"]):
        return "Lao động & BHXH"
    elif any(k in text_lower for k in ["hình sự", "tội phạm", "tù", "truy cứu", "án phạt", "bị can", "bị cáo", "vũ lực"]):
        return "Hình sự & Tố tụng"
    elif any(k in text_lower for k in ["đất đai", "sổ đỏ", "bồi thường", "tái định cư", "nhà ở", "xây dựng", "quy hoạch"]):
        return "Đất đai & Nhà ở"
    elif any(k in text_lower for k in ["thuế", "doanh nghiệp", "đăng ký kinh doanh", "hợp đồng", "kinh doanh", "đấu thầu", "giá"]):
        return "Kinh doanh & Thuế"
    elif any(k in text_lower for k in ["hôn nhân", "gia đình", "ly hôn", "ly dị", "con cái", "kết hôn"]):
        return "Hôn nhân & Gia đình"
    elif any(k in text_lower for k in ["giao thông", "xe máy", "ô tô", "bằng lái", "vi phạm hành chính", "phạt nguội"]):
        return "Giao thông & Xử phạt"
    else:
        return "Khác"

def run_tsne_visualization(
    model_path="fine_tuned_vietnamese_bi_encoder",
    num_samples=150,
    output_image="embedding_tsne_visualization.png",
    random_state=42
):
    print("=" * 65)
    print("📊 TRỰC QUAN HÓA KHÔNG GIAN EMBEDDING VỚI t-SNE (LEGAL IR)")
    print("=" * 65)

    # 1. Nạp dữ liệu
    corpus = load_corpus()
    train_data = load_train_data()
    
    # 2. Nạp mô hình embedding
    print(f"🤖 Đang nạp mô hình Embedding: '{model_path}'...")
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        print(f"⚠️ Không nạp được {model_path}, fallback sang bkai-foundation-models/vietnamese-bi-encoder: {e}")
        model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

    # 3. Lấy mẫu câu hỏi & văn bản tương ứng
    queries = []
    documents = []
    labels = []
    
    count = 0
    for qid, item in train_data.items():
        if count >= num_samples:
            break
        q_text = item.get("question", "").strip()
        ans_ids = item.get("answer", [])
        if not q_text or not ans_ids:
            continue
        
        valid_ans_ids = [str(a) for a in ans_ids if str(a) in corpus]
        if not valid_ans_ids:
            continue
            
        target_doc_id = valid_ans_ids[0]
        doc_text = corpus[target_doc_id][:350]  # Cắt ngắn để trích xuất đại diện
        
        domain = categorize_legal_domain(q_text)
        
        queries.append(q_text)
        labels.append(domain)
        
        documents.append(doc_text)
        count += 1

    all_texts = queries + documents

    print(f"📝 Đang mã hóa {len(all_texts)} câu và đoạn văn bản sang vector 768 chiều...")
    embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    # 4. Giảm chiều bằng t-SNE (768D -> 2D)
    print("🌀 Đang chạy thuật toán t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_texts) - 1), random_state=random_state, max_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 5. Vẽ biểu đồ với Matplotlib
    print("🎨 Đang vẽ biểu đồ phân bố không gian ngữ nghĩa...")
    unique_domains = list(set(labels))
    colors = plt.cm.get_cmap('tab10', len(unique_domains))
    domain_to_color = {d: colors(i) for i, d in enumerate(unique_domains)}

    plt.figure(figsize=(14, 10), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Vẽ đường nối giữa Query và Document tương ứng để thấy sự hội tụ ngữ nghĩa
    half = len(queries)
    for i in range(half):
        q_coord = reduced_embeddings[i]
        d_coord = reduced_embeddings[half + i]
        plt.plot([q_coord[0], d_coord[0]], [q_coord[1], d_coord[1]], color='gray', alpha=0.25, linestyle='--', linewidth=0.8)

    # Vẽ các điểm Query (hình tròn tròn)
    for domain in unique_domains:
        idxs = [i for i in range(half) if labels[i] == domain]
        if idxs:
            plt.scatter(
                reduced_embeddings[idxs, 0],
                reduced_embeddings[idxs, 1],
                color=domain_to_color[domain],
                marker='o',
                s=65,
                alpha=0.85,
                label=f"Query ({domain})"
            )

    # Vẽ các điểm Document (hình vuông)
    for domain in unique_domains:
        idxs = [half + i for i in range(half) if labels[i] == domain]
        if idxs:
            plt.scatter(
                reduced_embeddings[idxs, 0],
                reduced_embeddings[idxs, 1],
                color=domain_to_color[domain],
                marker='s',
                s=75,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5,
                label=f"Doc ({domain})"
            )

    plt.title("t-SNE 2D Visualization of Legal Embedding Space\n(Query & Target Document Semantic Alignment)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu biểu đồ t-SNE thành công vào: '{output_image}'")

if __name__ == "__main__":
    run_tsne_visualization()
