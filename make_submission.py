import json
import zipfile
from tqdm import tqdm
from data_loader import load_corpus, load_test_data
from hybrid_retriever import HybridSearcher

def main():
    # 1. Nạp kho luật và tập test chính thức
    corpus = load_corpus()
    test_data = load_test_data()
    print(f"Số lượng câu hỏi trong tập test: {len(test_data)}")

    # 2. Khởi tạo Pipeline Hybrid SOTA
    searcher = HybridSearcher(corpus)

    # 2.5 Nạp sức mạnh HyDE (nếu có) - ĐÃ TẮT BỎ VÌ LLM ẢO GIÁC
    import os
    hyde_data = {}
    # if os.path.exists("test_hyde.json"):
    #     with open("test_hyde.json", "r", encoding="utf-8") as f:
    #         hyde_data = json.load(f)
    #         print(f"\n🔥 KÍCH HOẠT VŨ KHÍ TỐI THƯỢNG (HyDE): Đã nạp {len(hyde_data)} câu trả lời giả lập!")

    # 3. Dự đoán chính xác theo từng câu hỏi của test_data
    submission = {}
    print("\n🚀 Đang chạy dự đoán HYBRID SOTA cho TOÀN BỘ câu hỏi Public Test...")
    for qid, item in tqdm(test_data.items(), desc="Predicting SOTA Hybrid"):
        question = item["question"]
        
        # Gộp câu hỏi và câu trả lời giả lập để search
        search_query = question # + " " + hyde_data.get(str(qid), "")
        
        top_docs = searcher.search(search_query, top_k=5)
        
        submission[str(qid)] = {
            "answer": [str(d) for d in top_docs[:5]]
        }

    print(f"\nTổng số câu hỏi đã dự đoán: {len(submission)}")

    # 4. Ghi file submission.json và nén thành submission.zip
    output_json = "submission.json"
    output_zip = "submission.zip"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname="submission.json")

    print(f"🎉 Đã tạo thành công '{output_zip}'! Sẵn sàng để nộp bài.")

if __name__ == "__main__":
    main()
