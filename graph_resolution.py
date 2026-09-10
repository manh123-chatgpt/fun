import json
import re
from tqdm import tqdm

from data_loader import load_corpus

def resolve_references(output_path="legal_corpus_resolved.json"):
    print("Đang nạp Corpus gốc...")
    corpus = load_corpus(force_original=True)
        
    resolved_corpus = {}
    
    # Regex để tìm số Điều
    article_extract_pattern = re.compile(r'^Điều\s+(\d+[a-zA-Z]?)', re.IGNORECASE)
    reference_pattern = re.compile(r'Điều\s+(\d+[a-zA-Z]?)', re.IGNORECASE)
    
    total_references_resolved = 0
    
    for doc_id, text in tqdm(corpus.items(), desc="Đang phân tích Graph Reference"):
        lines = text.split("\n", 1)
        title = lines[0] if lines else ""
        body = lines[1] if len(lines) > 1 else text
        
        # Chia văn bản thành các Điều
        parts = re.split(r'(?=\bĐiều\s+\d+)', body)
        
        article_dict = {}
        # Lược đồ các Điều
        for part in parts:
            part = part.strip()
            if not part: continue
            
            match = article_extract_pattern.match(part)
            if match:
                article_num = match.group(1).lower()
                article_dict[article_num] = part
                
        resolved_parts = []
        for part in parts:
            part = part.strip()
            if not part: continue
            
            match = article_extract_pattern.match(part)
            current_article_num = match.group(1).lower() if match else None
            
            # Tìm tất cả các tham chiếu trong Điều này
            refs = reference_pattern.findall(part)
            unique_refs = set([r.lower() for r in refs])
            
            appended_text = ""
            for ref in unique_refs:
                if ref != current_article_num and ref in article_dict:
                    # Lấy 500 ký tự đầu tiên của Điều được tham chiếu để tránh làm phình văn bản quá mức
                    ref_text = article_dict[ref][:500] 
                    if len(article_dict[ref]) > 500:
                        ref_text += "..."
                    appended_text += f"\n[Nội dung tham chiếu Điều {ref}]: {ref_text}"
                    total_references_resolved += 1
                    
            resolved_parts.append(part + appended_text)
            
        resolved_body = "\n\n".join(resolved_parts)
        resolved_corpus[doc_id] = f"{title}\n{resolved_body}"
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resolved_corpus, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Đã xử lý xong! Tổng cộng giải quyết thành công {total_references_resolved} tham chiếu chéo.")
    print(f"💾 Đã lưu Corpus mới vào '{output_path}'.")

if __name__ == "__main__":
    resolve_references()
