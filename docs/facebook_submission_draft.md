# Facebook Submission Draft — AI Challenge 24h

- **Facebook Group:** https://www.facebook.com/groups/meitnghiencode (*Mê IT Nghiện Code*)
- **Certification Form (Step 2):** https://m4beo6fqhrl.sg.larksuite.com/share/base/form/shrlgB534MeoVcQaAjNI8oP2cBe
- **Deadline:** 12:00 (Trưa), Thứ Hai ngày 23/11/2026

---

## Nội dung bài viết (sao chép nguyên văn)

```text
[AI CHALLENGE 24H] - BÀI DỰ THI CỦA: Nguyễn Đức Bình
Sau workshop AI Engineer Roadmap cùng Viện AI4LIFE - ĐH Bách Khoa Hà Nội & Rikkei, mình đã hoàn thiện sản phẩm: ViCare V2 – Trợ lý hỏi đáp thông tin sức khỏe tiếng Việt bằng RAG có cưỡng chế trích dẫn và lớp bảo vệ an toàn.

1. Ý tưởng & Bài toán giải quyết:
- Bài toán: Xây dựng và cải tiến trợ lý hỏi đáp thông tin sức khỏe tiếng Việt (ViCare) trên bộ dữ liệu ViHealthQA (9.895 tài liệu đã khử trùng lặp và ẩn PII, 11.227 chunk), đánh giá công bằng trên tập cố định 100 câu dev (seed=2026) và so sánh trực tiếp trước/sau với hệ thống Baseline.
- Giải pháp: Nâng cấp toàn diện hệ thống RAG theo kiến trúc mô-đun hóa (package Python + FastAPI + giao diện Gradio + notebook Colab miễn phí GPU Tesla T4 để Ban giám khảo dễ tái lập):
  + Nâng cấp mô hình embedding sang google/embeddinggemma-2 (nạp chọn lọc chỉ nhánh văn bản 271M tham số thay vì 740M đa phương thức, tự chọn float32 gốc trên T4 SM 7.5 để tránh chậm do giả lập BF16, giữ baseline multilingual-e5-small chạy song song đối chứng).
  + Kết hợp Hybrid Retrieval (BM25 + Vector qua Weighted Reciprocal Rank Fusion 0.65/0.35) cùng tinh chỉnh hnsw_ef_search=200 cho không gian 768 chiều, đồng thời thực nghiệm nén chiều vector Matryoshka (MRL 768d → 512d → 256d → 128d).
  + Hiệu chỉnh tự động ngưỡng từ chối MIN_SCORE theo phân bố điểm cosine thực tế của từng model (0.6707 cho EmbeddingGemma 2 thay vì giữ cứng 0.86 của E5), nâng cấp nhận diện cấp cứu 115 tiếng Việt (hỗ trợ không dấu và xử lý phủ định theo mệnh đề như "không đau ngực", "chưa từng bị ngất hay co giật"), và bổ sung tầng kiểm tra/chuẩn hóa/sửa trích dẫn [1], [2] hậu sinh.

2. Công cụ & Kỹ thuật sử dụng:
- Framework/Thư viện: PyTorch, Hugging Face Transformers & Sentence-Transformers, ChromaDB, rank-bm25, Ollama, FastAPI, Gradio, Pytest (15 unit tests).
- Mô hình:
  + Embedding cải tiến: google/embeddinggemma-2 (271M text-only, 768d, MRL 512d/256d/128d) — đối chứng trực tiếp với Baseline intfloat/multilingual-e5-small (118M, 384d).
  + LLM sinh câu trả lời & LLM-as-a-judge: qwen2.5:3b (qua Ollama, temperature=0, seed=42).

3. Kết quả đạt được:
- Chạy ổn định trên Google Colab miễn phí (GPU Tesla T4 — 15 GB VRAM) với giao diện Gradio hiển thị trực tiếp nguồn trích dẫn, điểm tương đồng, ngưỡng từ chối và cảnh báo cấp cứu 115.
- Bảng so sánh Trước / Sau trên cùng 100 câu dev cố định (seed=2026) & 10 câu Judge@10:
  + Recall@3: 47.0% → 50.0% (+3.0 đ.%) | Recall@5: 51.0% → 54.0% (+3.0 đ.%) | Recall@10: 55.0% → 58.0% (+3.0 đ.%) | MRR@10: 42.7% → 42.8% (+0.1 đ.%) | Coverage: 100.0%. (Ở chế độ Dense-only với ef_search=200, EmbeddingGemma 2 đạt Recall@5 = 55.0%, Recall@10 = 60.0%, MRR@10 = 43.2%, tốc độ truy vấn 78.5 ms).
  + Judge@10: giữ vững 80.0% trong khi giảm tỷ lệ từ chối sai câu trong kho (in_scope_refusal) từ 20.0% xuống 10.0% (-10.0 đ.%) và giữ tỷ lệ chặn câu ngoài phạm vi (oos_refusal) đạt 100.0%.
  + Trích dẫn nguồn (citation_compliance): tăng từ 12.5% (1/8 câu) lên 88.9% tự nhiên ngay lượt đầu (8/9 câu) và đạt 100.0% sau 1 lượt sửa trích dẫn tự động (citation_repairs = 1).
  + Thực nghiệm Matryoshka (MRL): nén từ 768d (32.89 MB) xuống 512d (21.93 MB, giảm 1.5x) thậm chí đạt Recall@1 = 37.0% và MRR@10 = 43.3%; xuống 256d (10.96 MB, giảm 3x) vẫn giữ Recall@5 = 52.0%.
- Đánh đổi & Hạn chế đã nhận diện (trung thực):
  + Đánh đổi tài nguyên: EmbeddingGemma 2 (271M, 768d) tốn VRAM và thời gian lập chỉ mục lớn hơn Baseline (1034.7 MB vs 449.6 MB VRAM; 266.0s vs 71.3s lập chỉ mục; 205.4 ms Hybrid vs 22.4 ms Baseline) và Recall@1 trên Hybrid giảm nhẹ (-3.0 đ.%, 37.0% → 34.0%).
  + Giới hạn chỉ số trích dẫn: citation_compliance = 100% mang tính cấu trúc (bảo đảm có thẻ [k] hợp lệ trong phạm vi 1..5 nhờ cơ chế fail-closed), chưa thay thế kiểm định ngữ nghĩa (entailment) từng câu.
  + Giới hạn an toàn & dữ liệu: Nhận diện cấp cứu vẫn dựa trên bộ luật/từ điển (mở rộng từ 8 từ khóa gốc), chưa phải bộ phân loại lâm sàng; dữ liệu ViHealthQA đánh giá theo kho đóng và có thể đã cũ; Judge@10 trên 10 câu chỉ mang tính tham khảo. ViCare chỉ là demo học thuật, không thay thế tư vấn hay chẩn đoán của bác sĩ.

---
THÔNG TIN DÀNH CHO BAN GIÁM KHẢO:
- Link Source Code (Github/Colab để chế độ Public):
  + GitHub Repo: https://github.com/humman0id/vicare-rag
  + Google Colab Notebook (T4 miễn phí): https://colab.research.google.com/drive/1oobFwEBHSVCTt2S8qAH-UNI63OraRCHN
- Link Web Demo trực tiếp (nếu có): Chạy trực tiếp giao diện Gradio trong Mục 5 của Colab Notebook (link gradio.live tạm thời chỉ hoạt động khi phiên Colab đang mở; chi tiết demo đã đính kèm trong video/hình ảnh bài viết).
(Chi tiết thao tác chạy sản phẩm đã được đính kèm trong video/hình ảnh của bài viết).
#AIChallenge24h #AI4LIFE #Rikkei
```
