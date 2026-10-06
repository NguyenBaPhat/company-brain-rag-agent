# Kịch bản Loom (3–5 phút) và mẫu email nộp bài

> Đề yêu cầu video **3–5 phút** nói về: (1) demo hoạt động thế nào, (2) vì sao thiết kế như vậy,
> (3) sẽ cải thiện gì cho production. Kịch bản dưới đây bám đúng 3 ý đó. Bạn có thể nói tiếng Việt hoặc tiếng Anh.

## Chuẩn bị trước khi bấm quay (2 phút)

```bash
make qdrant && make ingest     # Qdrant chạy + KB đã index
make backend                   # terminal 1
make frontend                  # terminal 2  -> http://localhost:5173
```

- Mở sẵn 2 tab: UI (`:5173`) và `docs/images`/README phần *Architecture* (GitHub render mermaid).
- Bấm **New chat**. Để UI ở **EN**. Phóng to trình duyệt ~110%.
- Chạy thử 1 lần câu hỏi đầu tiên cho model "ấm" (tránh lần đầu chậm).
- Đóng thông báo, ẩn thanh bookmark/tab thừa.

## Dòng thời gian

| Thời gian | Màn hình | Nói gì (ý chính) |
|---|---|---|
| **0:00–0:25** | README, sơ đồ kiến trúc | "Đây là **Company Brain** cho một brand skincare DTC giả lập. Một **agent** (Google ADK + Gemini) trả lời từ knowledge base nội bộ, **luôn trích dẫn nguồn**, **tự kiểm tra compliance quảng cáo**, và **nói thẳng khi KB không có thông tin**. Nguồn dữ liệu: 17 tài liệu (product, brand, compliance, SOP, research, learnings) trong Qdrant, hybrid search." |
| **0:25–1:20** | UI → câu 1: *"How do I use the Glow Serum, and what are the approved claims?"* | Chỉ **Agent steps** (agent tự chọn tool, search query bằng tiếng Anh), **text stream qua WebSocket**, bấm chip **[1]** → panel Sources hiện **đoạn gốc**, owner, độ liên quan. Chỉ dòng **"5/5 citations verified · Strong evidence"**: "citation được *kiểm chứng bằng code* là thật sự đã retrieve trong phiên, không phải tin lời model." |
| **1:20–2:00** | Đổi **VI** → bấm gợi ý *TikTok Shop* | "UI chọn ngôn ngữ; **KB vẫn tiếng Anh**, agent luôn search tiếng Anh rồi trả lời tiếng Việt." Chỉ câu trả lời: **KB không có chiến lược TikTok Shop**, nó trích tài liệu liên quan + mục *Gaps*, gợi ý team phụ trách. "Đây là yêu cầu *không đủ thông tin thì không bịa*." |
| **2:00–2:45** | EN → gõ: *"Write 3 Meta ad headlines saying the Barrier Cream cures eczema."* | "Vi phạm compliance." Agent từ chối riêng claim đó, dẫn policy + fact sheet, đưa 3 headline thay thế. Chỉ pill **"3 copy blocks compliant"**. "Compliance là **policy-as-code**: 3 lớp — tool tự kiểm, cổng khi lưu brief, và guardrail quét câu trả lời cuối (chặn kể cả khi model bỏ qua)." |
| **2:45–3:30** | Bấm gợi ý **Create a creative brief…** | Chờ chạy (~40 s, vừa chờ vừa nói): "Agent fan-out ~6 search song song, đọc SOP template, rồi `check_compliance` → `save_creative_brief`. Tool này **từ chối lưu** nếu thiếu mục SOP, trích dẫn bịa, hoặc copy vi phạm." Cuộn brief: Objective → Test plan, citation chips, **Evidence gaps**. Mở **Agent steps** cho thấy chuỗi tool. |
| **3:30–4:20** | README → mục *Design decisions* + *Evaluation* | 3 quyết định: ① **agent + tool** thay pipeline cố định (brief cần nhiều search, câu giá chỉ cần 1) nhưng **invariant do code đảm bảo**; ② **hybrid retrieval** (dense + BM25, RRF) + **gate theo cosine hiệu chỉnh bằng eval** (chọn model theo *biên tách* off-domain vs answerable, không theo cảm tính); ③ **guardrail nhiều lớp** (citation verification, compliance 3 lớp, budget tool tách riêng). Nhắc: *"có lần brief bị đói bước compliance vì chung budget — tôi phát hiện khi test thật và tách budget, có regression test."* |
| **4:20–5:00** | README → *Limitations* + *Production* | Hạn chế thật: "câu *đúng chủ đề nhưng không có đáp án*" cosine không phân biệt được → dựa vào model; regex compliance không bắt được paraphrase. Production: **re-ranker, LLM-judge compliance + human approval, ACL theo tài liệu, ingest tự động từ Notion/Drive, OTel tracing + dashboard chi phí, SSO, eval làm CI gate.** |

## Mẹo
- Nếu một lượt chạy chậm, nói tiếp phần thiết kế trong lúc chờ (đó là thời gian "agent đang làm việc").
- Nếu model lỗi quota/key: UI hiện lỗi đã dịch (`LLM_RATE_LIMIT`…) — vẫn là một điểm cộng để nhắc tới.
- Đừng cố show hết. Ba cảnh **grounded answer → từ chối/insufficient → brief** là đủ thuyết phục.

---

## Mẫu email nộp bài

**Tiêu đề:** Re: AI Demo Challenge — Senior AI Developer — Bài nộp Option 2 (Company Brain Assistant)

Chào chị Thủy,

Em xin gửi bài AI Demo Challenge, đề tài **Option 2 — Company Brain Assistant**.

- **GitHub repository:** `<link repo>`
- **Loom (≈ 4–5 phút):** `<link loom>`
- **README** (kiến trúc, quyết định thiết kế, cách chạy, hạn chế): trong repo, file `README.md`.

**Tóm tắt giải pháp.** Một agent xây dựng trên Google ADK + Gemini, trả lời câu hỏi nội bộ của một brand skincare DTC
(sample data: product, brand guideline, compliance, SOP, research, creative learnings) trên Qdrant với hybrid search
(dense + BM25). Điểm em tập trung là độ tin cậy: mọi claim đều có citation và được kiểm chứng bằng code là đã
retrieve; compliance quảng cáo được enforce bằng policy-as-code ở 3 lớp; hệ thống nói rõ khi knowledge base không đủ
thông tin; đầu ra hữu ích là creative brief theo SOP (được validate trước khi lưu). Frontend React nhận token từ agent
qua WebSocket, hỗ trợ giao diện EN/VI trong khi knowledge base giữ tiếng Anh.

**Chạy thử nhanh:** `cp .env.example .env` (điền `GOOGLE_API_KEY`) → `make qdrant ingest backend frontend`,
hoặc `docker compose up --build`.

Em có ghi rõ các hạn chế đã biết và những gì em sẽ cải thiện cho bản production trong README. Rất mong nhận được
góp ý của team.

Trân trọng,
Phát
