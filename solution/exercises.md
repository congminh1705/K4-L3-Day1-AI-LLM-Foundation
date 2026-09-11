# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Cả bốn mức temperature đều cho phản hồi về hang Sơn Đoòng, nhưng khác nhau về cách diễn đạt, thứ tự trình bày và các chi tiết bổ sung. Riêng mức 1.5 còn thêm thông tin về hạt điều, cà phê và bánh mì, khiến nội dung mở rộng ngoài sự thật chính được hỏi. Tuy nhiên, kết quả chưa cho thấy mức độ đa dạng tăng đều theo temperature; cần chạy lặp lại mỗi mức để đánh giá rõ hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Em chọn temperature = 0.2 làm mức khởi đầu cho chatbot hỗ trợ khách hàng, nhằm giảm tính ngẫu nhiên và ưu tiên câu trả lời nhất quán. Chatbot cần diễn đạt rõ ràng, bám sát chính sách và hướng dẫn của doanh nghiệp. Tuy nhiên, temperature thấp không bảo đảm thông tin chính xác, nên chatbot vẫn cần dữ liệu đáng tin cậy và chuyển cho nhân viên khi không đủ thông tin.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá trong lab, chi phí được tính như sau: Tổng số lượt gọi mỗi ngày: 10.000 × 3 = 30.000 lượt. Tổng token đầu ra mỗi ngày: 30.000 × 350 = 10.500.000 token. GPT-4o có giá 0,010 USD/1.000 token đầu ra, nên chi phí là: 10.500.000 ÷ 1.000 × 0,010 = 105 USD/ngày. GPT-4o-mini có giá 0,0006 USD/1.000 token đầu ra, nên chi phí là: 10.500.000 ÷ 1.000 × 0,0006 = 6,30 USD/ngày. Tỷ lệ chi phí: 105 ÷ 6,30 ≈ 16,67 lần. Như vậy, chi phí đầu ra của GPT-4o gấp khoảng 16,67 lần GPT-4o-mini với workload này. Phép tính chưa bao gồm chi phí token đầu vào vì đề không cung cấp số lượng. GPT-4o có thể xứng đáng với chi phí khi xử lý yêu cầu khách hàng phức tạp, cần đối chiếu nhiều chính sách và thử nghiệm cho thấy chất lượng tốt hơn rõ rệt. Với các câu hỏi thường gặp như giờ mở cửa hoặc hướng dẫn đặt hàng, em sẽ chọn GPT-4o-mini nếu chất lượng đáp ứng yêu cầu, nhằm tiết kiệm chi phí.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *CVới system prompt dành cho giáo viên tiểu học, câu trả lời dài 1.793 ký tự, sử dụng ngôn ngữ đơn giản và ví dụ gần gũi như các bạn nhỏ cùng ghi chép vào một cuốn sổ. Với system prompt dành cho chuyên gia tài chính, câu trả lời dài 6.957 ký tự, sử dụng nhiều thuật ngữ kỹ thuật như DLT, hàm băm, Merkle Tree, Proof of Work và Proof of Stake. Như vậy, system prompt ảnh hưởng rõ rệt đến độ dài, từ vựng, mức độ chuyên sâu và cách đưa ra ví dụ của model. Cùng một câu hỏi nhưng persona khác nhau khiến model điều chỉnh cách giải thích cho phù hợp với đối tượng người đọc.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn của em có 100 từ. Số token đo bằng tiktoken là 119 token, trong khi cách ước lượng của Part 1 cho kết quả (100 / 0,75 = 133,33) token. Chênh lệch tuyệt đối là (|119 - 133,33| = 14,33) token, tương đương ((14,33 / 133,33) \times 100 \approx 10,75\%); số token thực tế thấp hơn số ước lượng. Nguyên nhân là tokenizer không xem mỗi từ là một token mà chia văn bản thành các token nhỏ hơn dựa trên cách mã hóa và dữ liệu huấn luyện. Tiếng Việt có dấu, cách tách âm tiết bằng khoảng trắng và các cụm từ ít xuất hiện trong dữ liệu huấn luyện tiếng Anh, nên đôi khi kém hiệu quả hơn; tuy nhiên mức chênh lệch còn phụ thuộc vào nội dung, tokenizer và model.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng khi câu trả lời dài hoặc người dùng cần phản hồi ngay, chẳng hạn như chatbot trò chuyện, trợ lý viết nội dung hoặc công cụ sinh mã. Model gửi từng phần kết quả ngay khi tạo ra, nên người dùng có thể bắt đầu đọc sớm thay vì phải chờ toàn bộ câu trả lời hoàn thành; điều này làm giảm thời gian chờ cảm nhận, dù tổng thời gian xử lý có thể không giảm. Non-streaming phù hợp với câu trả lời ngắn hoặc khi ứng dụng cần nhận toàn bộ kết quả để kiểm tra, phân tích JSON, lưu dữ liệu hoặc xử lý các bước tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff là cách tăng dần thời gian chờ sau mỗi lần gọi API bị lỗi, ví dụ lần đầu chờ 1 giây, lần sau chờ 2 giây và lần tiếp theo chờ 4 giây. Cách này giúp API có thời gian giảm tải và xử lý các request đang chờ. Nếu hàng nghìn client đều chờ đúng 1 giây rồi retry cùng lúc, chúng sẽ tạo thêm một lượng request rất lớn, khiến API càng dễ bị quá tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona là một trợ giảng thân thiện dành cho người mới học AI. System prompt: “Bạn là trợ giảng thân thiện của khóa học AI dành cho người mới. Hãy trả lời bằng tiếng Việt, giải thích ngắn gọn và dễ hiểu, đồng thời đưa ra ví dụ đơn giản khi cần. Nếu không chắc chắn, hãy nói rõ thay vì đoán.” Em dùng cụm “dành cho người mới” để model không giải thích quá chuyên sâu. Em yêu cầu trả lời bằng tiếng Việt và ngắn gọn để câu trả lời phù hợp với trình độ của người học và dễ theo dõi hơn.


### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý là chỉ nhớ được một số lượt trò chuyện gần nhất. Nếu cuộc hội thoại quá dài hoặc người dùng khởi động lại chương trình, trợ lý có thể quên những thông tin đã nói trước đó. Một cải thiện cụ thể là lưu lịch sử trò chuyện vào một tệp JSON và đọc lại tệp này khi chương trình khởi động. Khi gửi câu hỏi mới, chương trình có thể dùng phần tóm tắt của các cuộc trò chuyện cũ cùng với những tin nhắn gần đây để trợ lý vẫn hiểu được ngữ cảnh.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
