# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV: 2A202603858**
- **Họ và tên: Đoàn Xuân Duy Bảo**
- **SEED** (in ở ô đầu notebook): 3544

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra tự động: **15/15 đạt** (10 TODO code + 5 phép tính tay). Đủ 10/10 dự đoán `predict(...)`.

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (SEED 3544). Dữ liệu sau khi bỏ trùng: 5.169 tin, train 3.619 / val 775 / test 775. Các hàng 1–3 đo trên toàn bộ test (775 tin); hàng 4 (LLM) chỉ chạy trên SUB (40 tin cân bằng), nên cột "F1 test" của hàng 4 thực chất là F1 trên SUB.

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.876 | 0.919 | 0.11 | 0 |
| 2. BoW + MLP | 0.939 | 0.947 | 2.43 | 0 |
| 3a. Encoder MiniLM + kNN | 0.911 | 0.947 | 93.79 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.949 | 0.919 | 57.33 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.949 | 0.919 | 46.67 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | 0.900 (SUB) | 0.900 | 2243.65 | 0.0074 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | 0.947 (SUB) | 0.947 | 1663.80 | 0.0738 |
| 4c. LLM nhỏ few-shot | 0.900 (SUB) | 0.900 | 1347.34 | 0.0214 |

Ghi chú:
- Chênh F1 < 0.05 trên SUB coi như hoà (1 tin sai đổi F1 khoảng 2.5–2.7 điểm %).
- Chỉ 3 cách đạt SLA p95 < 50 ms: 1, 2, 3c. Cách 3b (57 ms) và 3a (94 ms) không đạt.
- Đường Pareto (1 − F1 SUB theo p95): BoW + LogReg và BoW + MLP.

## 3. Dự đoán và thực tế

**(1) `tradeoff` — Pareto và SLA.**
- Đoán: Pareto gồm BoW + MLP và Encoder + LogReg (3b); chỉ 3 cách LLM vượt SLA 50 ms, các cách còn lại đều dưới 50 ms.
- Ra: Pareto là BoW + LogReg và BoW + MLP, không có 3b. Vượt SLA có 5 cách: 3a (94 ms), 3b (57 ms) và 3 cách LLM.
- Vì sao lệch: tôi đánh giá thấp chi phí chạy MiniLM khi gọi từng tin trên CPU (p50 38 ms, p95 57 ms cho 3b). F1 SUB của 3b (0.919) chỉ bằng BoW + LogReg nhưng chậm hơn ~500 lần, và thua BoW + MLP (0.947) cả về tốc độ lẫn chất lượng nên bị lấn át. Throughput theo lô (607 tin/s) làm tôi tưởng nó đủ nhanh, trong khi SLA tính theo từng request.

**(2) `dl_relu` (và `dl_curve`) — bỏ ReLU.**
- Đoán: bỏ ReLU thì mạng tương đương logistic regression nên F1 tương đương LogReg; MLP có ReLU cũng xấp xỉ LogReg.
- Ra: không ReLU F1 = 0.939, bằng MLP có ReLU (0.939) và cao hơn hẳn LogReg Phần 1 (0.876, recall 0.786 so với 0.903).
- Vì sao lệch: về mặt mô hình, hai lớp Linear liên tiếp vẫn là một hàm tuyến tính, nên chênh lệch không đến từ việc ReLU hay deep learning "học đặc trưng". Khác biệt nằm ở cách huấn luyện: Adam + mini-batch 64 + 30 epoch so với gradient descent toàn batch lr 0.5, 300 epoch, khởi tạo w = 0 ở Phần 1 (train recall mới 0.844). Tôi nhầm "cùng họ mô hình" với "cùng kết quả".

**(3) `hard_eval` — độ chính xác trên 50 case khó.**
- Đoán: BoW ~50%, Encoder 65–75%, LLM nhỏ 75–80%, LLM lớn 85–90%; nhóm khó nhất với LLM là prompt injection.
- Ra: BoW 52%, Encoder 66–72% (đúng dự đoán), nhưng LLM nhỏ zero-shot **98%** và nhóm prompt injection đạt 100%. Lỗi duy nhất của LLM nằm ở nhóm "Tưởng spam nhưng không phải" (90%).
- Vì sao lệch: tôi đánh giá thấp khả năng đọc ý định của LLM và đánh giá quá cao mức độ bị câu "Ignore all previous instructions…" đánh lừa. Với BoW, nhóm khó nhất không chỉ là "né từ khoá" (17%) mà còn "spam giọng tự nhiên" (10%) và "thao túng / lừa đảo" (12%).

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất: BoW + LogReg**, F1 từ 0.919 (SUB) xuống **0.333** trên 50 case khó (giảm 0.59); BoW + MLP cũng tụt tương tự (0.947 → 0.400). Encoder tụt vừa (3a 0.947 → 0.759; 3b 0.919 → 0.667; 3c 0.919 → 0.680). LLM không tụt (0.900 → 0.983). Độ chính xác theo nhóm của BoW + LogReg:

| Nhóm | BoW + LogReg đúng |
| --- | --- |
| Spam giọng tự nhiên | 10% |
| Thao túng / lừa đảo | 12% |
| Né từ khoá | 17% |
| Tin giao dịch hợp lệ | 83% |
| Tưởng spam nhưng không phải, Nói về spam | 100% |

Hai nhóm ham đạt 100% vì BoW gần như luôn đoán ham với tin ít từ khoá spam. Tổng 52% là con số "đẹp giả". Test thường lạc quan vì cùng phân phối với train: chỉ cần học mẹo từ khoá là điểm cao.

**Case 1 — #29 (né từ khoá, spam): `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m`.**
Cơ chế: `tokenize` chỉ tách theo `[a-z0-9£$€]+`, nên "c0ngr4tulat10ns", "w0n", "cl4im" thành token lạ không có trong vocab 2.000 từ; BoW chỉ còn thấy vài token trung tính, gần như "mù". Encoder xử lý tốt hơn (nhóm này 83%) vì tách từ con và vẫn thấy dạng gần giống "congratulations/won/claim".

**Case 2 — #47 (nói về spam, ham): `lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?`.**
Cơ chế: tin chứa đủ từ khoá spam nhưng ý định là chê bai. Embedding chỉ nắm chủ đề và văn phong, nên vector nằm gần cụm spam "won £1000 / call". Vì vậy nhóm này chỉ đạt 25% với kNN, 50% với LogReg và MLP head, trong khi LLM đọc được ý định "kể lại để chê" và đúng 100%. (Cũng nhóm này, BoW đúng 100% nhưng vì đoán ham tràn lan, không phải vì hiểu.)

Với bộ 8 case tự viết thêm (id 100–107), bảng trên chưa gồm chúng vì ô stress test chạy trước khi tôi thêm.

## 5. Thử thách cá nhân

**Đề (SEED 3544 → đề #5, Embedding):** viết 4 tin nhắn riêng (2 spam né từ khoá, 2 ham chứa "free"/"win"), so dự đoán BoW và Embedding + LogReg, giải thích từng tin.

**Cách làm:** tự viết 4 tin, tính xác suất bằng `predict_proba` với vocab BoW (`vectorize`) và với embedding MiniLM đã chuẩn hoá (`(encode − mu) / sd`) rồi so ngưỡng 0.5; thêm cột LLM nhỏ để đối chiếu.

**Kết quả:** BoW đúng 2/4, Embedding + LogReg đúng 3/4, LLM đúng 4/4.

| Tin | Thật | BoW P | Emb P |
| --- | --- | --- | --- |
| "Your loyalty points are about to lapse, redeem them today at rewards-hub.co before midnight" | spam | 0.20 ✗ | 0.85 ✓ |
| "I'm new in the area and looking for friends. Message me on my private page: meet-new-people.net" | spam | 0.01 ✗ | 0.38 ✗ |
| "I'm free after 6, want to grab dinner? If we win the quiz tonight you're buying" | ham | 0.06 ✓ | 0.23 ✓ |
| "Are you free tomorrow? Our team might win the league if we beat them on Saturday" | ham | 0.12 ✓ | 0.00 ✓ |

**Giải thích:**
- Tin 1: BoW có 13 từ trong vocab nhưng toàn từ thường gặp, không có từ spam mạnh như free/claim/prize nên P chỉ 0.20. Embedding hiểu ý "điểm thưởng sắp hết hạn + đường link lạ" nên cho 0.85.
- Tin 2: cả hai sai. BoW thấy toàn từ đời thường như tin ham (P 0.01). Embedding thấy giọng thân mật nên chỉ cho 0.38. Chỉ LLM bắt được ý định dụ vào trang lạ.
- Tin 3 và 4: cả hai đúng. "free" và "win" chỉ là 1–2 từ giữa nhiều từ đời thường nên tổng trọng số BoW vẫn nghiêng về ham; embedding đọc ra chuyện hẹn đi ăn / bóng đá (tin 4 cho P = 0.00).
- Kết luận: BoW không bị "free/win" đánh lừa như tôi lo, điểm yếu thật của nó là spam viết bằng từ đời thường.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin, 10 triệu SMS/ngày (≈ 116 tin/giây trung bình).

**Đạt độ trễ:** chỉ BoW + LogReg (0.11 ms), BoW + MLP (2.43 ms) và Encoder + MLP head 3c (46.67 ms, sát ngưỡng). 3b (57 ms), kNN (94 ms) và mọi cách LLM (1.3–2.2 giây) không đạt. LLM không thể nằm trên đường xử lý trực tiếp.

**Chọn:** BoW + MLP làm lớp chính cho mọi tin. Số liệu: F1 test 0.939, F1 SUB 0.947 (hoà với 3b/3c, 0.949/0.919 vì chênh < 0.05), p95 2.43 ms, ~17.5k tin/s trên một tiến trình, chi phí inference 0 USD, train 14 s. Chọn nó thay vì 3c vì 3c chỉ có ~3 ms dư địa trên CPU Colab: sang máy khác chậm hơn một chút là vỡ SLA.

**Lai (router):** tin có p ∈ [0.3, 0.7] gửi tiếp qua hàng đợi bất đồng bộ cho 3c, rồi cho LLM (không chặn việc giao tin). Lý do cần lớp thứ hai: trên 50 case khó BoW + MLP chỉ đúng 52% (F1 0.400), trong khi LLM đúng 98–100%.

**Chi phí LLM nếu dùng cho toàn bộ 10 triệu tin/ngày (ngoại suy từ USD/1k tin đo được):** model nhỏ ≈ 74 USD/ngày (~2.2k USD/tháng), model lớn ≈ 738 USD/ngày (~22k USD/tháng). Nếu router chỉ đẩy 10% tin sang LLM nhỏ thì ≈ 7.4 USD/ngày (tỉ lệ 10% là giả định, chưa đo).

**Đánh đổi chấp nhận:**
- Lớp nhanh vẫn bỏ sót spam giọng tự nhiên, thao túng, né từ khoá; chỉ bắt được phần nằm trong vùng lưỡng lự.
- Độ trễ cao hơn và tốn tiền cho tin đi qua LLM.
- LLM có rủi ro prompt injection, nên không dùng làm lớp kiểm tra duy nhất; cần chuẩn hoá đầu vào và một mô hình không đọc lệnh làm lớp song song.
- Số liệu LLM chỉ dựa trên 40 tin, nhiễu lớn; cần đo lại trên tập lớn hơn trước khi chốt.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì | AI trả lời / viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Các ô dự đoán `predict(...)` | Đưa code và kết quả các ô, hỏi dự đoán và cách nghĩ ra dự đoán đó | Đưa ra dự đoán kèm phân tích, thường dài dòng, không nêu ý chính | Viết lại thành các ý chính ngắn, tự chỉnh từ ngữ và ký hiệu |
| Điền báo cáo `.md` | Đọc notebook và `lab1_report.json`, điền các mục còn trống | Tổng hợp bảng kết quả, so dự đoán với thực tế, soạn nháp từng mục | Cần tôi đối chiếu từng số với output notebook, kiểm tra lại các case ở mục 4 (notebook không in đúng từng case), và sửa lại lời giải thích bằng cách nói của tôi |

**Điều đã học:**
- Điểm cao trên test thường chưa chứng minh model "hiểu": BoW đạt F1 0.92 trên SUB nhưng chỉ 0.33 trên 50 case khó, vì test cùng phân phối với train. Phải đo bằng dữ liệu, không bằng trực giác (tôi từng lo BoW bị "free/win" đánh lừa, trong khi điểm yếu thật lại là spam viết bằng từ đời thường).
- Độ trễ cần đọc theo từng request (p95), không theo throughput theo lô. 3b nhanh 607 tin/s theo lô nhưng p95 vẫn 57 ms nên rớt SLA.
- Số tham số hay "độ sâu" không quyết định kết quả: MLP bỏ ReLU vẫn đạt 0.939. Cách tối ưu hoá (Adam, mini-batch, số epoch) có thể tạo chênh lệch lớn hơn kiến trúc.
