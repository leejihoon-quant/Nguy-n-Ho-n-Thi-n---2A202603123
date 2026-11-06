# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về cả bức ảnh. Vì (1) record không có thông tin vị trí (không có bbox), chỉ có một class_name cho toàn ảnh; và (2) top-5 là các phỏng đoán xếp hạng cho cùng một ảnh, không phải các vật khác nhau.
3. Danh sách tên lớp đi kèm taxonomy và checkpoint. Dẫn chứng: record có taxonomy_name = ImageNet-1K và model_file = yolo11n-cls.pt; notebook nói taxonomy là danh sách lớp cố định đi kèm checkpoint.
4. class_id giúp máy phân biệt chính xác từng lớp, không nhầm giữa các tên giống hoặc khác nhau; class_name giúp người đọc hiểu ngay ý nghĩa mà không cần tra bảng; taxonomy_name cho biết class_id và class_name thuộc bộ lớp nào, vì cùng một số có thể là lớp khác ở bộ khác. Ví dụ, 468 chỉ có nghĩa là cab khi biết bộ lớp là ImageNet-1K.
5. Guideline cần quy định rõ: mỗi ảnh chỉ gán một lớp, chọn từ taxonomy ImageNet-1K. Khi ảnh có nhiều vật như ảnh traffic (có cả cab, minibus, police_van,...) người gán nhãn chọn theo vật lớn nhất, gần giữa nhất. Nếu có nhiều hơn 1 nhãn thỏa mãn, người gán nhãn báo cáo cho Lab Coach thay vì tự đoán.
6. Không được lấy score làm ground truth vì score chỉ là độ tự tin của model, chưa được con người kiểm chứng. Ví dụ cab có score 0.51, nghĩa là model chỉ khá chắc chứ không chắc chắn, và con số này không bảo đảm cab là nhãn đúng. Ground truth phải do người gán nhãn (annotator) gán theo guideline, rồi reviewer kiểm tra lại.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Hộp có góc trên-trái ở (385.33, 69.24): cách mép trái ảnh khoảng 385 pixel và cách mép trên khoảng 69 pixel. Góc dưới-phải ở (498.92, 348.92). Hộp rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel. Gốc (0, 0) ở góc trên-trái của ảnh, x tăng sang phải, y tăng xuống dưới. Hộp cao và hẹp, phù hợp với dáng một người đứng.
3. Ở ngưỡng 0.20 có 17 prediction; ở ngưỡng 0.60 có 6 prediction (ở 0.35 có 11).
4. Khi hạ ngưỡng, giữ lại nhiều hộp hơn, trong đó có nhiều hộp có score thấp và dễ sai (ở 0.20 xuất hiện thêm spoon, bottle, potted plant, dining table...). Reviewer phải kiểm tra nhiều hộp hơn: bỏ hộp sai hoặc thừa, sửa hộp lệch, và xem còn thiếu vật nào.
5. Quy tắc hộp: vẽ một hộp cho mỗi vật, ôm sát phần nhìn thấy của vật, bao trọn vật đó nhưng không lấy thêm nhiều nền. Quy tắc này áp dụng được cho mọi ảnh.
6. Trong ảnh kitchen, hộp person score 0.61 ở mép trái bị cắt bởi mép ảnh và bị bàn, bát che, chỉ thấy một phần nhỏ. Với trường hợp vật bị cắt mép hoặc che một phần, tôi sẽ viết thành quy tắc guideline (vẫn gán nhãn phần nhìn thấy, hộp sát phần nhìn thấy), vì tình huống này lặp lại ở nhiều ảnh. Nếu phần nhìn thấy quá ít nên không chắc là vật gì, thì escalation, hỏi Lab Coach thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. Hộp chỉ cho biết vị trí và kích thước tổng thể (hình chữ nhật). Polygon cho thêm hình dạng thật của vật: các điểm [x, y] bám theo biên của vật. Ví dụ ảnh kitchen (15_segmentation-visual_1.png), mask của person 0.90 ôm theo dáng người (đầu, thân, chân), còn hộp thì chứa cả nền xung quanh.
3. instance_id (dạng kitchen-001) là mã định danh cho từng đối tượng riêng trong output của lab, để phân biệt hai vật cùng lớp (ví dụ hai bowl vẫn có hai instance_id và hai polygon). Nó không phải class ID (không nói vật thuộc lớp nào) và không phải tracking ID (không theo dõi cùng một vật qua nhiều khung hình).
4. Quy tắc vẽ biên mask: đi theo biên ngoài của vật, bao đúng phần pixel của vật và không lấy nền; chỗ vật che khuất thì chỉ vẽ phần nhìn thấy.
5. Trên ảnh kitchen: dining table 0.58 có mask lớn chồng lên các bowl (0.70, 0.74) và oven 0.49, còn person 0.43 ở mép trái bị che và dính vào bát, nên biên không rõ. Với quy tắc chung (vẽ theo phần nhìn thấy) thì dùng guideline; còn ca không chắc đó là vật nào hoặc biên không phân biệt được thì escalation lên Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh (class_id, class_name, taxonomy_name) | Ảnh traffic có nhiều xe buýt và ô tô; top-1 cab chỉ 0.510915, minibus 0.164284, police_van 0.085848 nên khó chọn một nhãn | Chọn một lớp từ taxonomy theo tiêu chí guideline (vật lớn nhất, gần giữa nhất) | Nhãn có nằm trong taxonomy và đúng tiêu chí không; không dùng score làm căn cứ |
| Phát hiện vật thể | Một hộp bbox_xyxy (pixel) và một lớp cho mỗi vật | Ảnh kitchen: person 0.61 bị cắt ở mép ảnh; ngưỡng 0.20 cho 17 hộp nhưng 0.60 chỉ 6 hộp, nhiều hộp yếu (spoon, bottle) | Vẽ hộp ôm sát phần nhìn thấy của từng vật | Hộp có sát không, có thiếu hoặc thừa vật, có sai lớp không |
| Instance segmentation | Một polygon polygon_xy và một instance_id cho mỗi vật | Ảnh kitchen: dining table 0.58 chồng lên bowl 0.70 và 0.74; person 0.43 bị che; traffic-001 (bus) có 120 điểm polygon nên vẽ tay tốn công | Vẽ biên theo vật, tách riêng từng instance | Biên mask có khớp vật không, các instance chồng nhau có bị lẫn không |

Nguồn số liệu: output của notebook (classification, detection kèm so sánh ngưỡng 0.20 / 0.35 / 0.60 với 17 / 11 / 6 vật, segmentation) và các ảnh trong `day1_lab_outputs/visuals/`.

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai (traffic, kitchen, dining) do notebook tải về; không tải ảnh cá nhân hay dữ liệu nội bộ, và không ghi họ tên, MSSV, email hay số điện thoại vào báo cáo.

2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi dừng lại và báo Lab Coach (mentor) thay vì tự xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
