# Synthetic Smoke Fixture (`ml/tests/fixtures/`)

> **Quy định Nghiêm ngặt về Dữ liệu Giả lập (Synthetic Smoke Fixture)**:  
> 1. **KHÔNG PHẢI DỮ LIỆU ẢNH THẬT**: Toàn bộ mẫu tại đây là các hình vẽ hình học (hình vuông, hình tròn), dải chuyển màu (gradients) hoặc nhiễu toán học (Gaussian noise) được sinh hoàn toàn bằng mã nguồn dự án.  
> 2. **KHÔNG DÙNG ĐỂ ĐO ĐỘ CHÍNH XÁC (ACCURACY)**: Tuyệt đối không dùng dữ liệu này để tính accuracy, F1, AUROC hay ECE.  
> 3. **KHÔNG DÙNG ĐỂ CHỨNG MINH NĂNG LỰC AI**: Không chứng minh hay đại diện cho khả năng phát hiện hay định vị ảnh AI thật.  
> 4. **KHÔNG GỌI LÀ TRAINING DATASET**: Đây chỉ là fixture kiểm thử luồng kỹ thuật (code execution smoke test).  
> 5. **KHÔNG TẠO SỐ LIỆU KHOA HỌC**: Bất kỳ metric nào sinh ra từ fixture này đều bị cấm công bố.  
> 6. **CẤM ĐƯA CHECKPOINT VÀO `models/`**: Checkpoint (nếu có sinh tạm thời trong test) tuyệt đối không được đưa vào `models/registry.json` hoặc production bundle.  
> 7. **Ưu tiên sinh động (dynamic generation)**: Fixture được sinh trực tiếp trong quá trình chạy test hoặc trong thư mục tạm (`tempfile.TemporaryDirectory`) để tránh commit binary vào Git.
