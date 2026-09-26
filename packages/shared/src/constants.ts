export const PRODUCT_NAME = 'Forensics Web Lab';
export const RESEARCH_TITLE =
  'Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web';

export const MAX_FILE_SIZE_BYTES = 35 * 1024 * 1024; // 35 MB
export const MAX_PIXEL_COUNT = 64_000_000; // 64 Megapixels (e.g. 8000x8000)
export const MAX_CANVAS_PROCESSING_DIM = 2048; // Bounded internal canvas dimension

export const SUPPORTED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;

export const MAGIC_BYTES = {
  JPEG: [0xff, 0xd8, 0xff],
  PNG: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
  WEBP_RIFF: [0x52, 0x49, 0x46, 0x46], // 'RIFF'
  WEBP_TAG: [0x57, 0x45, 0x42, 0x50], // 'WEBP' at offset 8
} as const;

export const MODEL_CONFIG = {
  PATCH_SIZE: 224,
  PATCH_STRIDE: 112, // 50% overlap
  MAX_PATCHES: 48, // Adaptive clamp to ensure UI thread fluidity
  NORM_MEAN: [0.485, 0.456, 0.406] as const,
  NORM_STD: [0.229, 0.224, 0.225] as const,
} as const;

export const CALIBRATION_THRESHOLDS = {
  UNCERTAIN_CONFIDENCE_THRESHOLD: 0.65,
  HIGH_CONFIDENCE_THRESHOLD: 0.85,
  MARGIN_CONFLICT_THRESHOLD: 0.15,
  PATCH_SUSPICIOUS_THRESHOLD: 0.70,
  LOCALIZED_AREA_MIN_RATIO: 0.02,
  LOCALIZED_AREA_MAX_RATIO: 0.50,
} as const;

export const STANDARD_LIMITATIONS = [
  'Hệ thống là công cụ hỗ trợ điều tra kỹ thuật số, không phải công cụ xác minh pháp lý tuyệt đối.',
  'Trạng thái no_ai_evidence chỉ biểu thị không phát hiện thấy dấu vết AI trong phạm vi thuật toán, không phải chứng chỉ ảnh thật 100%.',
  'Ảnh bị nén nhiều lần qua mạng xã hội (JPEG Q < 60) hoặc downscale kích thước nhỏ có thể làm mất dấu vết tần số của mô hình tạo sinh.',
  'Một số kỹ thuật nhiếp ảnh truyền thống (HDR đa phơi sáng, khử nhiễu nặng, tone-mapping nghệ thuật) có thể gây sai lệch nhẹ.',
  'Các dòng mô hình AI mới chưa có trong tập dữ liệu huấn luyện có thể đòi hỏi cập nhật định kỳ để duy trì độ nhạy tối ưu.',
] as const;
