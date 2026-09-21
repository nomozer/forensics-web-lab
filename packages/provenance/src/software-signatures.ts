export interface AiSignatureRule {
  pattern: RegExp;
  generatorName: string;
  confidence: number;
  description: string;
}

export const KNOWN_AI_SIGNATURES: AiSignatureRule[] = [
  {
    pattern: /stable[\s_-]?diffusion/i,
    generatorName: 'Stable Diffusion',
    confidence: 0.95,
    description: 'Tìm thấy chữ ký phần mềm Stable Diffusion trong metadata',
  },
  {
    pattern: /midjourney/i,
    generatorName: 'Midjourney',
    confidence: 0.95,
    description: 'Tìm thấy chữ ký phần mềm Midjourney trong metadata',
  },
  {
    pattern: /dall[-_]?e/i,
    generatorName: 'DALL-E',
    confidence: 0.95,
    description: 'Tìm thấy chữ ký mô hình DALL-E trong metadata',
  },
  {
    pattern: /novelai/i,
    generatorName: 'NovelAI',
    confidence: 0.95,
    description: 'Tìm thấy chữ ký NovelAI sinh ảnh anime/nghệ thuật trong metadata',
  },
  {
    pattern: /comfyui/i,
    generatorName: 'ComfyUI',
    confidence: 0.95,
    description: 'Tìm thấy workflow đồ họa sinh ảnh ComfyUI trong metadata',
  },
  {
    pattern: /automatic1111|webui/i,
    generatorName: 'A1111 WebUI',
    confidence: 0.9,
    description: 'Tìm thấy metadata giao diện sinh ảnh Automatic1111',
  },
  {
    pattern: /adobe[\s_-]?firefly/i,
    generatorName: 'Adobe Firefly',
    confidence: 0.95,
    description: 'Tìm thấy chữ ký Adobe Firefly generative tool',
  },
  {
    pattern: /photoshop[\s_-]?(generative|firefly)/i,
    generatorName: 'Photoshop Generative Fill',
    confidence: 0.95,
    description: 'Tìm thấy dấu vết Photoshop Generative Fill (AI inpainting)',
  },
];

export const TRADITIONAL_SOFTWARE_PATTERNS = [
  /adobe photoshop/i,
  /adobe lightroom/i,
  /gimp/i,
  /affinity photo/i,
  /snapseed/i,
];
