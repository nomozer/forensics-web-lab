/// <reference types="vite/client" />
import React, { useState, useRef } from 'react';
import { AnalysisProgressEvent, ForensicSignal, ProvenanceResult } from '@forensics/shared';
import { WorkerController } from '@forensics/inference';
import { analyzeForensicSignals } from '@forensics/forensics';
import { analyzeProvenance } from '@forensics/provenance';
import { Header } from './components/Header.js';
import { ImageDropzone, ValidatedImageData } from './components/ImageDropzone.js';
import { AnalysisProgress } from './components/AnalysisProgress.js';
import { BinaryVerdictCard, BinaryInferenceData } from './components/BinaryVerdictCard.js';
import { ForensicsInspector } from './components/ForensicsInspector.js';
import { ProvenanceInspector } from './components/ProvenanceInspector.js';

export const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<ValidatedImageData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressEvent, setProgressEvent] = useState<AnalysisProgressEvent | null>(null);
  const [binaryResult, setBinaryResult] = useState<BinaryInferenceData | null>(null);
  const [forensicSignals, setForensicSignals] = useState<ForensicSignal[] | null>(null);
  const [provenanceResult, setProvenanceResult] = useState<ProvenanceResult | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'verdict' | 'dsp' | 'provenance'>('verdict');

  const workerControllerRef = useRef<WorkerController | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const getWorkerController = () => {
    if (!workerControllerRef.current) {
      workerControllerRef.current = new WorkerController();
    }
    return workerControllerRef.current;
  };

  const handleImageSelected = async (data: ValidatedImageData) => {
    setSelectedImage(data);
    setBinaryResult(null);
    setForensicSignals(null);
    setProvenanceResult(null);
    setGlobalError(null);
    setIsProcessing(true);
    setActiveTab('verdict');

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      setProgressEvent({
        stage: 'validating',
        progress: 0.1,
        message: 'Đang kiểm tra chữ ký nhị phân và khởi tạo môi trường...',
      });

      // 1. Run exploratory DSP and Provenance in parallel on client thread
      try {
        const prov = await analyzeProvenance(data.rawBuffer);
        setProvenanceResult(prov);
      } catch (err) {
        console.warn('Exploratory provenance extraction note:', err);
      }

      try {
        const forensicSummary = analyzeForensicSignals({
          rgba: data.rgba,
          width: data.width,
          height: data.height,
        });
        setForensicSignals(forensicSummary.signals);
      } catch (err) {
        console.warn('Exploratory DSP extraction note:', err);
      }

      // 2. Run verified live FP32 binary forensic pipeline inside Web Worker
      const controller = getWorkerController();
      const result: BinaryInferenceData = await controller.analyzeResearchImage(
        {
          rgba: data.rgba,
          width: data.width,
          height: data.height,
          basePath: ((import.meta as any).env?.BASE_URL as string) || '/forensics-web-lab/',
        },
        {
          onProgress: (event) => setProgressEvent(event),
          signal: abortController.signal,
        }
      );

      setBinaryResult(result);
    } catch (err: unknown) {
      if (err instanceof Error && err.message.includes('hủy bỏ')) {
        setProgressEvent({
          stage: 'cancelled',
          progress: 0,
          message: 'Quá trình phân tích đã bị hủy.',
        });
      } else {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setGlobalError(`Lỗi vận hành mô hình ONNX: ${errorMsg}`);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleClearSession = () => {
    if (selectedImage?.previewUrl) {
      URL.revokeObjectURL(selectedImage.previewUrl);
    }
    setSelectedImage(null);
    setBinaryResult(null);
    setForensicSignals(null);
    setProvenanceResult(null);
    setProgressEvent(null);
    setGlobalError(null);
    setIsProcessing(false);
  };

  return (
    <div className="app-container">
      <Header />

      {/* Mandatory Scientific Limitations & Forensic Boundaries Box */}
      <div
        id="scientific-limitations-banner"
        style={{
          maxWidth: '1200px',
          margin: '0 auto 24px auto',
          padding: '16px 20px',
          borderRadius: '10px',
          background: 'rgba(30, 41, 59, 0.7)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          fontSize: '0.8125rem',
          lineHeight: '1.6',
          color: '#cbd5e1',
          backdropFilter: 'blur(8px)',
        }}
      >
        <div style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '6px', color: '#60a5fa' }}>
          📌 Phạm Vi Nghiên Cứu Khoa Học & Giới Hạn Pháp Chứng Bắt Buộc:
        </div>
        <ul style={{ margin: 0, paddingLeft: '20px' }}>
          <li>
            <strong>Mục đích nghiên cứu:</strong> Hệ thống phục vụ minh chứng thực nghiệm học thuật (research demonstration artifact) cho đề tài và công bố khoa học; vận hành hoàn toàn client-side (Zero-Egress).
          </li>
          <li>
            <strong>Mô hình nhị phân:</strong> Pipeline trực tiếp vận hành mô hình 2 lớp (<code>authentic</code> vs <code>ai_edited</code>) kết hợp 5 outer-fold checkpoints (MobileNetV3 FP32 backbone + 16 đặc trưng DSP chuẩn tắc + Stacker).
          </li>
          <li>
            <strong>Ý nghĩa xác suất:</strong> Xác suất P(ai_edited) thể hiện độ tương thích với phân phối dữ liệu nghiên cứu; <em>không phải độ chính xác tổng quát của hệ thống</em>. Quyết định <code>AUTHENTIC</code> không chứng minh ảnh thật hay nguyên bản tuyệt đối.
          </li>
          <li>
            <strong>Giới hạn kỹ thuật:</strong> Ảnh AI tạo sinh toàn phần (<code>fully_generated</code>) nằm ngoài phạm vi đã kiểm chứng của detector hiện tại; hệ thống chưa tuyên bố hỗ trợ tạo heatmap định vị (<code>localization</code>).
          </li>
        </ul>
      </div>

      {/* Main Interaction Area */}
      {!selectedImage && !isProcessing && (
        <ImageDropzone onImageSelected={handleImageSelected} isProcessing={isProcessing} />
      )}

      {/* Processing State */}
      {isProcessing && progressEvent && (
        <AnalysisProgress progressEvent={progressEvent} onCancel={handleCancel} />
      )}

      {/* Error State */}
      {globalError && (
        <div
          className="glass-panel"
          style={{
            maxWidth: '1200px',
            margin: '0 auto 24px auto',
            padding: '20px',
            borderLeft: '4px solid var(--accent-rose, #f43f5e)',
            background: 'rgba(244, 63, 94, 0.12)',
            color: '#fca5a5',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <strong style={{ fontSize: '1rem', color: '#ff859b' }}>⚠️ Không thể thực thi suy luận mô hình:</strong>
              <p style={{ margin: '8px 0 0 0', fontSize: '0.875rem' }}>{globalError}</p>
              <p style={{ margin: '4px 0 0 0', fontSize: '0.75rem', color: '#e2e8f0' }}>
                Tuân thủ quy tắc trung thực khoa học: Khi mô hình không tải được hoặc xảy ra lỗi môi trường, hệ thống tuyệt đối không đưa ra kết luận phân loại giả mạo.
              </p>
            </div>
            <button
              onClick={handleClearSession}
              className="action-button secondary"
              style={{ padding: '8px 16px', fontSize: '0.8rem', cursor: 'pointer' }}
            >
              🔄 Thử Lại
            </button>
          </div>
        </div>
      )}

      {/* Results Section */}
      {binaryResult && selectedImage && (
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {/* 1. Primary Binary Verdict & 5-Fold Ensembles Card */}
          <BinaryVerdictCard
            data={binaryResult}
            filename={selectedImage.file.name}
            fileSizeBytes={selectedImage.file.size}
            sha256={selectedImage.sha256}
            width={selectedImage.width}
            height={selectedImage.height}
            previewUrl={selectedImage.previewUrl}
            onNewImage={handleClearSession}
          />

          {/* 2. Exploratory Forensic & Metadata Tools Section */}
          <div
            className="glass-panel"
            style={{
              padding: '20px',
              marginBottom: '24px',
              borderTop: '2px solid rgba(59, 130, 246, 0.4)',
            }}
          >
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
                    🔬 Công Cụ Phân Tích Khám Phá Tín Hiệu Số & Siêu Dữ Liệu
                  </h3>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
                    Các chỉ số dưới đây mang tính khám phá bổ trợ (Exploratory Forensics), hoạt động độc lập và không thay thế cho quyết định phân loại của mô hình nơ-ron học sâu.
                  </p>
                </div>

                {/* Sub-tab navigation */}
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    id="subtab-dsp"
                    onClick={() => setActiveTab('dsp')}
                    style={{
                      padding: '6px 14px',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      borderRadius: '6px',
                      border: activeTab === 'dsp' ? '1px solid #38bdf8' : '1px solid rgba(255,255,255,0.1)',
                      background: activeTab === 'dsp' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255,255,255,0.03)',
                      color: activeTab === 'dsp' ? '#38bdf8' : '#94a3b8',
                      cursor: 'pointer',
                    }}
                  >
                    📊 Tín Hiệu DSP Tần Số & Nhiễu
                  </button>

                  <button
                    id="subtab-provenance"
                    onClick={() => setActiveTab('provenance')}
                    style={{
                      padding: '6px 14px',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      borderRadius: '6px',
                      border: activeTab === 'provenance' ? '1px solid #c084fc' : '1px solid rgba(255,255,255,0.1)',
                      background: activeTab === 'provenance' ? 'rgba(192, 132, 252, 0.15)' : 'rgba(255,255,255,0.03)',
                      color: activeTab === 'provenance' ? '#c084fc' : '#94a3b8',
                      cursor: 'pointer',
                    }}
                  >
                    📋 Siêu Dữ Liệu & Nguồn Gốc (EXIF/C2PA)
                  </button>
                </div>
              </div>
            </div>

            {/* Exploratory View Content */}
            {activeTab === 'dsp' && forensicSignals && (
              <ForensicsInspector signals={forensicSignals} />
            )}

            {activeTab === 'provenance' && provenanceResult && (
              <ProvenanceInspector provenance={provenanceResult} />
            )}

            {activeTab === 'verdict' && (
              <div className="grid-2col" style={{ gap: '20px', marginTop: '16px' }}>
                {forensicSignals && <ForensicsInspector signals={forensicSignals} />}
                {provenanceResult && <ProvenanceInspector provenance={provenanceResult} />}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
