import React, { useState, useRef } from 'react';
import { AnalysisProgressEvent, AnalysisResult } from '@forensics/shared';
import { WorkerController } from '@forensics/inference';
import { Header } from './components/Header.js';
import { ImageDropzone, ValidatedImageData } from './components/ImageDropzone.js';
import { AnalysisProgress } from './components/AnalysisProgress.js';
import { ResultVerdictCard } from './components/ResultVerdictCard.js';
import { HeatmapViewer } from './components/HeatmapViewer.js';
import { EvidenceLedger } from './components/EvidenceLedger.js';
import { ForensicsInspector } from './components/ForensicsInspector.js';
import { ProvenanceInspector } from './components/ProvenanceInspector.js';
import { ReportActions } from './components/ReportActions.js';
import { ResearchLabView } from './components/ResearchLabView.js';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'standard' | 'research'>('standard');
  const [selectedImage, setSelectedImage] = useState<ValidatedImageData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressEvent, setProgressEvent] = useState<AnalysisProgressEvent | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const workerControllerRef = useRef<WorkerController | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleImageSelected = async (data: ValidatedImageData) => {
    setSelectedImage(data);
    setResult(null);
    setGlobalError(null);
    setIsProcessing(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      if (!workerControllerRef.current) {
        workerControllerRef.current = new WorkerController();
      }

      const analysisResult = await workerControllerRef.current.analyzeImage(
        {
          rgba: data.rgba,
          width: data.width,
          height: data.height,
          rawBuffer: data.rawBuffer,
          filename: data.file.name,
          sha256: data.sha256,
          mimeType: data.file.type || 'image/jpeg',
          fileSizeBytes: data.file.size,
        },
        {
          onProgress: (event) => setProgressEvent(event),
          signal: abortController.signal,
        }
      );

      setResult(analysisResult);
    } catch (err: unknown) {
      if (err instanceof Error && err.message.includes('hủy bỏ')) {
        setProgressEvent({
          stage: 'cancelled',
          progress: 0,
          message: 'Quá trình phân tích đã bị hủy.',
        });
      } else {
        setGlobalError(`Lỗi phân tích: ${String(err)}`);
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
    setResult(null);
    setProgressEvent(null);
    setGlobalError(null);
    setIsProcessing(false);
  };

  const handleNewImage = () => {
    handleClearSession();
  };

  return (
    <div className="app-container">
      <Header />

      <div
        style={{
          display: 'flex',
          gap: '12px',
          margin: '0 auto 24px auto',
          maxWidth: '1200px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          paddingBottom: '12px',
        }}
      >
        <button
          onClick={() => setActiveTab('standard')}
          style={{
            padding: '10px 20px',
            borderRadius: '8px',
            border: activeTab === 'standard' ? '1px solid var(--accent-cyan, #06b6d4)' : '1px solid transparent',
            background: activeTab === 'standard' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255, 255, 255, 0.05)',
            color: activeTab === 'standard' ? '#38bdf8' : '#94a3b8',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
        >
          🔬 Giám Định Tiêu Chuẩn (Production Track)
        </button>

        <button
          onClick={() => setActiveTab('research')}
          style={{
            padding: '10px 20px',
            borderRadius: '8px',
            border: activeTab === 'research' ? '1px solid #a855f7' : '1px solid transparent',
            background: activeTab === 'research' ? 'rgba(168, 85, 247, 0.15)' : 'rgba(255, 255, 255, 0.05)',
            color: activeTab === 'research' ? '#c084fc' : '#94a3b8',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
        >
          🧪 Research Lab FP32 (RQ3–RQ4 Outer-Fold Parity)
        </button>
      </div>

      {activeTab === 'research' ? (
        <ResearchLabView />
      ) : (
        <>
          {!selectedImage && !isProcessing && (
            <ImageDropzone onImageSelected={handleImageSelected} isProcessing={isProcessing} />
          )}

          {isProcessing && progressEvent && (
            <AnalysisProgress progressEvent={progressEvent} onCancel={handleCancel} />
          )}

          {globalError && (
            <div
              className="glass-panel"
              style={{
                padding: '16px 20px',
                marginBottom: '24px',
                borderLeft: '4px solid var(--accent-rose)',
                background: 'rgba(244, 63, 94, 0.1)',
                color: '#fca5a5',
              }}
            >
              <strong>Lỗi hệ thống:</strong> {globalError}
            </div>
          )}

          {result && selectedImage && (
            <>
              <ResultVerdictCard
                verdict={result.result.label}
                confidence={result.result.confidence}
                probabilities={result.result.probabilities}
                explanation={result.result.explanation}
                modelAvailable={result.modelAvailable}
                modelStatus={result.modelStatus}
              />

              <HeatmapViewer
                previewUrl={selectedImage.previewUrl}
                width={selectedImage.width}
                height={selectedImage.height}
                localization={result.localization}
              />

              <EvidenceLedger
                supportingEvidence={result.forensics.supportingEvidence}
                refutingEvidence={result.forensics.refutingEvidence}
                limitations={result.limitations}
              />

              <div className="grid-2col">
                <ForensicsInspector signals={result.forensics.signals} />
                <ProvenanceInspector provenance={result.provenance} />
              </div>

              <ReportActions
                result={result}
                onClearSession={handleClearSession}
                onNewImage={handleNewImage}
              />
            </>
          )}
        </>
      )}
    </div>
  );
};

export default App;
