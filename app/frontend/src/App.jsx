import React, { useEffect, useMemo, useState } from 'react';
import Header from './components/Header';
import ImageSidebar from './components/ImageSidebar';
import XrayViewer from './components/XrayViewer';
import PatientCard from './components/PatientCard';
import ModelPipelineCard from './components/ModelPipelineCard';
import AnalysisSummary from './components/AnalysisSummary';
import ExportModal from './components/ExportModal';

const demoImage = '/xray_original.jpg';

export default function App() {
  const [viewMode, setViewMode] = useState('original');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(demoImage);
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [sex, setSex] = useState('F');
  const [chronologicalAgeYears, setChronologicalAgeYears] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    return () => {
      if (originalImage?.startsWith('blob:')) URL.revokeObjectURL(originalImage);
    };
  }, [originalImage]);

  const displayImage = useMemo(() => {
    if (viewMode === 'preprocess' && preprocessedImage) return preprocessedImage;
    return originalImage || demoImage;
  }, [viewMode, preprocessedImage, originalImage]);

  const handleFileSelected = (file) => {
    if (!file) return;

    setErrorMessage('');
    setPredictionData(null);
    setPreprocessedImage(null);
    setViewMode('original');

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setOriginalImage(objectUrl);
  };

  const handleRunPrediction = async () => {
    if (!selectedFile) {
      setErrorMessage('먼저 실제 수부 X-ray 이미지를 업로드해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage('');

    try {
      const form = new FormData();
      form.append('image', selectedFile);
      form.append('sex', sex);

      const chrono = Number(chronologicalAgeYears);
      if (chronologicalAgeYears !== '' && Number.isFinite(chrono) && chrono >= 0) {
        form.append('chronological_age_months', String(chrono * 12));
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: form,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail || 'AI 추론에 실패했습니다.');
      }

      setPredictionData(payload);
      setPreprocessedImage(payload.preprocessed_image);
      setViewMode('preprocess');
    } catch (error) {
      setErrorMessage(error?.message || 'AI 추론 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#eef2f6] flex flex-col font-['Noto_Sans_KR','Inter',sans-serif]">
      <Header onOpenExport={() => setIsExportOpen(true)} hasPrediction={Boolean(predictionData)} />

      <main className="flex-1 p-4 max-w-[1680px] w-full mx-auto flex flex-col space-y-4">
        {errorMessage && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm font-medium text-red-700">
            {errorMessage}
          </div>
        )}

        <div className="flex flex-1 space-x-4 min-h-[590px]">
          <ImageSidebar
            viewMode={viewMode}
            setViewMode={setViewMode}
            originalImage={originalImage}
            preprocessedImage={preprocessedImage}
          />

          <XrayViewer
            viewMode={viewMode}
            isAnalyzing={isAnalyzing}
            displayImage={displayImage}
            hasUploadedFile={Boolean(selectedFile)}
            hasPreprocessedImage={Boolean(preprocessedImage)}
            onFileSelected={handleFileSelected}
          />

          <div className="w-80 flex flex-col space-y-4 shrink-0">
            <PatientCard
              onRunPrediction={handleRunPrediction}
              isAnalyzing={isAnalyzing}
              predictionData={predictionData}
              sex={sex}
              setSex={setSex}
              chronologicalAgeYears={chronologicalAgeYears}
              setChronologicalAgeYears={setChronologicalAgeYears}
              selectedFilename={selectedFile?.name || ''}
            />
            <ModelPipelineCard predictionData={predictionData} />
          </div>
        </div>

        <AnalysisSummary
          originalImage={originalImage}
          preprocessedImage={preprocessedImage}
          predictionData={predictionData}
          onOpenExport={() => setIsExportOpen(true)}
        />
      </main>

      <ExportModal
        isOpen={isExportOpen}
        onClose={() => setIsExportOpen(false)}
        predictionData={predictionData}
        baseImage={originalImage}
        sex={sex}
        chronologicalAgeYears={chronologicalAgeYears}
      />
    </div>
  );
}
