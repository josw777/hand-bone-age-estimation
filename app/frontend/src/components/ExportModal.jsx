import React from 'react';
import { X, Printer, FileText } from 'lucide-react';

export default function ExportModal({ isOpen, onClose, predictionData, baseImage, sex, chronologicalAgeYears }) {
  if (!isOpen || !predictionData) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">
        <div className="bg-slate-900 text-white px-5 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FileText className="w-5 h-5 text-blue-400" />
            <h3 className="font-bold text-base">Shilla BoneAge AI 분석 결과</h3>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white p-1 rounded-lg transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto bg-slate-50">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div className="border-b border-slate-200 pb-4">
              <h1 className="text-xl font-extrabold text-blue-950">Pediatric Bone Age AI Report</h1>
              <p className="text-xs text-slate-500">YOLOX-S + Hand Segmentation + ConvNeXt V1-Tiny</p>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs bg-slate-50 p-3 rounded-lg">
              <div><span className="text-slate-400 block">Sex</span><strong>{sex === 'M' ? 'Male' : 'Female'}</strong></div>
              <div><span className="text-slate-400 block">Chronological age</span><strong>{chronologicalAgeYears || 'Not entered'}{chronologicalAgeYears ? ' y' : ''}</strong></div>
              <div><span className="text-slate-400 block">Input</span><strong>Left hand X-ray</strong></div>
            </div>

            <div className="border border-blue-200 bg-blue-50/50 p-4 rounded-lg">
              <span className="text-xs font-bold text-blue-900 block mb-1">AI 예측 뼈나이</span>
              <span className="text-2xl font-extrabold text-blue-700">{predictionData.predicted_age_display}</span>
              <span className="text-sm text-slate-600 block mt-1">{predictionData.predicted_age_months.toFixed(1)}개월</span>
              {predictionData.difference_months !== null && predictionData.difference_months !== undefined && (
                <span className="text-[11px] text-slate-600 block mt-2">실제 나이와 예측 차이: {predictionData.difference_months >= 0 ? '+' : ''}{predictionData.difference_months.toFixed(1)}개월</span>
              )}
            </div>

            <div className="flex space-x-4 items-center bg-slate-900 p-3 rounded-lg text-white text-xs">
              <img src={baseImage} alt="Report Xray" className="h-24 w-auto rounded object-contain bg-black" />
              <div className="space-y-1">
                <p className="font-bold text-blue-400">Inference Information</p>
                <p className="text-slate-300">Device: {predictionData.device}</p>
                <p className="text-slate-300">Processing: {(predictionData.processing_time_ms / 1000).toFixed(2)} sec</p>
                <p className="text-slate-400 text-[11px]">Fallback: {predictionData.used_fallback ? 'used' : 'not used'}</p>
              </div>
            </div>

            <p className="text-[10px] text-slate-400">본 결과는 연구 및 시연 목적으로 제공되며 임상 진단을 대체하지 않습니다.</p>
          </div>
        </div>

        <div className="bg-white border-t border-slate-200 px-6 py-4 flex items-center justify-between">
          <span className="text-xs text-slate-500">브라우저 인쇄 기능에서 PDF로 저장할 수 있습니다.</span>
          <div className="flex items-center space-x-3">
            <button onClick={onClose} className="px-4 py-2 text-xs font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg">닫기</button>
            <button onClick={() => window.print()} className="px-5 py-2 text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center space-x-2">
              <Printer className="w-4 h-4" /><span>인쇄 / PDF 저장</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
