import React from 'react';
import { Activity, FileText, ShieldCheck } from 'lucide-react';

export default function AnalysisSummary({ originalImage, preprocessedImage, predictionData, onOpenExport }) {
  return (
    <div className="grid grid-cols-12 gap-4 mt-4">
      <div className="col-span-8 bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-700 uppercase tracking-wider">
            <Activity className="w-4 h-4 text-blue-600" />
            <span>PREPROCESSING RESULT</span>
          </div>
          <span className="text-[10px] bg-blue-50 text-blue-700 font-semibold px-2 py-0.5 rounded">Actual model input</span>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-slate-200 bg-slate-950 overflow-hidden">
            <div className="px-3 py-1.5 text-[11px] font-bold text-slate-200 border-b border-slate-800">Original X-ray</div>
            <div className="h-48 flex items-center justify-center p-2">
              <img src={originalImage} alt="Original" className="h-full max-w-full object-contain" />
            </div>
          </div>
          <div className="rounded-lg border border-blue-200 bg-slate-950 overflow-hidden">
            <div className="px-3 py-1.5 text-[11px] font-bold text-blue-300 border-b border-slate-800">Final 512×512 AI Input</div>
            <div className="h-48 flex items-center justify-center p-2">
              {preprocessedImage ? (
                <img src={preprocessedImage} alt="Preprocessed" className="h-full max-w-full object-contain" />
              ) : (
                <div className="text-xs text-slate-500 text-center">AI Prediction 실행 후<br />실제 전처리 결과가 표시됩니다.</div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="col-span-4 bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex flex-col justify-between">
        <div>
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">
            <ShieldCheck className="w-4 h-4 text-blue-600" />
            <span>ANALYSIS SUMMARY</span>
          </div>

          {predictionData ? (
            <div className="space-y-2 text-xs">
              <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                <div className="text-[10px] uppercase tracking-wider font-bold text-blue-700">Predicted Bone Age</div>
                <div className="text-2xl font-extrabold text-blue-950 mt-1">{predictionData.predicted_age_display}</div>
                <div className="text-[11px] text-slate-600 mt-1">{predictionData.predicted_age_months.toFixed(1)} months</div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="bg-slate-50 border border-slate-100 rounded p-2"><span className="text-slate-400 block">Sex</span><strong>{predictionData.sex}</strong></div>
                <div className="bg-slate-50 border border-slate-100 rounded p-2"><span className="text-slate-400 block">Fallback</span><strong>{predictionData.used_fallback ? 'Used' : 'No'}</strong></div>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-200 p-6 text-xs text-slate-400 text-center">분석 전입니다.</div>
          )}
        </div>

        <div className="mt-4 space-y-2">
          <button
            onClick={onOpenExport}
            disabled={!predictionData}
            className={`w-full py-2.5 px-4 rounded-lg text-xs font-bold transition flex items-center justify-center space-x-2 ${
              predictionData ? 'bg-slate-900 hover:bg-slate-800 text-white' : 'bg-slate-100 text-slate-400 cursor-not-allowed'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>분석 결과 보기</span>
          </button>
          <div className="text-[10px] leading-relaxed text-slate-400 text-center">
            연구·시연용 시스템이며 의료진의 임상 판독을 대체하지 않습니다.
          </div>
        </div>
      </div>
    </div>
  );
}
