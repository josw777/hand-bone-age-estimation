import React from 'react';
import { User, Play, RefreshCw, CheckCircle2, UploadCloud } from 'lucide-react';

export default function PatientCard({
  onRunPrediction,
  isAnalyzing,
  predictionData,
  sex,
  setSex,
  chronologicalAgeYears,
  setChronologicalAgeYears,
  selectedFilename,
}) {
  const diff = predictionData?.difference_months;

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex flex-col space-y-4">
      <div>
        <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2.5">
          <User className="w-3.5 h-3.5 text-blue-600" />
          <span>INPUT</span>
        </div>

        <div className="space-y-3 text-xs bg-slate-50 p-3 rounded-lg border border-slate-100">
          <div>
            <span className="text-slate-400 block text-[11px] mb-1">X-ray Image</span>
            <div className="font-medium text-slate-700 truncate flex items-center gap-1.5" title={selectedFilename || 'No file selected'}>
              <UploadCloud className="w-3.5 h-3.5 text-blue-500 shrink-0" />
              {selectedFilename || 'Upload 버튼으로 이미지 선택'}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label>
              <span className="text-slate-400 block text-[11px] mb-1">Sex</span>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                className="w-full rounded-md border border-slate-200 bg-white px-2 py-1.5 text-slate-800 font-medium outline-none focus:border-blue-400"
              >
                <option value="F">Female</option>
                <option value="M">Male</option>
              </select>
            </label>

            <label>
              <span className="text-slate-400 block text-[11px] mb-1">Chronological age (y)</span>
              <input
                value={chronologicalAgeYears}
                onChange={(e) => setChronologicalAgeYears(e.target.value)}
                type="number"
                min="0"
                max="20"
                step="0.1"
                placeholder="optional"
                className="w-full rounded-md border border-slate-200 bg-white px-2 py-1.5 text-slate-800 outline-none focus:border-blue-400"
              />
            </label>
          </div>
        </div>
      </div>

      <button
        onClick={onRunPrediction}
        disabled={isAnalyzing}
        className={`w-full py-2.5 px-4 rounded-lg font-semibold text-xs text-white shadow-md flex items-center justify-center space-x-2 transition ${
          isAnalyzing ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:scale-[0.99] shadow-blue-500/25'
        }`}
      >
        {isAnalyzing ? (
          <><RefreshCw className="w-4 h-4 animate-spin" /><span>모델 로딩 / AI 분석 중...</span></>
        ) : (
          <><Play className="w-4 h-4 fill-current" /><span>Run AI Prediction</span></>
        )}
      </button>

      <div className="bg-blue-50/50 border border-blue-100 rounded-xl p-3.5 relative overflow-hidden min-h-[178px]">
        <div className="text-[11px] font-bold text-blue-900 uppercase tracking-wider mb-1 flex items-center justify-between">
          <span>PREDICTION BONE AGE</span>
          <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full text-[10px] font-semibold">AI Predict</span>
        </div>

        {predictionData ? (
          <>
            <div className="flex items-baseline space-x-1 my-1">
              <span className="text-3xl font-extrabold text-blue-900 tracking-tight">{predictionData.predicted_age_years.toFixed(2)}</span>
              <span className="text-sm font-bold text-blue-700">years</span>
            </div>
            <div className="text-sm font-bold text-slate-700 mb-2">{predictionData.predicted_age_display} · {predictionData.predicted_age_months.toFixed(1)}개월</div>
            <div className="text-[11px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200/60 rounded-md px-2 py-1.5 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span>AI inference completed</span>
            </div>
            {diff !== null && diff !== undefined && (
              <div className="mt-2 text-[11px] pt-2 border-t border-blue-100 flex justify-between">
                <span className="text-slate-500">실제 나이와 예측 차이:</span>
                <span className="font-semibold text-blue-700">{diff >= 0 ? '+' : ''}{diff.toFixed(1)}개월</span>
              </div>
            )}
          </>
        ) : (
          <div className="h-32 flex items-center justify-center text-center text-xs text-slate-400 leading-relaxed">
            X-ray와 성별을 입력한 뒤<br />AI Prediction을 실행하세요.
          </div>
        )}
      </div>
    </div>
  );
}
