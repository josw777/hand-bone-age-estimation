import React from 'react';
import { Cpu, CheckCircle2, AlertTriangle, Timer } from 'lucide-react';

const steps = [
  'YOLOX-S 손 영역 검출',
  'Hand Segmentation',
  'PCA + 손가락/손목 방향 정렬',
  'Masked Percentile (p1~p99)',
  '512×512 Resize + Padding',
  'ConvNeXt V1-Tiny + Sex-specific LDL',
];

export default function ModelPipelineCard({ predictionData }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex flex-col space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5 text-xs font-bold text-slate-700 tracking-wider">
          <Cpu className="w-4 h-4 text-blue-600" />
          <span>AI PIPELINE</span>
        </div>
        <span className="text-[10px] bg-slate-100 text-slate-600 font-semibold px-2 py-0.5 rounded">FINAL MODEL</span>
      </div>

      <div className="space-y-1.5">
        {steps.map((step, index) => (
          <div key={step} className="flex items-center gap-2 text-[11px] text-slate-700">
            <span className="w-5 h-5 shrink-0 rounded-full bg-blue-50 text-blue-700 border border-blue-100 flex items-center justify-center font-bold text-[10px]">{index + 1}</span>
            <span>{step}</span>
          </div>
        ))}
      </div>

      {predictionData && (
        <div className="pt-2 border-t border-slate-100 space-y-1.5 text-[11px]">
          <div className="flex justify-between"><span className="text-slate-500">Device</span><span className="font-semibold text-slate-700">{predictionData.device}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Input</span><span className="font-semibold text-slate-700">{predictionData.input_size}</span></div>
          <div className="flex justify-between items-center">
            <span className="text-slate-500 flex items-center gap-1"><Timer className="w-3 h-3" />Processing</span>
            <span className="font-semibold text-slate-700">{(predictionData.processing_time_ms / 1000).toFixed(2)} s</span>
          </div>
          <div className={`flex items-center gap-1.5 rounded-md px-2 py-1 ${predictionData.used_fallback ? 'bg-amber-50 text-amber-700' : 'bg-emerald-50 text-emerald-700'}`}>
            {predictionData.used_fallback ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
            <span>{predictionData.used_fallback ? '전처리 fallback 사용' : '전체 전처리 정상 적용'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
