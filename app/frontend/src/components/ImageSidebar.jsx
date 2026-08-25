import React from 'react';

export default function ImageSidebar({ viewMode, setViewMode, originalImage, preprocessedImage }) {
  const modes = [
    { id: 'original', label: '원본', image: originalImage },
    { id: 'preprocess', label: '전처리', image: preprocessedImage || originalImage, disabled: !preprocessedImage },
  ];

  return (
    <div className="w-36 flex flex-col space-y-3.5 shrink-0">
      {modes.map((mode) => {
        const isActive = viewMode === mode.id;
        return (
          <div
            key={mode.id}
            onClick={() => !mode.disabled && setViewMode(mode.id)}
            className={`relative rounded-xl border-2 overflow-hidden transition-all duration-200 bg-slate-900 shadow-md ${
              mode.disabled
                ? 'opacity-45 cursor-not-allowed border-slate-300'
                : isActive
                ? 'cursor-pointer border-blue-500 ring-2 ring-blue-400/50 shadow-blue-500/20 scale-[1.02]'
                : 'cursor-pointer border-slate-300 hover:border-blue-400 hover:shadow-lg'
            }`}
          >
            <div className="bg-blue-600 text-white text-center text-xs font-bold py-1 px-2 tracking-wider shadow">
              {mode.label}
            </div>
            <div className="relative w-full h-36 bg-black flex items-center justify-center p-1 overflow-hidden">
              <img src={mode.image} alt={mode.label} className="w-full h-full object-contain" />
            </div>
          </div>
        );
      })}
      <div className="text-[10px] leading-relaxed text-slate-500 bg-white border border-slate-200 rounded-lg p-2.5 shadow-sm">
        <strong className="text-slate-700">전처리</strong><br />
        YOLOX-S → Segmentation → 방향 정렬 → Masked Percentile → 512×512
      </div>
    </div>
  );
}
