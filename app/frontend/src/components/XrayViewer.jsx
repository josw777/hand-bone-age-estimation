import React, { useRef, useState } from 'react';
import { Move, ZoomIn, Ruler, Sun, RotateCcw, Upload, Image as ImageIcon } from 'lucide-react';

export default function XrayViewer({
  viewMode,
  isAnalyzing,
  displayImage,
  hasUploadedFile,
  hasPreprocessedImage,
  onFileSelected,
}) {
  const [zoomLevel, setZoomLevel] = useState(65);
  const [isInverted, setIsInverted] = useState(false);
  const [activeTool, setActiveTool] = useState('pan');
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const fileInputRef = useRef(null);
  const viewerContainerRef = useRef(null);

  const handleReset = () => {
    setZoomLevel(65);
    setIsInverted(false);
    setActiveTool('pan');
    setPanOffset({ x: 0, y: 0 });
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 8 : -8;
    setZoomLevel((prev) => Math.min(Math.max(prev + delta, 30), 250));
  };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    if (activeTool === 'pan') {
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX - panOffset.x, y: e.clientY - panOffset.y };
    } else if (activeTool === 'zoom') {
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX, y: e.clientY, initialZoom: zoomLevel };
    }
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    if (activeTool === 'pan') {
      setPanOffset({
        x: e.clientX - dragStartRef.current.x,
        y: e.clientY - dragStartRef.current.y,
      });
    } else if (activeTool === 'zoom') {
      const deltaY = dragStartRef.current.y - e.clientY;
      const next = Math.min(Math.max(dragStartRef.current.initialZoom + deltaY * 0.5, 30), 250);
      setZoomLevel(Math.round(next));
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onFileSelected(file);
    e.target.value = '';
  };

  return (
    <div className="flex-1 bg-[#090d16] rounded-xl border border-slate-800 shadow-2xl flex flex-col overflow-hidden relative select-none">
      <div className="bg-slate-900/90 border-b border-slate-800/80 px-4 py-2 flex items-center justify-between text-xs text-slate-300">
        <div className="flex items-center space-x-4">
          <span className="font-medium text-slate-400">Series: <strong className="text-slate-200">Left Hand X-ray</strong></span>
          <span className="text-slate-500">| View: <strong className="text-blue-400">{viewMode === 'preprocess' ? 'AI Input' : 'Original'}</strong></span>
        </div>
        <div className="flex items-center space-x-3 text-slate-400">
          <span>Zoom: <strong className="text-blue-400 font-mono">{zoomLevel}%</strong></span>
        </div>
      </div>

      <div
        ref={viewerContainerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        className={`relative flex-1 bg-[#05080e] overflow-hidden flex items-center justify-center ${
          activeTool === 'pan' ? (isDragging ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-ns-resize'
        }`}
      >
        <div className="absolute left-3 top-4 z-30 bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg p-1 flex flex-col space-y-2 shadow-xl">
          <button onClick={() => setActiveTool('pan')} className={`p-2 rounded-md ${activeTool === 'pan' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`} title="Pan">
            <Move className="w-4 h-4" />
          </button>
          <button onClick={() => setActiveTool('zoom')} className={`p-2 rounded-md ${activeTool === 'zoom' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`} title="Zoom">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button disabled className="p-2 rounded-md text-slate-600" title="Measurement scale is not calibrated">
            <Ruler className="w-4 h-4" />
          </button>
          <button onClick={() => setIsInverted((v) => !v)} className={`p-2 rounded-md ${isInverted ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`} title="Invert">
            <Sun className="w-4 h-4" />
          </button>
        </div>

        {isAnalyzing && <div className="animate-scan z-30 pointer-events-none" />}

        {!displayImage ? (
          <div className="text-slate-500 flex flex-col items-center gap-3">
            <ImageIcon className="w-12 h-12" />
            <span>수부 X-ray를 업로드하세요.</span>
          </div>
        ) : (
          <div
            className="relative flex items-center justify-center transition-transform duration-75 ease-out"
            style={{ transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel / 100})` }}
          >
            <img
              src={displayImage}
              alt="Hand X-Ray"
              className="max-h-[520px] w-auto object-contain shadow-2xl rounded pointer-events-none"
              style={{ filter: isInverted ? 'invert(100%)' : 'none' }}
            />
          </div>
        )}
      </div>

      <div className="bg-slate-900 border-t border-slate-800 px-4 py-2 flex items-center justify-between text-xs text-slate-300 z-30">
        <div className="flex items-center space-x-2">
          <button onClick={() => setZoomLevel((z) => Math.max(30, z - 10))} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded">-</button>
          <span className="font-mono text-slate-200">{zoomLevel}%</span>
          <button onClick={() => setZoomLevel((z) => Math.min(250, z + 10))} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded">+</button>
          <button onClick={() => { setZoomLevel(65); setPanOffset({ x: 0, y: 0 }); }} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded">Fit</button>
          <button onClick={handleReset} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded flex items-center space-x-1">
            <RotateCcw className="w-3 h-3" /><span>Reset</span>
          </button>
          <button onClick={() => fileInputRef.current?.click()} className="px-2.5 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white flex items-center space-x-1">
            <Upload className="w-3 h-3" /><span>Upload X-ray</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,image/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        <div className="flex items-center space-x-1.5 text-slate-400">
          <span className={`w-2 h-2 rounded-full ${hasUploadedFile ? 'bg-emerald-500' : 'bg-amber-400'}`} />
          <span>{hasPreprocessedImage ? 'AI preprocessing completed' : hasUploadedFile ? 'Image ready' : 'Demo image'}</span>
        </div>
      </div>
    </div>
  );
}
