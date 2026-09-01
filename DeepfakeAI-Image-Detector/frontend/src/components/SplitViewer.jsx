import React, { useState, useRef, useCallback } from 'react';
import { Eye, Layers, Split, Sliders, Grid } from 'lucide-react';

export default function SplitViewer({ rawImageUrl, gradcamImageUrl, aiProbability, verdict }) {
  const [sliderPos, setSliderPos] = useState(50); // percentage 0 to 100
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'side-by-side' | 'blend'
  const [blendOpacity, setBlendOpacity] = useState(0.75);
  const [showGrid, setShowGrid] = useState(false);
  const containerRef = useRef(null);
  const isDraggingRef = useRef(false);

  const handleMove = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percent);
  }, []);

  const handleMouseDown = () => {
    isDraggingRef.current = true;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleMouseMove = (e) => {
    if (isDraggingRef.current) {
      handleMove(e.clientX);
    }
  };

  const handleTouchMove = (e) => {
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  };

  return (
    <div className="glass-panel rounded-lg p-5 border border-cyan-500/20 shadow-2xl flex flex-col gap-4 animate-reveal-stagger-1 transition-all duration-300">
      
      {/* Viewer Header with Mode Selectors */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div>
          <h3 className="label-caps text-cyan-400 text-xs flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            Grad-CAM Spatial Activation Inspector
          </h3>
          <p className="text-[11px] font-mono text-slate-400 mt-0.5">
            ConvNeXt-Tiny stage-4 attention maps highlighting generative manipulation cues
          </p>
        </div>

        {/* View Mode Controls */}
        <div className="flex items-center gap-1.5 p-1 rounded bg-slate-900/90 border border-slate-800 text-xs font-mono">
          <button
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 text-[11px] font-semibold ${
              viewMode === 'split' ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/40 shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>Split Slider</span>
          </button>
          
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 text-[11px] font-semibold ${
              viewMode === 'side-by-side' ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/40 shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>

          <button
            onClick={() => setViewMode('blend')}
            className={`px-2.5 py-1 rounded transition-all flex items-center gap-1.5 text-[11px] font-semibold ${
              viewMode === 'blend' ? 'bg-violet-500/25 text-violet-300 border border-violet-500/40 shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Alpha Blend</span>
          </button>

          {/* Grid Toggle */}
          <button
            onClick={() => setShowGrid(!showGrid)}
            className={`p-1.5 rounded transition-all ${
              showGrid ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/40' : 'text-slate-500 hover:text-slate-300'
            }`}
            title="Toggle Forensic Coordinate Grid"
          >
            <Grid className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Image Display Area */}
      {viewMode === 'split' && (
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onMouseMove={handleMouseMove}
          onTouchMove={handleTouchMove}
          className="relative w-full aspect-square max-w-xl mx-auto rounded overflow-hidden cursor-ew-resize select-none border border-cyan-500/30 bg-black shadow-inner"
        >
          {/* Base Layer: Grad-CAM Heatmap */}
          {gradcamImageUrl ? (
            <img
              src={gradcamImageUrl}
              alt="Grad-CAM Forensic Heatmap"
              className="absolute inset-0 w-full h-full object-contain"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-slate-500 font-mono text-xs">
              Generating Heatmap...
            </div>
          )}

          {/* Top Layer: Raw Input Image clipped by slider position */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ width: `${sliderPos}%` }}
          >
            <img
              src={rawImageUrl}
              alt="Original RGB Source"
              className="absolute inset-0 w-full h-full object-contain max-w-none"
              style={{
                width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%',
                height: containerRef.current ? `${containerRef.current.clientHeight}px` : '100%',
              }}
            />
          </div>

          {/* Optional Coordinate Grid Overlay */}
          {showGrid && (
            <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />
          )}

          {/* Draggable Divider Line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-gradient-to-b from-cyan-400 via-sky-300 to-violet-400 shadow-[0_0_12px_2px_rgba(6,182,212,0.85)] pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            {/* Center Handle Badge */}
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-slate-900 border-2 border-cyan-400 flex items-center justify-center shadow-neon-cyan">
              <Split className="w-3 h-3 text-cyan-400" />
            </div>
          </div>

          {/* Corner Labels */}
          <div className="absolute top-3 left-3 bg-black/80 backdrop-blur-md px-2.5 py-1 rounded border border-slate-700 label-caps text-[10px] text-slate-300 pointer-events-none">
            RAW SOURCE
          </div>
          <div className="absolute top-3 right-3 bg-black/80 backdrop-blur-md px-2.5 py-1 rounded border border-cyan-500/40 label-caps text-[10px] text-cyan-300 pointer-events-none flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
            GRAD-CAM HEATMAP
          </div>

          {/* Bottom Slider Position Indicator */}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/85 backdrop-blur-md px-3 py-1 rounded-full border border-slate-800 font-mono text-[10px] text-slate-400 pointer-events-none">
            Slide to Compare • {Math.round(sliderPos)}% : {100 - Math.round(sliderPos)}%
          </div>
        </div>
      )}

      {viewMode === 'side-by-side' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1.5 text-center">
            <div className="relative aspect-square rounded overflow-hidden border border-slate-800 bg-black">
              <img src={rawImageUrl} alt="Raw Input" className="w-full h-full object-contain" />
              {showGrid && <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />}
            </div>
            <span className="label-caps text-[10px] text-slate-400">Original RGB Input (224x224)</span>
          </div>

          <div className="space-y-1.5 text-center">
            <div className="relative aspect-square rounded overflow-hidden border border-cyan-500/40 bg-black shadow-neon-cyan flex items-center justify-center">
              {gradcamImageUrl ? (
                <img src={gradcamImageUrl} alt="Grad-CAM" className="w-full h-full object-contain" />
              ) : (
                <span className="font-mono text-xs text-slate-500">Grad-CAM Not Available</span>
              )}
              {showGrid && <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />}
            </div>
            <span className="label-caps text-[10px] text-cyan-300">ConvNeXt Activation Heatmap</span>
          </div>
        </div>
      )}

      {viewMode === 'blend' && (
        <div className="space-y-3">
          <div className="relative w-full aspect-square max-w-xl mx-auto rounded overflow-hidden border border-violet-500/40 bg-black shadow-neon-violet">
            <img src={rawImageUrl} alt="Raw Source" className="absolute inset-0 w-full h-full object-contain" />
            {gradcamImageUrl && (
              <img
                src={gradcamImageUrl}
                alt="Grad-CAM Overlay"
                className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
                style={{ opacity: blendOpacity }}
              />
            )}
            {showGrid && <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />}
          </div>

          {/* Opacity Slider Control */}
          <div className="flex items-center justify-between gap-4 max-w-md mx-auto px-4 py-2 rounded bg-slate-900/80 border border-slate-800 text-xs font-mono">
            <span className="text-slate-400">Heatmap Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <span className="text-cyan-300 font-bold w-12 text-right">
              {Math.round(blendOpacity * 100)}%
            </span>
          </div>
        </div>
      )}

      {/* Heatmap Legend Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-2 p-2.5 rounded bg-slate-900/60 border border-slate-800/90 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="label-caps text-slate-400 text-[10px]">Intensity:</span>
          <div className="w-32 h-2.5 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-rose-600 border border-slate-700" />
        </div>
        <div className="flex items-center gap-4 text-[10px] text-slate-400">
          <span>🔵 Low Background</span>
          <span>🟡 Mid Frequency</span>
          <span>🔴 Synthetic Anomaly</span>
        </div>
      </div>

    </div>
  );
}
