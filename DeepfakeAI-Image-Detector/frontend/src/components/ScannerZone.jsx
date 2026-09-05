import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  Crosshair, 
  CheckCircle2, 
  Sparkles, 
  Zap, 
  Layers, 
  Camera, 
  Cpu, 
  Eye, 
  Fingerprint,
  Radio
} from 'lucide-react';

const PRESET_SAMPLES = [
  {
    id: 'nikon_d850',
    name: '1. DSLR Camera Photo',
    type: 'AUTHENTIC',
    filePath: '/test_images/1_camera_dslr_photo.jpg',
    desc: 'Real optical sensor capture from a camera with natural sensor grain'
  },
  {
    id: 'diffusion_gen',
    name: '2. AI Diffusion Synthesis',
    type: 'SYNTHETIC',
    filePath: '/test_images/6_ai_generated_diffusion.jpg',
    desc: 'Synthetic diffusion generation with high-frequency latent anomalies'
  },
  {
    id: 'street_scene',
    name: '3. AI Street Synthesis',
    type: 'SYNTHETIC',
    filePath: '/test_images/3_street_photo.jpg',
    desc: 'Synthetic photorealistic street composition'
  },
  {
    id: 'digital_graphic',
    name: '4. Digital Graphic Capture',
    type: 'AUTHENTIC',
    filePath: '/test_images/4_digital_graphic.png',
    desc: 'Clean digital visual graphic capture'
  },
  {
    id: 'web_screen',
    name: '5. Web Interface Capture',
    type: 'SYNTHETIC',
    filePath: '/test_images/web_screenshot.png',
    desc: 'Synthetic UI screenshot capture'
  }
];

export default function ScannerZone({ onFileSelect, onError, isAnalyzing, selectedPreviewUrl }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [scanStep, setScanStep] = useState(0);
  const fileInputRef = useRef(null);

  const pipelineStages = [
    { id: 1, name: 'Image Ingestion & Letterbox Normalization', short: 'INGEST' },
    { id: 2, name: 'RGB Spatial Domain Boundary Analysis', short: 'SPATIAL' },
    { id: 3, name: '2D Fast Fourier Transform (Log-Magnitude Spectrum)', short: '2D-FFT' },
    { id: 4, name: 'ConvNeXt-Tiny Deep Feature Extraction (768-D)', short: 'CONVNEXT' },
    { id: 5, name: 'Dual-Stream Fusion & Explainability Synthesis', short: 'VERDICT' }
  ];

  useEffect(() => {
    let interval;
    if (isAnalyzing) {
      setScanStep(0);
      interval = setInterval(() => {
        setScanStep((prev) => {
          if (prev < pipelineStages.length - 1) {
            return prev + 1;
          }
          return prev;
        });
      }, 320);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isAnalyzing) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelect(file);
      } else if (onError) {
        onError('Invalid file format. Please upload a valid image file (JPEG, PNG, WEBP).');
      }
    }
  };

  const handlePresetClick = async (preset) => {
    if (isAnalyzing) return;
    try {
      if (preset.filePath) {
        const res = await fetch(preset.filePath);
        if (res.ok) {
          const blob = await res.blob();
          const ext = preset.filePath.endsWith('.png') ? 'png' : 'jpg';
          const file = new File([blob], `${preset.id}.${ext}`, { type: blob.type || (ext === 'png' ? 'image/png' : 'image/jpeg') });
          onFileSelect(file);
          return;
        }
      }
    } catch (err) {
      console.warn('Preset fetch error, falling back to canvas:', err);
    }

    // Fallback simple canvas
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = preset.type === 'AUTHENTIC' ? '#1e293b' : '#31103f';
    ctx.fillRect(0, 0, 224, 224);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `${preset.id}.png`, { type: 'image/png' });
        onFileSelect(file);
      }
    }, 'image/png');
  };

  return (
    <div className="glass-panel rounded-lg p-5 sm:p-6 flex flex-col relative overflow-hidden transition-all duration-300">
      
      {/* Scanline Background Layer */}
      <div className="absolute inset-0 scanline-bg opacity-30 pointer-events-none" />

      {/* Target Acquisition Header */}
      <div className="flex justify-between items-center mb-4 relative z-10">
        <div className="flex items-center gap-2">
          <Crosshair className="w-4 h-4 text-cyan-400 animate-pulse" />
          <h2 className="label-caps text-cyan-400 tracking-widest text-xs">
            Target Acquisition & Evidence Ingestion
          </h2>
        </div>
        <span className="label-caps text-[10px] text-slate-500">
          RGB MATRIX // 224x224 DOMAIN
        </span>
      </div>

      {/* Dropzone Container */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !isAnalyzing && fileInputRef.current?.click()}
        className={`flex-grow min-h-[230px] border border-dashed rounded bg-[#060e20]/60 flex flex-col items-center justify-center p-6 sm:p-7 relative group cursor-pointer transition-all duration-300 ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-950/40 shadow-neon-cyan scale-[1.005]'
            : selectedPreviewUrl
            ? 'border-cyan-500/40 bg-[#060e20]/80'
            : 'border-slate-800 hover:border-cyan-500/50 hover:bg-[#060e20]/90'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              const file = e.target.files[0];
              e.target.value = '';
              if (file.type.startsWith('image/')) {
                onFileSelect(file);
              } else if (onError) {
                onError('Invalid file format. Please upload a valid image file (JPEG, PNG, WEBP).');
              }
            }
          }}
        />

        {/* Ambient Targeting Crosshairs on Hover */}
        <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <div className="absolute top-1/2 left-0 w-full h-[1px] bg-cyan-400/25" />
          <div className="absolute top-0 left-1/2 w-[1px] h-full bg-cyan-400/25" />
          <div className="absolute top-1/2 left-1/2 w-10 h-10 border border-cyan-400/40 -translate-x-1/2 -translate-y-1/2 rounded-full" />
        </div>

        {/* Corner HUD Reticles */}
        <div className="absolute top-2 left-2 w-3.5 h-3.5 border-t-2 border-l-2 border-cyan-400/70 pointer-events-none" />
        <div className="absolute top-2 right-2 w-3.5 h-3.5 border-t-2 border-r-2 border-cyan-400/70 pointer-events-none" />
        <div className="absolute bottom-2 left-2 w-3.5 h-3.5 border-b-2 border-l-2 border-cyan-400/70 pointer-events-none" />
        <div className="absolute bottom-2 right-2 w-3.5 h-3.5 border-b-2 border-r-2 border-cyan-400/70 pointer-events-none" />

        {/* Preview or Ingestion Prompt */}
        {selectedPreviewUrl ? (
          <div className="relative w-48 h-48 rounded-lg overflow-hidden border border-cyan-500/50 shadow-2xl bg-black animate-reveal">
            <img
              src={selectedPreviewUrl}
              alt="Target Ingestion Preview"
              className="w-full h-full object-contain"
            />
            {/* Smooth Continuous Laser Scan Animation */}
            {isAnalyzing && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                <div className="absolute w-full h-1.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_18px_4px_rgba(6,182,212,0.95)] animate-smooth-laser" />
                <div className="absolute inset-0 bg-cyan-500/10" />
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-3 relative z-10 transition-all duration-300">
            <div className="w-14 h-14 rounded-full bg-slate-900/90 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 group-hover:border-cyan-400 group-hover:shadow-neon-cyan transition-all duration-300">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <p className="font-mono text-sm text-slate-200 font-semibold mb-1">
                Drag & Drop visual evidence or click to browse
              </p>
              <p className="label-caps text-slate-500 text-[10px]">
                Supported: JPG, PNG, WEBP • Max 50MB
              </p>
            </div>
          </div>
        )}

        {/* Sequential 5-Stage Forensic Analysis Progress */}
        {isAnalyzing && (
          <div className="mt-4 w-full max-w-lg space-y-2 relative z-10 animate-reveal">
            <div className="flex items-center justify-between text-xs font-mono text-cyan-300">
              <div className="flex items-center gap-2">
                <Radio className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
                <span className="font-semibold">{pipelineStages[scanStep].name}</span>
              </div>
              <span className="text-[11px] text-cyan-400/80">
                {Math.min(100, Math.round(((scanStep + 1) / pipelineStages.length) * 100))}%
              </span>
            </div>

            {/* Stage Progress Pills */}
            <div className="grid grid-cols-5 gap-1.5">
              {pipelineStages.map((stage, idx) => (
                <div
                  key={stage.id}
                  className={`h-1.5 rounded-full transition-all duration-300 ${
                    idx < scanStep
                      ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]'
                      : idx === scanStep
                      ? 'bg-cyan-400 shadow-[0_0_10px_#06b6d4] animate-pulse'
                      : 'bg-slate-800'
                  }`}
                  title={stage.name}
                />
              ))}
            </div>
          </div>
        )}

      </div>

      {/* Preset Generator Buttons with Real Image Thumbnails */}
      <div className="mt-4 flex items-center gap-2 overflow-x-auto pb-1 relative z-10">
        <span className="label-caps text-[10px] text-slate-500 whitespace-nowrap hidden sm:inline-block">
          FAST SAMPLES:
        </span>
        {PRESET_SAMPLES.map((preset) => (
          <button
            key={preset.id}
            onClick={(e) => {
              e.stopPropagation();
              handlePresetClick(preset);
            }}
            disabled={isAnalyzing}
            className="font-mono text-[11px] px-2.5 py-1.5 rounded-md border border-slate-800 bg-slate-900/80 hover:border-cyan-400/60 hover:text-cyan-300 hover:bg-slate-900 transition-all duration-200 whitespace-nowrap flex items-center gap-2 disabled:opacity-50 hover:-translate-y-0.5 shadow-sm hover:shadow-neon-cyan"
            title={preset.desc}
          >
            {preset.filePath && (
              <img
                src={preset.filePath}
                alt={preset.name}
                className="w-4 h-4 rounded object-cover border border-slate-700"
              />
            )}
            <span className={`w-1.5 h-1.5 rounded-full ${
              preset.type === 'AUTHENTIC' ? 'bg-emerald-400' : 'bg-rose-400'
            }`} />
            <span>{preset.name}</span>
          </button>
        ))}
      </div>

    </div>
  );
}
