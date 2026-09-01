import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Activity, 
  Radio, 
  Cpu, 
  CheckCircle2, 
  AlertTriangle, 
  Layers
} from 'lucide-react';

function useCountUp(target, duration = 800) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (target === null || target === undefined) {
      setVal(0);
      return;
    }
    let startTimestamp = null;
    let frameId;
    const startVal = 0;
    const endVal = Number(target);

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setVal(Number((startVal + (endVal - startVal) * ease).toFixed(2)));
      if (progress < 1) {
        frameId = requestAnimationFrame(step);
      }
    };
    frameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameId);
  }, [target, duration]);
  return val;
}

export default function ForensicMetrics({ result }) {
  if (!result) return null;

  const {
    verdict,
    ai_probability,
    confidence,
    artifact_flags = [],
    fft_spectrum_base64,
    metrics = {}
  } = result;

  const isAi = verdict === 'AI-GENERATED';
  const animatedProb = useCountUp(ai_probability, 850);
  const animatedConf = useCountUp(confidence, 850);

  // Circular gauge parameters
  const radius = 56;
  const strokeWidth = 8;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (animatedProb / 100) * circumference;

  // Diagnostic Telemetry Definitions
  const telemetryItems = [
    {
      id: 'peak_zscore',
      label: 'Spectral Peak Z-Score',
      value: metrics.peak_zscore !== undefined ? `${metrics.peak_zscore}` : 'N/A',
      desc: 'High-frequency spike intensity vs noise floor',
      isAbnormal: metrics.peak_zscore !== undefined && metrics.peak_zscore > 3.8,
      statusText: metrics.peak_zscore !== undefined && metrics.peak_zscore > 3.8 ? 'Spike Detected' : 'Normal Noise'
    },
    {
      id: 'top_1pct_diff',
      label: 'Top 1% Energy Delta',
      value: metrics.top_1pct_diff !== undefined ? `${metrics.top_1pct_diff} dB` : 'N/A',
      desc: 'Spectral energy concentration in outlier bins',
      isAbnormal: metrics.top_1pct_diff !== undefined && metrics.top_1pct_diff > 40.0,
      statusText: metrics.top_1pct_diff !== undefined && metrics.top_1pct_diff > 40.0 ? 'Concentrated (>40 dB)' : 'Uniform Dispersion'
    },
    {
      id: 'grid_spike_score',
      label: 'Harmonic Grid Score',
      value: metrics.grid_spike_score !== undefined ? `${metrics.grid_spike_score}` : 'N/A',
      desc: 'Periodic checkerboard / transposed conv cues',
      isAbnormal: metrics.grid_spike_score !== undefined && metrics.grid_spike_score > 25.0,
      statusText: metrics.grid_spike_score !== undefined && metrics.grid_spike_score > 25.0 ? 'Harmonic Spikes' : 'Non-Periodic'
    },
    {
      id: 'smooth_patch_ratio',
      label: 'Latent Smooth Ratio',
      value: metrics.smooth_patch_ratio !== undefined ? `${(metrics.smooth_patch_ratio * 100).toFixed(1)}%` : 'N/A',
      desc: 'Spatial residual variance flatness across patches',
      isAbnormal: metrics.smooth_patch_ratio !== undefined && metrics.smooth_patch_ratio > 0.25,
      statusText: metrics.smooth_patch_ratio !== undefined && metrics.smooth_patch_ratio > 0.25 ? 'Diffusion Flatness' : 'Natural Grain'
    },
    {
      id: 'r_squared',
      label: 'Power Law Fit (R²)',
      value: metrics.r_squared !== undefined ? `${metrics.r_squared}` : 'N/A',
      desc: 'Adherence to natural 1/f^α optical spectral decay',
      isAbnormal: metrics.r_squared !== undefined && metrics.r_squared < 0.90,
      statusText: metrics.r_squared !== undefined && metrics.r_squared >= 0.90 ? 'Conforms to Optics' : 'Distorted Decay'
    }
  ];

  return (
    <div className="space-y-4 animate-reveal-stagger-2">
      
      {/* Verdict Hero Banner */}
      <div className={`glass-panel rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden text-center transition-all duration-500 animate-reveal ${
        isAi ? 'border-rose-500/40 shadow-neon-rose' : 'border-emerald-500/40 shadow-neon-emerald'
      }`}>
        
        {/* Ambient Glow Aura */}
        <div className={`absolute inset-0 opacity-10 pointer-events-none transition-all duration-700 ${
          isAi ? 'bg-rose-500' : 'bg-emerald-500'
        }`} />

        <div className="relative z-10 flex flex-col items-center">
          
          {/* Circular SVG Gauge */}
          <div className="w-32 h-32 rounded-full relative mb-4 flex items-center justify-center">
            <svg height={radius * 2} width={radius * 2} className="rotate-[-90deg]">
              <circle
                stroke="#1e293b"
                fill="transparent"
                strokeWidth={strokeWidth}
                r={normalizedRadius}
                cx={radius}
                cy={radius}
              />
              <circle
                stroke={isAi ? '#f43f5e' : '#10b981'}
                fill="transparent"
                strokeWidth={strokeWidth}
                strokeDasharray={`${circumference} ${circumference}`}
                style={{ strokeDashoffset }}
                strokeLinecap="round"
                r={normalizedRadius}
                cx={radius}
                cy={radius}
                className="transition-all duration-700 ease-out"
              />
            </svg>
            <div className="absolute flex flex-col items-center justify-center">
              <span className={`font-display font-black text-2xl tracking-tighter ${
                isAi ? 'text-rose-400 text-glow-rose' : 'text-emerald-400 text-glow-emerald'
              }`}>
                {animatedProb}<span className="text-sm font-normal">%</span>
              </span>
              <span className="label-caps text-[8px] text-slate-400">
                PROBABILITY
              </span>
            </div>
          </div>

          {/* Verdict Title & Subtitle */}
          <h1 className={`font-display font-black text-xl tracking-wider mb-1.5 uppercase ${
            isAi ? 'text-rose-400 text-glow-rose' : 'text-emerald-400 text-glow-emerald'
          }`}>
            {isAi ? 'SYNTHETIC DETECTED' : 'AUTHENTIC SENSOR'}
          </h1>
          
          <p className="font-mono text-xs text-slate-400 uppercase tracking-wider mb-2">
            {isAi ? 'High Probability Generative Source' : 'Optical Lens Physics Verified'}
          </p>

          <div className="flex items-center gap-2 text-[11px] font-mono text-slate-300">
            <span>Confidence:</span>
            <strong className="text-white px-2 py-0.5 rounded bg-slate-900 border border-slate-700 font-bold">
              {animatedConf}%
            </strong>
          </div>
        </div>
      </div>

      {/* Diagnostic Forensic Telemetry HUD */}
      {Object.keys(metrics).length > 0 && (
        <div className="glass-panel rounded-lg p-5 border border-cyan-500/20 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="label-caps text-cyan-400 text-xs flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5" />
              Diagnostic Forensic Telemetry
            </h3>
            <span className="label-caps text-[9px] text-slate-500">
              2D-FFT & SPATIAL RESIDUALS
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {telemetryItems.map((item) => (
              <div
                key={item.id}
                className={`p-3 rounded border text-xs font-mono transition-all ${
                  item.isAbnormal
                    ? 'bg-rose-950/30 border-rose-500/35 text-slate-200'
                    : 'bg-slate-900/60 border-slate-800 text-slate-300'
                }`}
              >
                <div className="flex items-center justify-between gap-1 mb-1">
                  <span className="text-[11px] text-slate-400 truncate">{item.label}</span>
                  {item.isAbnormal ? (
                    <span className="label-caps text-[9px] text-rose-400 bg-rose-950/80 border border-rose-500/40 px-1.5 py-0.5 rounded shrink-0">
                      SPIKE
                    </span>
                  ) : (
                    <span className="label-caps text-[9px] text-emerald-400 bg-emerald-950/80 border border-emerald-500/40 px-1.5 py-0.5 rounded shrink-0">
                      NORMAL
                    </span>
                  )}
                </div>
                <p className={`font-display font-bold text-base ${item.isAbnormal ? 'text-rose-300' : 'text-slate-100'}`}>
                  {item.value}
                </p>
                <p className="text-[10px] text-slate-500 truncate mt-0.5" title={item.desc}>
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detected Forensic Artifact Flags */}
      <div className="glass-panel rounded-lg p-5 border border-cyan-500/20 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="label-caps text-cyan-400 text-xs flex items-center gap-2">
            <Activity className="w-3.5 h-3.5" />
            Detected Forensic Cues & Artifact Flags
          </h3>
          <span className="label-caps text-[9px] text-slate-500">
            {artifact_flags.length} ANOMALIES IDENTIFIED
          </span>
        </div>

        {artifact_flags.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {artifact_flags.map((flag, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2.5 p-2.5 rounded bg-slate-900/70 border border-slate-800 text-xs font-mono text-slate-200"
              >
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                <div className="truncate">
                  <p className="font-semibold text-slate-200">{flag}</p>
                  <p className="text-[10px] text-slate-400">Structural frequency anomaly</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-3 rounded bg-emerald-950/20 border border-emerald-500/30 text-xs font-mono text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>No anomalous generative fingerprints or periodic upsampling grids identified.</span>
          </div>
        )}
      </div>

      {/* 2D-FFT Spectrum Visualizer Card */}
      <div className="glass-panel rounded-lg p-5 border border-violet-500/20 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
          <div>
            <h3 className="label-caps text-violet-400 text-xs flex items-center gap-2">
              <Radio className="w-3.5 h-3.5" />
              2D Fast Fourier Transform (FFT) Spectrum
            </h3>
            <p className="text-[11px] font-mono text-slate-400 mt-0.5">
              Log-magnitude frequency domain with high-pass circular filter (Inferno Colormap)
            </p>
          </div>
          <span className="label-caps text-[9px] text-violet-400 bg-violet-950/60 border border-violet-500/30 px-2 py-0.5 rounded w-fit">
            20 · log(|F| + 10⁻⁸)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center pt-1">
          {/* Spectrum Image Display */}
          <div className="relative aspect-square max-w-[200px] mx-auto rounded overflow-hidden border border-violet-500/40 bg-black shadow-neon-violet group">
            {fft_spectrum_base64 ? (
              <img
                src={fft_spectrum_base64}
                alt="2D-FFT Spectrum Visualization"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-slate-500 text-xs font-mono">
                No Spectrum Data
              </div>
            )}
            
            {/* Center High-Pass Circle Indicator */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-10 h-10 rounded-full border border-dashed border-cyan-400/50 bg-cyan-950/20 flex items-center justify-center">
                <span className="text-[7px] font-mono text-cyan-300">DC Cut</span>
              </div>
            </div>

            <div className="absolute bottom-1.5 left-1.5 bg-black/80 backdrop-blur-md px-1.5 py-0.5 rounded label-caps text-[8px] text-violet-300 border border-violet-500/30">
              Inferno (224x224)
            </div>
          </div>

          {/* Spectral Analysis Breakdown */}
          <div className="space-y-2 text-xs font-mono">
            <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-1">
              <span className="label-caps text-slate-400 text-[9px] block">High-Frequency Grid Residuals</span>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                {isAi
                  ? '⚠️ High energy spikes detected in outer radial rings, corresponding to transpose-convolution or latent patch upsamplers.'
                  : '✅ Smooth exponential decay from center to high frequencies conforming to natural optical sensor physics.'}
              </p>
            </div>

            <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-1">
              <span className="label-caps text-slate-400 text-[9px] block">Spatial vs Frequency Embedding Ratio</span>
              <div className="flex items-center justify-between text-[10px] text-slate-300 pt-0.5">
                <span>ConvNeXt: <strong>768-D</strong></span>
                <span>2D-FFT CNN: <strong>256-D</strong></span>
              </div>
              <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden flex">
                <div className="bg-cyan-400 h-full" style={{ width: '75%' }} />
                <div className="bg-violet-400 h-full" style={{ width: '25%' }} />
              </div>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
