import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  AlertOctagon, 
  Sparkles, 
  Binary, 
  Info,
  Shield,
  Layers,
  Radio,
  Eye,
  ShieldCheck,
  ShieldAlert,
  Cpu,
  Target
} from 'lucide-react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ScannerZone from './components/ScannerZone';
import SplitViewer from './components/SplitViewer';
import ForensicMetrics from './components/ForensicMetrics';
import { API_BASE_URL, API_ENDPOINTS } from './config';

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

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [activeTab, setActiveTab] = useState('scan');

  const animatedHeroProb = useCountUp(analysisResult?.ai_probability, 850);
  const animatedHeroConf = useCountUp(analysisResult?.confidence, 850);

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setErrorMsg(null);
    setAnalysisResult(null);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    
    // Trigger analysis automatically
    await runForensicAnalysis(file);
  };

  const runForensicAnalysis = async (file) => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(API_ENDPOINTS.ANALYZE, formData, {
        timeout: 30000
      });

      if (response.data) {
        setAnalysisResult(response.data);
        setSessionId(Math.floor(100000 + Math.random() * 900000));
      }
    } catch (err) {
      console.error('Forensic Analysis Error:', err);
      let message = `Failed to analyze image. Please ensure the backend API service is running on ${API_BASE_URL}.`;
      if (err.response?.data?.detail) {
        message = err.response.data.detail;
      }
      setErrorMsg(message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setSessionId(null);
    setErrorMsg(null);
  };

  return (
    <div className="min-h-screen bg-[#0b1326] text-[#dae2fd] font-sans grid-bg-animated flex flex-col overflow-x-hidden">
      
      {/* Top App Bar */}
      <Header onReset={handleReset} analysisResult={analysisResult} />

      {/* Side HUD Dock (Desktop & Mobile) */}
      <Sidebar 
        currentTab={activeTab} 
        onTabChange={(tab) => setActiveTab(tab)} 
        onEmergencyLock={handleReset} 
      />

      {/* Main Forensic Content Area */}
      <main className="flex-1 pt-20 pb-16 px-4 sm:px-6 md:pl-24 max-w-7xl w-full mx-auto space-y-6">
        
        {/* Error Notification Banner */}
        {errorMsg && (
          <div className="p-4 rounded-lg bg-rose-950/80 border border-rose-500/50 text-rose-200 text-xs font-mono flex items-center justify-between gap-3 shadow-lg shadow-rose-950/40 animate-reveal">
            <div className="flex items-center gap-2.5">
              <AlertOctagon className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-rose-400 hover:text-rose-200 font-bold px-2 py-1"
            >
              ✕
            </button>
          </div>
        )}

        {/* Top Hero Grid: Target Ingestion + Standby or Active Verdict Banner */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
          
          {/* Ingestion Dropzone (2 Cols) */}
          <div className="lg:col-span-2 flex flex-col">
            <ScannerZone
              onFileSelect={handleFileSelect}
              onError={(msg) => setErrorMsg(msg)}
              isAnalyzing={isAnalyzing}
              selectedPreviewUrl={previewUrl}
            />
          </div>

          {/* Verdict Banner (1 Col) */}
          <div className="flex flex-col">
            {analysisResult ? (
              <div className={`glass-panel rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden text-center h-full transition-all duration-500 animate-reveal ${
                analysisResult.verdict === 'AI-GENERATED' 
                  ? 'border-rose-500/40 shadow-neon-rose' 
                  : 'border-emerald-500/40 shadow-neon-emerald'
              }`}>
                <div className={`absolute inset-0 opacity-10 pointer-events-none transition-all duration-700 ${
                  analysisResult.verdict === 'AI-GENERATED' ? 'bg-rose-500' : 'bg-emerald-500'
                }`} />

                <div className="relative z-10 flex flex-col items-center">
                  <div className="w-28 h-28 rounded-full border-4 border-slate-800 relative mb-4 flex items-center justify-center shadow-inner">
                    <span className={`font-display font-black text-2xl tracking-tighter ${
                      analysisResult.verdict === 'AI-GENERATED' ? 'text-rose-400 text-glow-rose' : 'text-emerald-400 text-glow-emerald'
                    }`}>
                      {animatedHeroProb}<span className="text-sm font-normal">%</span>
                    </span>
                  </div>

                  <h1 className={`font-display font-black text-lg tracking-wider mb-1 uppercase ${
                    analysisResult.verdict === 'AI-GENERATED' ? 'text-rose-400 text-glow-rose' : 'text-emerald-400 text-glow-emerald'
                  }`}>
                    {analysisResult.verdict === 'AI-GENERATED' ? 'SYNTHETIC DETECTED' : 'AUTHENTIC SENSOR'}
                  </h1>

                  <p className="label-caps text-[10px] text-slate-400 uppercase tracking-widest mb-2">
                    {analysisResult.verdict === 'AI-GENERATED' ? 'High Probability Generative Source' : 'Natural Optical Physics'}
                  </p>

                  <div className="px-2.5 py-1 rounded bg-slate-900/90 border border-slate-800 text-[11px] font-mono text-slate-300">
                    Confidence: <strong className="text-slate-100">{animatedHeroConf}%</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="glass-panel rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden text-center h-full border-cyan-500/20 transition-all duration-300">
                <div className="w-24 h-24 rounded-full border border-dashed border-cyan-500/40 relative mb-4 flex items-center justify-center">
                  <Target className="w-8 h-8 text-cyan-400/60 animate-pulse" />
                </div>
                <h3 className="label-caps text-cyan-400 text-xs mb-1">
                  AWAITING TARGET EVIDENCE
                </h3>
                <p className="font-mono text-[11px] text-slate-400 max-w-[200px]">
                  Ingest suspect imagery to compute dual-stream 2D-FFT & spatial classification
                </p>
              </div>
            )}
          </div>

        </div>

        {/* Detailed Forensic Results Workspace */}
        {analysisResult && previewUrl && (
          <section className="space-y-5 pt-3 border-t border-cyan-500/20">
            <div className="flex items-center justify-between">
              <span className="font-display font-bold text-sm sm:text-base text-slate-100 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Forensic Inspection & Explainability Dossier
              </span>
              <span className="label-caps text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-500/30 px-2.5 py-1 rounded">
                SESSION ID #{sessionId || '000000'}
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
              
              {/* Left Column (7 Cols): Grad-CAM Split Viewer */}
              <div className="lg:col-span-7 space-y-4">
                <SplitViewer
                  rawImageUrl={previewUrl}
                  gradcamImageUrl={analysisResult.gradcam_heatmap_base64}
                  aiProbability={analysisResult.ai_probability}
                  verdict={analysisResult.verdict}
                />
              </div>

              {/* Right Column (5 Cols): Metrics & 2D-FFT Visualizer */}
              <div className="lg:col-span-5 space-y-4">
                <ForensicMetrics result={analysisResult} />
              </div>

            </div>
          </section>
        )}

        {/* Forensic Methodology Educational Brief */}
        <section id="methodology" className="glass-panel rounded-lg p-5 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-slate-300 font-display font-semibold text-xs sm:text-sm">
            <Info className="w-4 h-4 text-cyan-400" />
            Detection Methodology & Dual-Stream Architecture
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono text-slate-400">
            <div className="p-3.5 rounded bg-slate-900/60 border border-slate-800/90 space-y-1.5">
              <span className="label-caps text-cyan-400 block text-[10px]">1. Spatial Stream (ConvNeXt-Tiny)</span>
              <p className="text-[11px] leading-relaxed">Extracts 768-D hierarchical visual features, inspecting boundary blending, texture smoothness, and anatomical inconsistencies.</p>
            </div>

            <div className="p-3.5 rounded bg-slate-900/60 border border-slate-800/90 space-y-1.5">
              <span className="label-caps text-violet-400 block text-[10px]">2. Frequency Stream (2D-FFT CNN)</span>
              <p className="text-[11px] leading-relaxed">Computes log-magnitude spectra, exposing transpose-convolution checkerboards and latent upsampling grid spikes.</p>
            </div>

            <div className="p-3.5 rounded bg-slate-900/60 border border-slate-800/90 space-y-1.5">
              <span className="label-caps text-emerald-400 block text-[10px]">3. Grad-CAM Localization</span>
              <p className="text-[11px] leading-relaxed">Calculates gradient activations on stage-4 blocks to highlight the precise spatial regions triggering the synthetic verdict.</p>
            </div>
          </div>
        </section>

      </main>

      {/* Classified Data Footer (Stitch Spec) */}
      <footer className="md:pl-20 w-full bg-[#060e20]/95 border-t border-slate-800/80 py-4 px-6 sm:px-10 flex flex-col sm:flex-row justify-between items-center gap-3 font-mono text-[11px] text-slate-500 z-10 relative">
        <div className="label-caps text-slate-400 text-[10px]">
          © 2024 SPECTRA FORENSICS // CLASSIFIED SENSOR TELEMETRY
        </div>
        <div className="flex gap-4 sm:gap-6 text-[11px]">
          <a href="#methodology" className="text-slate-400 hover:text-cyan-400 transition-colors">Methodology</a>
          <span className="text-slate-700">•</span>
          <span className="text-slate-400">ConvNeXt-Tiny + 2D-FFT</span>
          <span className="text-slate-700">•</span>
          <span className="text-emerald-400 font-semibold">Node Online</span>
        </div>
      </footer>

    </div>
  );
}
