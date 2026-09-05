import React, { useState, useEffect, useRef } from 'react';
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

  // AbortController reference to safely cancel any previous in-flight request
  const activeAbortControllerRef = useRef(null);

  const animatedHeroProb = useCountUp(analysisResult?.ai_probability, 850);
  const animatedHeroConf = useCountUp(analysisResult?.confidence, 850);

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setErrorMsg(null);
    setAnalysisResult(null);

    // Revoke previous preview URL to avoid browser memory accumulation
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    // Trigger fresh forensic analysis
    await runForensicAnalysis(file);
  };

  const runForensicAnalysis = async (file) => {
    // 1. Abort any previous pending request before launching a new one
    if (activeAbortControllerRef.current) {
      try {
        activeAbortControllerRef.current.abort();
      } catch (abortErr) {
        console.warn('[Spectra Client] Previous request abort:', abortErr);
      }
      activeAbortControllerRef.current = null;
    }

    // 2. Create fresh AbortController for this request
    const controller = new AbortController();
    activeAbortControllerRef.current = controller;

    setIsAnalyzing(true);
    setErrorMsg(null);
    setAnalysisResult(null);

    // 3. Create a fresh FormData instance for every upload
    const formData = new FormData();
    formData.append('file', file);

    const startTime = performance.now();
    const endpoint = API_ENDPOINTS.ANALYZE;

    try {
      console.log(`[Spectra Client] Sending analyze request to ${endpoint} for ${file.name} (${file.size} bytes)`);

      const response = await axios.post(endpoint, formData, {
        timeout: 60000,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        }
      });

      const elapsed = Math.round(performance.now() - startTime);
      console.log(`[Spectra Client] Analysis completed in ${elapsed}ms:`, response.data);

      if (response.data) {
        setAnalysisResult(response.data);
        setSessionId(Math.floor(100000 + Math.random() * 900000));
        setErrorMsg(null);
      }
    } catch (err) {
      // If request was canceled by a subsequent upload, silently return
      if (axios.isCancel(err) || err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        console.log(`[Spectra Client] Request for ${file.name} canceled due to a newer upload.`);
        return;
      }

      const elapsed = Math.round(performance.now() - startTime);
      console.error(`[Spectra Client] Analysis Request Failed after ${elapsed}ms:`, {
        code: err.code,
        message: err.message,
        status: err.response?.status,
        statusText: err.response?.statusText,
        data: err.response?.data,
        endpoint
      });

      // Distinguish specific failure modes accurately
      let friendlyError = '';
      if (err.code === 'ECONNABORTED' || err.message?.toLowerCase().includes('timeout')) {
        friendlyError = `Request Timeout (${elapsed}ms): The backend server did not finish processing within 60 seconds. The image may be too large or the server is processing high load.`;
      } else if (err.code === 'ERR_NETWORK' || (!err.response && err.request)) {
        friendlyError = `Network Error (${err.code || 'CONNECTION_FAILED'}): Unable to connect to backend at ${API_BASE_URL}. Check network connection or verify service status.`;
      } else if (err.response) {
        const status = err.response.status;
        const resData = err.response.data;
        const detail = (typeof resData === 'object' && resData !== null)
          ? (resData.detail || resData.message || JSON.stringify(resData))
          : (typeof resData === 'string' ? resData : null);

        if (detail) {
          friendlyError = `[Server HTTP ${status}] ${detail}`;
        } else if (status === 413) {
          friendlyError = `[Server HTTP 413] Payload Too Large: The uploaded image exceeds the 50MB maximum upload limit.`;
        } else if (status === 400) {
          friendlyError = `[Server HTTP 400] Bad Request: The image format is invalid or could not be decoded.`;
        } else if (status >= 500) {
          friendlyError = `[Server HTTP ${status}] Internal Server Error: The backend encountered an error during inference (${err.response.statusText || 'Error'}).`;
        } else {
          friendlyError = `[Server HTTP ${status}] Request failed with status ${status}: ${err.response.statusText || 'Unknown'}`;
        }
      } else {
        friendlyError = err.message || 'An unexpected client-side error occurred during analysis.';
      }

      setErrorMsg(friendlyError);
    } finally {
      // Only reset analyzing state if this was the active controller
      if (activeAbortControllerRef.current === controller) {
        setIsAnalyzing(false);
        activeAbortControllerRef.current = null;
      }
    }
  };

  const handleReset = () => {
    if (activeAbortControllerRef.current) {
      try {
        activeAbortControllerRef.current.abort();
      } catch (e) {}
      activeAbortControllerRef.current = null;
    }
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
        onTabChange={setActiveTab}
        hasResult={!!analysisResult}
        isAnalyzing={isAnalyzing}
      />

      {/* Main Forensic Dashboard Grid */}
      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 pb-24 md:pb-10 flex flex-col gap-6">
        
        {/* Error Alert Banner with Detailed Context */}
        {errorMsg && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/40 backdrop-blur-md p-4 flex items-start gap-3 shadow-[0_0_20px_rgba(244,63,94,0.15)] animate-reveal">
            <AlertOctagon className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="label-caps text-xs text-rose-300 font-semibold mb-1">
                Forensic Analysis Notice
              </h4>
              <p className="text-xs text-rose-200/90 font-mono leading-relaxed">
                {errorMsg}
              </p>
            </div>
            <button 
              onClick={() => setErrorMsg(null)}
              className="text-xs text-rose-400 hover:text-rose-200 font-mono underline ml-2"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Dynamic Multi-Tab Content Engine */}
        {activeTab === 'scan' && (
          <div className="flex flex-col gap-6">
            
            {/* Top Row: Scanner Ingestion Zone */}
            <ScannerZone 
              onFileSelect={handleFileSelect}
              onError={(msg) => setErrorMsg(msg)}
              isAnalyzing={isAnalyzing}
              selectedPreviewUrl={previewUrl}
            />

            {/* Middle Row: Hero Forensic Verdict HUD (When Result Ready) */}
            {analysisResult && (
              <section className="glass-panel rounded-lg p-5 sm:p-6 border border-cyan-500/30 relative overflow-hidden animate-reveal">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                  
                  {/* Verdict & Badge */}
                  <div className="flex items-center gap-4">
                    <div className={`w-14 h-14 rounded-lg flex items-center justify-center border shadow-xl ${
                      analysisResult.verdict === 'AI-GENERATED'
                        ? 'bg-rose-500/10 border-rose-500/40 text-rose-400 shadow-[0_0_20px_rgba(244,63,94,0.2)]'
                        : 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.2)]'
                    }`}>
                      {analysisResult.verdict === 'AI-GENERATED' ? (
                        <ShieldAlert className="w-8 h-8 animate-pulse" />
                      ) : (
                        <ShieldCheck className="w-8 h-8" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="label-caps text-[11px] text-slate-400">Classification Verdict</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800/80 text-cyan-300 border border-slate-700">
                          ID: #{sessionId || '741829'}
                        </span>
                      </div>
                      <h2 className={`text-xl sm:text-2xl font-extrabold tracking-wider font-mono ${
                        analysisResult.verdict === 'AI-GENERATED' ? 'text-rose-400' : 'text-emerald-400'
                      }`}>
                        {analysisResult.verdict}
                      </h2>
                    </div>
                  </div>

                  {/* Dual Gauge Metrics */}
                  <div className="grid grid-cols-2 gap-4 w-full md:w-auto">
                    <div className="bg-[#060e20]/80 border border-slate-800 rounded-lg p-3 min-w-[140px]">
                      <span className="label-caps text-[10px] text-slate-400 block mb-1">AI Probability</span>
                      <div className="text-2xl font-bold font-mono text-cyan-400">
                        {animatedHeroProb}%
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full mt-2 overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-500 ${
                            analysisResult.ai_probability >= 50 ? 'bg-rose-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${analysisResult.ai_probability}%` }}
                        />
                      </div>
                    </div>

                    <div className="bg-[#060e20]/80 border border-slate-800 rounded-lg p-3 min-w-[140px]">
                      <span className="label-caps text-[10px] text-slate-400 block mb-1">Model Confidence</span>
                      <div className="text-2xl font-bold font-mono text-indigo-400">
                        {animatedHeroConf}%
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full mt-2 overflow-hidden">
                        <div 
                          className="h-full bg-indigo-500 transition-all duration-500"
                          style={{ width: `${analysisResult.confidence}%` }}
                        />
                      </div>
                    </div>
                  </div>

                </div>
              </section>
            )}

            {/* Split Visual Comparison & Heatmap Inspection */}
            {analysisResult && (
              <SplitViewer 
                originalUrl={previewUrl}
                fftUrl={analysisResult.fft_spectrum_base64}
                gradcamUrl={analysisResult.gradcam_heatmap_base64}
              />
            )}

            {/* Forensic Detail Metrics Table */}
            {analysisResult && (
              <ForensicMetrics 
                metrics={analysisResult.metrics}
                flags={analysisResult.artifact_flags}
                aiProbability={analysisResult.ai_probability}
              />
            )}

          </div>
        )}

      </main>

    </div>
  );
}
