import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Download, 
  Plus, 
  Settings, 
  Bell, 
  Cpu, 
  ShieldCheck, 
  AlertTriangle 
} from 'lucide-react';
import { API_ENDPOINTS } from '../config';

export default function Header({ onReset, analysisResult }) {
  const [backendStatus, setBackendStatus] = useState({ online: false, latency: null });

  useEffect(() => {
    const checkHealth = async () => {
      const startTime = performance.now();
      try {
        const res = await axios.get(API_ENDPOINTS.HEALTH, { timeout: 4000 });
        const endTime = performance.now();
        if (res.status === 200) {
          setBackendStatus({
            online: true,
            latency: Math.round(endTime - startTime)
          });
        }
      } catch (err) {
        setBackendStatus({ online: false, latency: null });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleExportJSON = () => {
    if (!analysisResult) return;
    const blob = new Blob([JSON.stringify(analysisResult, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `SPECTRA_Forensic_Dossier_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-[#0b1326]/90 backdrop-blur-xl border-b border-slate-800/80 shadow-[0_0_20px_rgba(6,182,212,0.12)] flex justify-between items-center px-4 md:px-8 py-3.5">
      
      {/* Left: Brand Identity & Network Telemetry */}
      <div className="flex items-center gap-3 sm:gap-4 md:pl-16">
        <div className="flex items-center gap-2">
          <span className="font-display font-black text-base sm:text-lg tracking-tighter text-cyan-400 uppercase text-glow-cyan select-none">
            SPECTRA FORENSICS
          </span>
          <span className="label-caps text-[9px] text-slate-500 hidden sm:inline-block">
            // V2.4 ACTIVE
          </span>
        </div>

        {/* API Health Pill */}
        <div className="flex items-center gap-2 px-2.5 py-1 bg-slate-900/90 rounded border border-slate-800 text-xs font-mono">
          <div className={`w-2 h-2 rounded-full ${
            backendStatus.online ? 'bg-cyan-400 animate-pulse shadow-[0_0_8px_#06b6d4]' : 'bg-rose-500'
          }`} />
          <span className="text-[11px] text-cyan-300 font-medium">
            {backendStatus.online ? `API Online (${backendStatus.latency}ms)` : 'API Offline'}
          </span>
        </div>

        {/* Model Architecture Badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 bg-slate-900/70 rounded border border-slate-800 text-[11px] font-mono text-slate-400">
          <Cpu className="w-3.5 h-3.5 text-violet-400" />
          <span>ConvNeXt-Tiny + 2D-FFT</span>
        </div>
      </div>

      {/* Right: Actions, Dossier Export, Operator */}
      <div className="flex items-center gap-2.5 sm:gap-4">
        
        {/* Export JSON Report */}
        {analysisResult && (
          <button
            onClick={handleExportJSON}
            className="label-caps text-slate-300 hover:text-cyan-400 transition-colors flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-800 hover:border-cyan-500/40 bg-slate-900/60 text-xs"
            title="Export full JSON analysis report"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Export</span>
          </button>
        )}

        {/* Cyber Button: New Scan */}
        <button
          onClick={onReset}
          className="cyber-button label-caps px-3.5 sm:px-4 py-1.5 sm:py-2 rounded text-[11px] uppercase font-bold flex items-center gap-1.5 shadow-sm"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New Scan</span>
        </button>

        {/* Utility Icons */}
        <div className="hidden sm:flex items-center gap-2.5 text-slate-400">
          <Settings className="w-4 h-4 cursor-pointer hover:text-cyan-400 transition-colors" />
          <Bell className="w-4 h-4 cursor-pointer hover:text-cyan-400 transition-colors" />
        </div>

        {/* Operator Profile Avatar */}
        <div className="w-8 h-8 rounded-full bg-slate-900 border border-cyan-400/60 overflow-hidden shadow-neon-cyan flex items-center justify-center shrink-0">
          <span className="font-mono text-[10px] font-bold text-cyan-300">OP-1</span>
        </div>

      </div>

    </header>
  );
}
