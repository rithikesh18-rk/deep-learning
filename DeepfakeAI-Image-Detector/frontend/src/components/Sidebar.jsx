import React from 'react';
import { 
  LayoutDashboard, 
  Binary, 
  FileCode, 
  Network, 
  Archive, 
  ShieldAlert, 
  BookOpen, 
  HelpCircle,
  Cpu
} from 'lucide-react';

export default function Sidebar({ currentTab = 'scan', onTabChange, onEmergencyLock }) {
  return (
    <>
      {/* Desktop Left HUD Dock */}
      <aside className="hidden md:flex flex-col fixed left-0 top-0 h-full w-16 hover:w-56 z-40 bg-[#060e20]/90 backdrop-blur-xl border-r border-cyan-500/20 pt-20 pb-6 transition-all duration-300 group overflow-hidden shadow-2xl">
        
        {/* Workspace Title on Expand */}
        <div className="px-4 mb-6 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <div className="font-display text-sm font-bold text-cyan-400 tracking-wider">SPECTRA // HUD</div>
          <div className="font-mono text-[10px] text-slate-400">V2.4 FORENSIC NODE</div>
        </div>

        {/* Navigation Items */}
        <div className="flex flex-col gap-1.5 w-full">
          <button
            onClick={() => onTabChange && onTabChange('scan')}
            className={`flex items-center gap-3.5 px-5 py-3 text-left transition-all w-full whitespace-nowrap font-mono text-xs ${
              currentTab === 'scan'
                ? 'bg-cyan-500/15 text-cyan-300 border-r-2 border-cyan-400 shadow-sm shadow-cyan-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <Binary className="w-5 h-5 shrink-0 text-cyan-400" />
            <span className="opacity-0 group-hover:opacity-100 transition-opacity font-semibold tracking-wider">
              DEEPFAKE SCAN
            </span>
          </button>

          <button
            onClick={() => onTabChange && onTabChange('telemetry')}
            className="flex items-center gap-3.5 px-5 py-3 text-left transition-all w-full whitespace-nowrap font-mono text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
          >
            <LayoutDashboard className="w-5 h-5 shrink-0 text-violet-400" />
            <span className="opacity-0 group-hover:opacity-100 transition-opacity font-semibold tracking-wider">
              TELEMETRY HUD
            </span>
          </button>

          <button
            onClick={() => onTabChange && onTabChange('spectrum')}
            className="flex items-center gap-3.5 px-5 py-3 text-left transition-all w-full whitespace-nowrap font-mono text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
          >
            <FileCode className="w-5 h-5 shrink-0 text-sky-400" />
            <span className="opacity-0 group-hover:opacity-100 transition-opacity font-semibold tracking-wider">
              2D-FFT METADATA
            </span>
          </button>

          <button
            onClick={() => onTabChange && onTabChange('network')}
            className="flex items-center gap-3.5 px-5 py-3 text-left transition-all w-full whitespace-nowrap font-mono text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
          >
            <Network className="w-5 h-5 shrink-0 text-emerald-400" />
            <span className="opacity-0 group-hover:opacity-100 transition-opacity font-semibold tracking-wider">
              NEURAL NETWORK
            </span>
          </button>
        </div>

        {/* Bottom Utility Items */}
        <div className="mt-auto flex flex-col gap-2 w-full px-2">
          {onEmergencyLock && (
            <button
              onClick={onEmergencyLock}
              className="mx-2 py-2 border border-rose-500/50 text-rose-400 rounded-md font-mono text-[10px] tracking-wider uppercase opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-950/40 flex items-center justify-center gap-1.5"
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>CLEAR SESSION</span>
            </button>
          )}

          <div className="pt-2 border-t border-slate-800/80 flex flex-col gap-1">
            <a
              href="#methodology"
              className="flex items-center gap-3.5 px-3 py-2 text-slate-400 hover:text-cyan-300 font-mono text-xs whitespace-nowrap transition-colors"
            >
              <BookOpen className="w-4 h-4 shrink-0" />
              <span className="opacity-0 group-hover:opacity-100 transition-opacity text-[11px]">Methodology</span>
            </a>
            <div className="flex items-center gap-3.5 px-3 py-2 text-slate-400 font-mono text-xs whitespace-nowrap">
              <Cpu className="w-4 h-4 shrink-0 text-violet-400" />
              <span className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px] text-slate-500">ConvNeXt+FFT</span>
            </div>
          </div>
        </div>

      </aside>

      {/* Mobile Bottom HUD Bar */}
      <nav className="md:hidden fixed bottom-0 left-0 w-full z-40 bg-[#060e20]/95 backdrop-blur-xl border-t border-cyan-500/20 px-4 py-2 flex justify-around items-center">
        <button
          onClick={() => onTabChange && onTabChange('scan')}
          className="flex flex-col items-center gap-1 text-cyan-400 font-mono text-[10px]"
        >
          <Binary className="w-5 h-5" />
          <span>SCAN</span>
        </button>
        <button
          onClick={() => onTabChange && onTabChange('telemetry')}
          className="flex flex-col items-center gap-1 text-slate-400 hover:text-slate-200 font-mono text-[10px]"
        >
          <LayoutDashboard className="w-5 h-5" />
          <span>TELEMETRY</span>
        </button>
        <button
          onClick={() => onTabChange && onTabChange('spectrum')}
          className="flex flex-col items-center gap-1 text-slate-400 hover:text-slate-200 font-mono text-[10px]"
        >
          <FileCode className="w-5 h-5" />
          <span>2D-FFT</span>
        </button>
      </nav>
    </>
  );
}
