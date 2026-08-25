import React from 'react';
import { Activity, FileText, HelpCircle, User, Home } from 'lucide-react';

export default function Header({ onOpenExport, hasPrediction }) {
  return (
    <header className="bg-[#0f172a] text-white px-4 py-2.5 flex items-center justify-between shadow-md border-b border-slate-800">
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2.5">
          <div className="bg-blue-600 p-1.5 rounded-lg flex items-center justify-center text-white shadow-lg shadow-blue-500/30">
            <Activity className="w-5 h-5 stroke-[2.5]" />
          </div>
          <span className="font-bold text-lg tracking-wide text-white font-['Inter']">
            Shilla BoneAge <span className="text-blue-400">AI</span>
          </span>
        </div>

        <nav className="flex items-center space-x-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700 text-xs font-medium">
          <button className="px-3 py-1.5 rounded-md text-slate-300 hover:text-white hover:bg-slate-700 transition flex items-center space-x-1.5">
            <Home className="w-3.5 h-3.5" />
            <span>Home</span>
          </button>
          <button className="px-3 py-1.5 rounded-md bg-blue-600 text-white font-semibold shadow-sm flex items-center space-x-1.5">
            <Activity className="w-3.5 h-3.5" />
            <span>Bone Age Analysis</span>
          </button>
        </nav>
      </div>

      <div className="flex items-center space-x-3 text-xs">
        <button
          onClick={onOpenExport}
          disabled={!hasPrediction}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md border transition ${
            hasPrediction
              ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'
              : 'bg-slate-900 text-slate-600 border-slate-800 cursor-not-allowed'
          }`}
        >
          <FileText className="w-3.5 h-3.5 text-blue-400" />
          <span>Report</span>
        </button>
        <button className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-slate-800 text-slate-200 border border-slate-700">
          <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
          <span>Demo</span>
        </button>
        <div className="h-4 w-[1px] bg-slate-700" />
        <div className="flex items-center space-x-2 pl-1">
          <div className="w-7 h-7 rounded-full bg-blue-600/30 border border-blue-500/50 flex items-center justify-center text-blue-400">
            <User className="w-4 h-4" />
          </div>
          <span className="font-medium text-slate-200 text-xs">Research Demo</span>
        </div>
      </div>
    </header>
  );
}
