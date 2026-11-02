import React from 'react';
import { Award, Flame, Trophy, Sparkles, Crown, CheckCircle2, History } from 'lucide-react';
import { BADGE_DEFINITIONS } from '../../services/ecoquestStorage';

const iconMap = {
  Sparkles,
  Trophy,
  Flame,
  Crown,
  Award,
};

export default function EcoQuestHeader({
  gamificationState,
  onShowHistory,
  isHistoryOpen,
}) {
  const { points, streak, badges, history, isTodayCompleted } = gamificationState;

  // Best score in history
  const bestScore = history.length > 0 ? Math.max(...history.map((h) => h.score)) : 0;

  return (
    <div className="w-full bg-gradient-to-br from-emerald-900 via-teal-900 to-slate-900 text-white rounded-3xl p-6 sm:p-8 shadow-2xl border border-emerald-500/20 mb-8 relative overflow-hidden">
      {/* Background organic glow elements */}
      <div className="absolute -top-24 -right-24 w-72 h-72 bg-emerald-500/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 -left-24 w-72 h-72 bg-teal-400/10 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        {/* Title & Tagline */}
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-400/30 text-emerald-300 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5 text-emerald-300" />
            Interactive Waste Masterclass
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white flex items-center gap-3">
            EcoQuest <span className="text-emerald-400 font-light">Challenge</span>
          </h1>
          <p className="text-sm sm:text-base text-emerald-100/80 max-w-xl leading-relaxed">
            Test your real-world sorting skills against contamination traps, greasy boxes, and municipal rules. Earn Eco Points and build your streak!
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full lg:w-auto">
          {/* Total Eco Points */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-3.5 border border-white/10 flex flex-col justify-between">
            <div className="flex items-center justify-between text-emerald-300 text-xs font-medium">
              <span>Eco Points</span>
              <Award className="w-4 h-4 text-emerald-300" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white tracking-tight">
              {points} <span className="text-xs text-emerald-400 font-normal">pts</span>
            </div>
          </div>

          {/* Daily Streak */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-3.5 border border-white/10 flex flex-col justify-between">
            <div className="flex items-center justify-between text-amber-300 text-xs font-medium">
              <span>Daily Streak</span>
              <Flame className="w-4 h-4 text-amber-400" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white tracking-tight">
              {streak} <span className="text-xs text-amber-300 font-normal">days</span>
            </div>
          </div>

          {/* Best Score */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-3.5 border border-white/10 flex flex-col justify-between">
            <div className="flex items-center justify-between text-teal-300 text-xs font-medium">
              <span>Best Score</span>
              <Trophy className="w-4 h-4 text-teal-300" />
            </div>
            <div className="mt-2 text-2xl font-bold text-white tracking-tight">
              {bestScore} <span className="text-xs text-teal-300 font-normal">/ 5</span>
            </div>
          </div>

          {/* Today Status */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-3.5 border border-white/10 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-300 text-xs font-medium">
              <span>Today</span>
              <CheckCircle2 className={`w-4 h-4 ${isTodayCompleted ? 'text-emerald-400' : 'text-slate-400'}`} />
            </div>
            <div className="mt-2 text-sm font-bold text-white tracking-tight">
              {isTodayCompleted ? (
                <span className="text-emerald-300 font-medium text-xs sm:text-sm">Completed ✓</span>
              ) : (
                <span className="text-amber-300 font-medium text-xs sm:text-sm">Ready to Play</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Badges Drawer / Ribbon */}
      <div className="mt-6 pt-5 border-t border-white/10 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2 overflow-x-auto py-1 max-w-full">
          <span className="text-xs font-medium text-slate-300 whitespace-nowrap mr-1">Badges:</span>
          {BADGE_DEFINITIONS.map((b) => {
            const isUnlocked = badges.includes(b.id);
            const Icon = iconMap[b.icon] || Award;
            return (
              <div
                key={b.id}
                title={`${b.title}: ${b.description} ${isUnlocked ? '(Unlocked)' : '(Locked)'}`}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-all ${
                  isUnlocked
                    ? `bg-gradient-to-r ${b.color} text-slate-950 font-bold shadow-md`
                    : 'bg-white/5 text-slate-400 border border-white/5 opacity-50'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">{b.title}</span>
              </div>
            );
          })}
        </div>

        {/* History Toggle Button */}
        {history.length > 0 && (
          <button
            onClick={onShowHistory}
            className="inline-flex items-center gap-1.5 text-xs text-emerald-300 hover:text-emerald-200 transition-colors bg-white/5 hover:bg-white/10 px-3 py-1.5 rounded-xl border border-white/10"
          >
            <History className="w-3.5 h-3.5" />
            <span>{isHistoryOpen ? 'Hide History' : 'View History'}</span>
          </button>
        )}
      </div>
    </div>
  );
}
