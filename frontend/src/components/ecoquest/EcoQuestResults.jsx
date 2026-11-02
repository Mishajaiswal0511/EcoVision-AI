import React from 'react';
import { Trophy, Award, CheckCircle2, XCircle, RotateCcw, ChevronDown, ChevronUp, ArrowLeft, Sparkles } from 'lucide-react';
import { BADGE_DEFINITIONS } from '../../services/ecoquestStorage';

export default function EcoQuestResults({
  score,
  totalQuestions,
  pointsAwarded,
  totalPoints: _totalPoints,
  streak: _streak,
  isReplay,
  newlyUnlockedBadges,
  questions,
  userAnswers,
  onPlayAgain,
  onBackToApp,
}) {
  const [showReview, setShowReview] = React.useState(false);

  // Personalized encouragement message
  let encouragement = {
    title: 'Sustainability Champion! 🌟',
    desc: 'You demonstrated an exceptional grasp of materials, contaminants, and proper disposal pathways.',
    color: 'text-emerald-700 bg-emerald-50 border-emerald-200',
  };

  if (score === 4) {
    encouragement = {
      title: 'Terrific Effort! 🌿',
      desc: 'Almost perfect! You caught the tricky contaminants and understand municipal sorting rules very well.',
      color: 'text-teal-700 bg-teal-50 border-teal-200',
    };
  } else if (score === 3) {
    encouragement = {
      title: 'Solid Eco Citizen! 🍃',
      desc: 'Good understanding! Reviewing the material science cards will help master tricky edge cases like food grease and broken glass.',
      color: 'text-amber-700 bg-amber-50 border-amber-200',
    };
  } else if (score < 3) {
    encouragement = {
      title: 'Keep Exploring & Learning! 🌱',
      desc: 'Waste sorting is full of counter-intuitive rules. Check the answer breakdown below to learn how sorting facilities handle tricky items.',
      color: 'text-slate-700 bg-slate-100 border-slate-200',
    };
  }

  return (
    <div className="bg-white rounded-3xl shadow-xl border border-slate-100 p-6 sm:p-10 max-w-3xl mx-auto space-y-8 animate-fadeIn">
      {/* Trophy & Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex p-4 rounded-3xl bg-gradient-to-tr from-emerald-500 to-teal-400 text-white shadow-lg shadow-emerald-500/30 mb-2">
          <Trophy className="w-12 h-12" />
        </div>
        <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Challenge Completed!
        </h2>
        <p className="text-sm text-slate-500">
          Daily EcoQuest Waste Sorting Assessment
        </p>
      </div>

      {/* Encouragement Card */}
      <div className={`p-5 rounded-2xl border text-center space-y-1 ${encouragement.color}`}>
        <p className="font-bold text-base">{encouragement.title}</p>
        <p className="text-xs sm:text-sm opacity-90">{encouragement.desc}</p>
      </div>

      {/* Score and Points Summary Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {/* Final Score */}
        <div className="bg-slate-50 p-4 rounded-2xl border border-slate-200 text-center">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Final Score</p>
          <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1">
            {score} <span className="text-sm font-normal text-slate-500">/ {totalQuestions}</span>
          </p>
        </div>

        {/* Correct Answers */}
        <div className="bg-emerald-50 p-4 rounded-2xl border border-emerald-200 text-center">
          <p className="text-xs font-semibold text-emerald-700 uppercase tracking-wider">Correct</p>
          <p className="text-2xl sm:text-3xl font-extrabold text-emerald-700 mt-1">
            {score}
          </p>
        </div>

        {/* Incorrect Answers */}
        <div className="bg-rose-50 p-4 rounded-2xl border border-rose-200 text-center">
          <p className="text-xs font-semibold text-rose-700 uppercase tracking-wider">Mistakes</p>
          <p className="text-2xl sm:text-3xl font-extrabold text-rose-700 mt-1">
            {totalQuestions - score}
          </p>
        </div>

        {/* Points Earned */}
        <div className="bg-amber-50 p-4 rounded-2xl border border-amber-200 text-center">
          <p className="text-xs font-semibold text-amber-800 uppercase tracking-wider">Eco Points</p>
          <p className="text-2xl sm:text-3xl font-extrabold text-amber-700 mt-1">
            +{pointsAwarded}
          </p>
        </div>
      </div>

      {/* Replay Notice if already completed today */}
      {isReplay && (
        <div className="p-3.5 rounded-xl bg-emerald-50/80 border border-emerald-200 text-xs text-emerald-800 text-center font-medium">
          🎯 Practice Session Bonus: You earned <strong>+{pointsAwarded} Eco Points</strong> (2 pts per correct answer) for sharpening your sorting skills!
        </div>
      )}

      {/* Unlocked Badges Celebration */}
      {newlyUnlockedBadges && newlyUnlockedBadges.length > 0 && (
        <div className="p-4 rounded-2xl bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-emerald-800 uppercase tracking-wider">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            <span>New Badge Unlocked!</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {newlyUnlockedBadges.map((badgeId) => {
              const b = BADGE_DEFINITIONS.find((item) => item.id === badgeId);
              if (!b) return null;
              return (
                <div
                  key={badgeId}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold bg-gradient-to-r ${b.color} text-slate-950 shadow-sm flex items-center gap-1.5`}
                >
                  <Award className="w-4 h-4" />
                  <span>{b.title}</span> — <span className="font-normal opacity-90">{b.description}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
        <button
          onClick={onPlayAgain}
          className="w-full sm:w-auto px-6 py-3 rounded-xl font-bold text-sm text-white bg-emerald-600 hover:bg-emerald-700 shadow-md shadow-emerald-600/20 transition-all flex items-center justify-center gap-2 cursor-pointer"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Practice Again</span>
        </button>

        <button
          onClick={() => setShowReview(!showReview)}
          className="w-full sm:w-auto px-6 py-3 rounded-xl font-bold text-sm text-slate-700 bg-slate-100 hover:bg-slate-200 transition-all flex items-center justify-center gap-2 cursor-pointer"
        >
          <span>{showReview ? 'Hide Answer Review' : 'Review Answers'}</span>
          {showReview ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        <button
          onClick={onBackToApp}
          className="w-full sm:w-auto px-6 py-3 rounded-xl font-bold text-sm text-slate-600 hover:text-slate-900 border border-slate-200 hover:bg-slate-50 transition-all flex items-center justify-center gap-2 cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Detector</span>
        </button>
      </div>

      {/* Answer Review Section */}
      {showReview && (
        <div className="mt-8 pt-6 border-t border-slate-200 space-y-6 animate-fadeIn">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            Detailed Answers & Explanations
          </h3>

          <div className="space-y-4">
            {questions.map((q, idx) => {
              const userAnsId = userAnswers[q.id];
              const selectedOpt = q.options.find((o) => o.id === userAnsId);
              const correctOpt = q.options.find((o) => o.isCorrect);
              const isUserCorrect = selectedOpt?.isCorrect;

              return (
                <div
                  key={q.id}
                  className="p-5 rounded-2xl border border-slate-200 bg-slate-50/70 space-y-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="w-6 h-6 rounded-full bg-slate-200 text-slate-700 text-xs font-bold flex items-center justify-center">
                        {idx + 1}
                      </span>
                      <h4 className="text-sm font-bold text-slate-900">{q.title}</h4>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {isUserCorrect ? (
                        <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-700 bg-emerald-100 px-2.5 py-0.5 rounded-full">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Correct
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-xs font-bold text-rose-700 bg-rose-100 px-2.5 py-0.5 rounded-full">
                          <XCircle className="w-3.5 h-3.5" /> Incorrect
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="text-xs text-slate-600">{q.scenario}</p>

                  <div className="space-y-1 text-xs">
                    <p className="text-slate-500">
                      Your answer:{' '}
                      <strong className={isUserCorrect ? 'text-emerald-700' : 'text-rose-700'}>
                        {selectedOpt?.id}) {selectedOpt?.text}
                      </strong>
                    </p>
                    {!isUserCorrect && (
                      <p className="text-slate-700">
                        Correct answer:{' '}
                        <strong className="text-emerald-700">
                          {correctOpt?.id}) {correctOpt?.text}
                        </strong>
                      </p>
                    )}
                  </div>

                  <p className="text-xs text-slate-600 bg-white p-3 rounded-xl border border-slate-200/80 leading-relaxed">
                    💡 <strong>Insight:</strong> {q.explanation}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
