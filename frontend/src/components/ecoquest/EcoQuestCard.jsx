import React from 'react';
import { CheckCircle2, XCircle, Info, Lightbulb, AlertCircle, ArrowRight } from 'lucide-react';

export default function EcoQuestCard({
  question,
  currentIndex,
  totalQuestions,
  selectedOptionId,
  isSubmitted,
  onSelectOption,
  onSubmitAnswer,
  onNextQuestion,
}) {
  const selectedOption = question.options.find((o) => o.id === selectedOptionId);
  const isCorrect = selectedOption?.isCorrect;

  return (
    <div className="bg-white rounded-3xl shadow-xl border border-slate-100 overflow-hidden transition-all duration-300">
      {/* Top Progress Bar */}
      <div className="bg-slate-50 border-b border-slate-100 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-bold uppercase tracking-wider">
            Question {currentIndex + 1} of {totalQuestions}
          </span>
          <span className="text-xs font-medium text-slate-500">
            Category: <strong className="text-slate-800">{question.categoryLabel}</strong>
          </span>
        </div>
        <div className="w-32 sm:w-48 bg-slate-200 h-2 rounded-full overflow-hidden">
          <div
            className="bg-emerald-500 h-full rounded-full transition-all duration-500 ease-out"
            style={{ width: `${((currentIndex + (isSubmitted ? 1 : 0)) / totalQuestions) * 100}%` }}
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="p-6 sm:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Real Waste Item Image Container */}
          <div className="lg:col-span-5 flex flex-col gap-3">
            <div className="relative aspect-4/3 rounded-2xl overflow-hidden bg-slate-100 border border-slate-200 shadow-inner group">
              <img
                src={question.image}
                alt={question.title}
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                onError={(e) => {
                  // Fallback styling if local image fails
                  e.target.style.display = 'none';
                  e.target.parentNode.classList.add('flex', 'items-center', 'justify-center', 'p-6', 'text-slate-400');
                  e.target.parentNode.innerText = 'Waste Photograph: ' + question.title;
                }}
              />
              <div className="absolute top-3 left-3 px-3 py-1 rounded-lg bg-black/60 backdrop-blur-md text-white text-xs font-medium">
                Real Waste Photograph
              </div>
            </div>

            {/* Scenario Text */}
            <div className="p-4 rounded-2xl bg-emerald-50/60 border border-emerald-100/80">
              <h2 className="text-base font-bold text-slate-900 mb-1 flex items-center gap-2">
                {question.title}
              </h2>
              <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                {question.scenario}
              </p>
            </div>
          </div>

          {/* Options & Action Area */}
          <div className="lg:col-span-7 flex flex-col justify-between h-full">
            <div>
              <p className="text-sm font-semibold text-slate-700 mb-3">
                Where should this waste item be disposed or sorted?
              </p>

              {/* 4 Options Grid */}
              <div className="space-y-3">
                {question.options.map((opt) => {
                  const isThisSelected = selectedOptionId === opt.id;
                  let btnStyle = 'border-slate-200 hover:border-emerald-300 hover:bg-emerald-50/30 text-slate-700';

                  if (isSubmitted) {
                    if (opt.isCorrect) {
                      btnStyle = 'border-emerald-500 bg-emerald-50 text-emerald-900 ring-2 ring-emerald-500/30 font-medium';
                    } else if (isThisSelected && !opt.isCorrect) {
                      btnStyle = 'border-rose-500 bg-rose-50 text-rose-900 ring-2 ring-rose-500/30';
                    } else {
                      btnStyle = 'border-slate-100 text-slate-400 bg-slate-50/50 opacity-60';
                    }
                  } else if (isThisSelected) {
                    btnStyle = 'border-emerald-600 bg-emerald-50/80 text-emerald-950 font-semibold ring-2 ring-emerald-600/20 shadow-sm';
                  }

                  return (
                    <button
                      key={opt.id}
                      onClick={() => !isSubmitted && onSelectOption(opt.id)}
                      disabled={isSubmitted}
                      className={`w-full text-left p-4 rounded-2xl border-2 transition-all flex items-start gap-3.5 ${btnStyle} cursor-pointer disabled:cursor-default`}
                    >
                      <span
                        className={`w-7 h-7 rounded-xl flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${
                          isSubmitted && opt.isCorrect
                            ? 'bg-emerald-600 text-white'
                            : isSubmitted && isThisSelected && !opt.isCorrect
                            ? 'bg-rose-600 text-white'
                            : isThisSelected
                            ? 'bg-emerald-600 text-white'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {opt.id}
                      </span>
                      <div className="flex-1 text-sm leading-snug pt-0.5">
                        {opt.text}
                      </div>

                      {/* Result Icons after submission */}
                      {isSubmitted && opt.isCorrect && (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
                      )}
                      {isSubmitted && isThisSelected && !opt.isCorrect && (
                        <XCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Bottom Actions: Submit or Next */}
            <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-end">
              {!isSubmitted ? (
                <button
                  onClick={onSubmitAnswer}
                  disabled={!selectedOptionId}
                  className="w-full sm:w-auto px-7 py-3 rounded-xl font-bold text-sm text-white bg-emerald-600 hover:bg-emerald-700 shadow-md shadow-emerald-600/20 disabled:opacity-40 disabled:pointer-events-none transition-all flex items-center justify-center gap-2"
                >
                  Confirm & Lock Answer
                </button>
              ) : (
                <button
                  onClick={onNextQuestion}
                  className="w-full sm:w-auto px-7 py-3 rounded-xl font-bold text-sm text-white bg-slate-900 hover:bg-emerald-900 shadow-md transition-all flex items-center justify-center gap-2 group"
                >
                  <span>{currentIndex < totalQuestions - 1 ? 'Next Question' : 'View Results'}</span>
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Immediate Post-Submission Explanation & Guidance */}
        {isSubmitted && (
          <div className="mt-8 pt-6 border-t border-slate-100 animate-fadeIn space-y-4">
            {/* Answer banner */}
            <div
              className={`p-4 rounded-2xl flex items-start gap-3.5 border ${
                isCorrect
                  ? 'bg-emerald-50 text-emerald-950 border-emerald-200'
                  : 'bg-rose-50 text-rose-950 border-rose-200'
              }`}
            >
              {isCorrect ? (
                <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-6 h-6 text-rose-600 shrink-0 mt-0.5" />
              )}
              <div className="space-y-1">
                <p className="font-bold text-sm sm:text-base">
                  {isCorrect ? 'Outstanding! +10 Eco Points earned.' : 'Not quite right this time.'}
                </p>
                <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
                  {selectedOption?.feedback}
                </p>
              </div>
            </div>

            {/* Why it works / Science explanation */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
                <div className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  <Info className="w-4 h-4 text-emerald-600" />
                  <span>The Material Science</span>
                </div>
                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                  {question.explanation}
                </p>
              </div>

              <div className="p-4 rounded-2xl bg-amber-50/70 border border-amber-200/80">
                <div className="flex items-center gap-2 text-xs font-bold text-amber-800 uppercase tracking-wider mb-1.5">
                  <Lightbulb className="w-4 h-4 text-amber-600" />
                  <span>Practical Recycling Tip</span>
                </div>
                <p className="text-xs sm:text-sm text-amber-900 leading-relaxed">
                  {question.disposalTip}
                </p>
              </div>
            </div>

            {/* Municipal Disclaimer */}
            <div className="flex items-center gap-2 text-[11px] text-slate-500 italic px-2">
              <AlertCircle className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              <span>{question.localDisclaimer}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
