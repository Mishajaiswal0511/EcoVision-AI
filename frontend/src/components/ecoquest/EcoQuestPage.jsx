import React, { useState } from 'react';
import { ECOQUEST_QUESTIONS } from '../../constants/ecoquestData';
import {
  getGamificationState,
  getDailyQuestions,
  getRandomPracticeQuestions,
  recordChallengeCompletion,
} from '../../services/ecoquestStorage';
import EcoQuestHeader from './EcoQuestHeader';
import EcoQuestCard from './EcoQuestCard';
import EcoQuestResults from './EcoQuestResults';
import { History, Calendar, X } from 'lucide-react';

export default function EcoQuestPage({ onBackToDetector }) {
  const [gamificationState, setGamificationState] = useState(getGamificationState);
  const [questions, setQuestions] = useState(() => getDailyQuestions(ECOQUEST_QUESTIONS));
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [selectedOptionId, setSelectedOptionId] = useState(null);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [completionResults, setCompletionResults] = useState(null);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  const currentQuestion = questions[currentIndex];

  // Option selection
  const handleSelectOption = (optionId) => {
    setSelectedOptionId(optionId);
  };

  // Submit and lock answer
  const handleSubmitAnswer = () => {
    if (!selectedOptionId || !currentQuestion) return;

    setUserAnswers((prev) => ({
      ...prev,
      [currentQuestion.id]: selectedOptionId,
    }));
    setIsSubmitted(true);
  };

  // Move to next question or complete
  const handleNextQuestion = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex((prev) => prev + 1);
      setSelectedOptionId(null);
      setIsSubmitted(false);
    } else {
      // Complete Challenge!
      // Calculate final score
      let score = 0;
      questions.forEach((q) => {
        const chosenId = userAnswers[q.id];
        const correctOpt = q.options.find((o) => o.isCorrect);
        if (chosenId === correctOpt?.id) {
          score += 1;
        }
      });

      const results = recordChallengeCompletion(score, questions.length, userAnswers);
      setCompletionResults(results);
      setGamificationState(getGamificationState());
      setIsCompleted(true);
    }
  };

  // Reset to play again with a freshly randomized diverse set of questions
  const handlePlayAgain = () => {
    // Collect previous question IDs so the user gets fresh unseen questions on practice
    const previousIds = questions.map((q) => q.id);
    const freshQuestions = getRandomPracticeQuestions(ECOQUEST_QUESTIONS, previousIds);
    setQuestions(freshQuestions);
    setCurrentIndex(0);
    setUserAnswers({});
    setSelectedOptionId(null);
    setIsSubmitted(false);
    setIsCompleted(false);
    setCompletionResults(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header with Stats & Ribbon */}
        <EcoQuestHeader
          gamificationState={gamificationState}
          onShowHistory={() => setIsHistoryOpen(!isHistoryOpen)}
          isHistoryOpen={isHistoryOpen}
        />

        {/* History Modal / Drawer */}
        {isHistoryOpen && (
          <div className="bg-white rounded-3xl p-6 shadow-xl border border-slate-200 animate-fadeIn space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2 text-slate-800 font-bold">
                <History className="w-5 h-5 text-emerald-600" />
                <span>Challenge History & Attempts</span>
              </div>
              <button
                onClick={() => setIsHistoryOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {gamificationState.history.length === 0 ? (
              <p className="text-xs text-slate-500 py-4 text-center">No challenges completed yet. Complete today's quest to build history!</p>
            ) : (
              <div className="divide-y divide-slate-100 max-h-60 overflow-y-auto pr-1">
                {gamificationState.history.map((item, idx) => (
                  <div key={idx} className="py-2.5 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-3">
                      <Calendar className="w-4 h-4 text-slate-400" />
                      <div>
                        <span className="font-semibold text-slate-800">{item.date}</span>
                        {item.isReplay && (
                          <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-slate-100 text-slate-600">
                            Practice
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-bold text-slate-900">
                        Score: {item.score}/{item.total}
                      </span>
                      <span className="font-bold text-emerald-600">
                        +{item.pointsEarned} pts
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Main Quiz Flow or Results */}
        {!isCompleted ? (
          currentQuestion && (
            <EcoQuestCard
              question={currentQuestion}
              currentIndex={currentIndex}
              totalQuestions={questions.length}
              selectedOptionId={selectedOptionId}
              isSubmitted={isSubmitted}
              onSelectOption={handleSelectOption}
              onSubmitAnswer={handleSubmitAnswer}
              onNextQuestion={handleNextQuestion}
            />
          )
        ) : (
          <EcoQuestResults
            score={
              questions.filter((q) => {
                const chosenId = userAnswers[q.id];
                const correctOpt = q.options.find((o) => o.isCorrect);
                return chosenId === correctOpt?.id;
              }).length
            }
            totalQuestions={questions.length}
            pointsAwarded={completionResults?.pointsAwarded ?? 0}
            totalPoints={gamificationState.points}
            streak={gamificationState.streak}
            isReplay={completionResults?.isReplay}
            newlyUnlockedBadges={completionResults?.newlyUnlockedBadges}
            questions={questions}
            userAnswers={userAnswers}
            onPlayAgain={handlePlayAgain}
            onBackToApp={onBackToDetector}
          />
        )}
      </div>
    </div>
  );
}
