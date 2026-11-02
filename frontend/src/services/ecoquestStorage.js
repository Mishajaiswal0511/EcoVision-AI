/**
 * EcoQuest Gamification & Storage Manager
 * Handles: Eco Points, streaks, badges, daily challenge selection, and attempt history.
 */

const STORAGE_KEYS = {
  POINTS: 'ecoquest_points',
  STREAK: 'ecoquest_streak',
  LAST_PLAYED_DATE: 'ecoquest_last_date',
  HISTORY: 'ecoquest_history',
  BADGES: 'ecoquest_badges',
  DAILY_COMPLETED_PREFIX: 'ecoquest_done_',
};

export const BADGE_DEFINITIONS = [
  {
    id: 'first_challenge',
    title: 'First Sort',
    description: 'Completed your first EcoQuest challenge.',
    icon: 'Sparkles',
    color: 'from-emerald-400 to-teal-500',
  },
  {
    id: 'perfect_score',
    title: 'Zero Waste Master',
    description: 'Achieved a perfect 5/5 score on a challenge.',
    icon: 'Trophy',
    color: 'from-amber-400 to-yellow-500',
  },
  {
    id: 'streak_3',
    title: 'Eco Enthusiast',
    description: 'Maintained a 3-day sorting streak.',
    icon: 'Flame',
    color: 'from-orange-400 to-red-500',
  },
  {
    id: 'streak_7',
    title: 'Sustainability Champion',
    description: 'Maintained a 7-day sorting streak.',
    icon: 'Crown',
    color: 'from-purple-500 to-indigo-600',
  },
  {
    id: 'points_100',
    title: 'Centurion Sorter',
    description: 'Accumulated 100+ Eco Points.',
    icon: 'Award',
    color: 'from-cyan-400 to-blue-500',
  },
];

/**
 * Returns formatted local date string YYYY-MM-DD
 */
export function getTodayKey() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/**
 * Deterministically pick 5 questions for today using the date as a seed
 */
export function getDailyQuestions(allQuestions) {
  if (!allQuestions || allQuestions.length === 0) return [];
  const today = getTodayKey();
  
  // Simple integer hash of the date string
  let hash = 0;
  for (let i = 0; i < today.length; i++) {
    hash = (hash << 5) - hash + today.charCodeAt(i);
    hash |= 0;
  }
  hash = Math.abs(hash);

  // Shuffle copy using seeded LCG
  const pool = [...allQuestions];
  const selected = [];
  const count = Math.min(5, pool.length);

  for (let i = 0; i < count; i++) {
    const idx = (hash + i * 7) % pool.length;
    selected.push(pool[idx]);
    pool.splice(idx, 1);
  }

  return selected;
}

/**
 * Generates a fresh randomized set of 5 questions for Practice / Play Again sessions,
 * avoiding immediate repeats where possible.
 */
export function getRandomPracticeQuestions(allQuestions, excludeIds = []) {
  if (!allQuestions || allQuestions.length === 0) return [];
  
  // Prioritize questions not in excludeIds
  let available = allQuestions.filter((q) => !excludeIds.includes(q.id));
  if (available.length < 5) {
    available = [...allQuestions];
  }

  // Fisher-Yates shuffle
  const shuffled = [...available];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled.slice(0, Math.min(5, shuffled.length));
}

/**
 * Load user gamification state from localStorage
 */
export function getGamificationState() {
  try {
    const points = parseInt(localStorage.getItem(STORAGE_KEYS.POINTS) || '0', 10);
    const streak = parseInt(localStorage.getItem(STORAGE_KEYS.STREAK) || '0', 10);
    const lastDate = localStorage.getItem(STORAGE_KEYS.LAST_PLAYED_DATE) || null;
    const history = JSON.parse(localStorage.getItem(STORAGE_KEYS.HISTORY) || '[]');
    const badges = JSON.parse(localStorage.getItem(STORAGE_KEYS.BADGES) || '[]');
    const today = getTodayKey();
    const isTodayCompleted = localStorage.getItem(`${STORAGE_KEYS.DAILY_COMPLETED_PREFIX}${today}`) === 'true';

    return {
      points,
      streak,
      lastDate,
      history,
      badges,
      isTodayCompleted,
      todayKey: today,
    };
  } catch (e) {
    console.error('Error loading EcoQuest state from localStorage:', e);
    return {
      points: 0,
      streak: 0,
      lastDate: null,
      history: [],
      badges: [],
      isTodayCompleted: false,
      todayKey: getTodayKey(),
    };
  }
}

/**
 * Calculate difference in calendar days
 */
function getDayDiff(dateStr1, dateStr2) {
  if (!dateStr1 || !dateStr2) return 999;
  const d1 = new Date(dateStr1);
  const d2 = new Date(dateStr2);
  const diffTime = Math.abs(d2.getTime() - d1.getTime());
  return Math.round(diffTime / (1000 * 60 * 60 * 24));
}

/**
 * Record challenge completion and calculate rewards
 */
export function recordChallengeCompletion(score, totalQuestions, _answers) {
  try {
    const today = getTodayKey();
    const currentState = getGamificationState();
    const alreadyCompletedToday = currentState.isTodayCompleted;

    let pointsAwarded = 0;
    let newStreak = currentState.streak;
    const newBadges = [...currentState.badges];

    if (!alreadyCompletedToday) {
      // First completion of the day: Main Daily Challenge reward (10 pts per correct answer)
      pointsAwarded = score * 10;

      // Update streak
      if (!currentState.lastDate) {
        newStreak = 1;
      } else {
        const dayDiff = getDayDiff(currentState.lastDate, today);
        if (dayDiff === 1) {
          newStreak += 1; // Consecutive day!
        } else if (dayDiff === 0) {
          // Same day, streak intact
        } else {
          newStreak = 1; // Broken streak, reset to 1
        }
      }

      // Mark today's main challenge complete
      localStorage.setItem(`${STORAGE_KEYS.DAILY_COMPLETED_PREFIX}${today}`, 'true');
      localStorage.setItem(STORAGE_KEYS.LAST_PLAYED_DATE, today);
    } else {
      // Practice / Replay Mode:
      // Award Bonus Practice points (2 pts per correct answer, up to +10 pts per practice session)
      // This rewards the user for practicing while preserving daily challenge integrity!
      pointsAwarded = score * 2;
    }

    const updatedPoints = currentState.points + pointsAwarded;

    // Check Badges
    const unlockBadge = (badgeId) => {
      if (!newBadges.includes(badgeId)) {
        newBadges.push(badgeId);
      }
    };

    unlockBadge('first_challenge');
    if (score === totalQuestions && totalQuestions > 0) {
      unlockBadge('perfect_score');
    }
    if (newStreak >= 3) {
      unlockBadge('streak_3');
    }
    if (newStreak >= 7) {
      unlockBadge('streak_7');
    }
    if (updatedPoints >= 100) {
      unlockBadge('points_100');
    }

    // Save history entry
    const historyEntry = {
      date: today,
      timestamp: new Date().toISOString(),
      score,
      total: totalQuestions,
      pointsEarned: pointsAwarded,
      isReplay: alreadyCompletedToday,
    };
    const updatedHistory = [historyEntry, ...currentState.history].slice(0, 30); // keep last 30

    // Persist all updates
    localStorage.setItem(STORAGE_KEYS.POINTS, updatedPoints.toString());
    localStorage.setItem(STORAGE_KEYS.STREAK, newStreak.toString());
    localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(updatedHistory));
    localStorage.setItem(STORAGE_KEYS.BADGES, JSON.stringify(newBadges));

    return {
      pointsAwarded,
      totalPoints: updatedPoints,
      streak: newStreak,
      badges: newBadges,
      history: updatedHistory,
      isReplay: alreadyCompletedToday,
      newlyUnlockedBadges: newBadges.filter((b) => !currentState.badges.includes(b)),
    };
  } catch (e) {
    console.error('Error saving EcoQuest results:', e);
    return {
      pointsAwarded: score * 10,
      totalPoints: score * 10,
      streak: 1,
      badges: ['first_challenge'],
      history: [],
      isReplay: false,
      newlyUnlockedBadges: [],
    };
  }
}
