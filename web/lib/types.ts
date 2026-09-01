export type Range = { mean: number; p10: number; p50: number; p90: number };

export type Player = {
  name: string;
  rank: number | null;
  hand: string;
  age: number | null;
  matches_seen: number;
};

export type PlayerStats = {
  aces: Range;
  double_faults: Range;
  service_points: Range;
  service_points_won: Range;
  breaks: Range;
  break_points_faced: Range;
  break_points_saved: Range;
};

export type KalshiSide = {
  de_vig_probability: number | null;
  bid: number | null;
  ask: number | null;
  volume: number;
};

export type Forecast = {
  model_version: string;
  trained_at: string;
  training_cutoff: string;
  context: {
    tour: "ATP" | "WTA";
    best_of: number;
    round: string;
    match_date: string;
    court: string | null;
  };
  player1: Player;
  player2: Player;
  probabilities: {
    elo: number;
    point_model: number;
    machine_learning: number;
    stack: number;
    final: number;
    champion_component: string;
  };
  simulation: {
    n_sims: number;
    total_games: Range;
    duration_minutes: Range;
    p_tiebreak: number;
    expected_tiebreaks: number;
    p_deciding_set: number;
    p1_straight_sets: number;
    p2_straight_sets: number;
    set_score_distribution: Record<string, number>;
    top_exact_scores: { score: string; probability: number }[];
    total_games_probabilities: Record<string, number>;
    player1_stats: PlayerStats;
    player2_stats: PlayerStats;
  };
  winner: { name: string; probability: number };
  model_disagreement: number;
  data_quality: string[];
  kalshi?: {
    event_ticker?: string;
    observed_at?: string;
    player1?: KalshiSide;
    player2?: KalshiSide;
    model_minus_market_p1?: number | null;
    status?: string;
  };
};
