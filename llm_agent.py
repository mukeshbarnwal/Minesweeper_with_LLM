"""
LLM Agent for Minesweeper

An agentic loop where the LLM selects actions given only the textual board.
Runs until win/loss with no human input.
"""

import json
import time
import random
from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass

from minesweeper_engine import MinesweeperBoard, GameResult

try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    LLM_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for LLM agent."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    api_key: Optional[str] = None
    use_advanced_strategies: bool = True
    max_moves_per_game: int = 500


class LLMMinesweeperAgent:
    """
    LLM agent that plays Minesweeper autonomously.
    
    Features:
    - Agentic loop with no human input
    - Structured prompting with board analysis
    - Chain-of-thought reasoning
    - Fallback strategies for robustness
    - Comprehensive move history tracking
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM agent.
        
        Args:
            config: Configuration for the LLM agent
        """
        self.config = config or LLMConfig()
        
        if LLM_AVAILABLE and self.config.api_key:
            self.client = OpenAI(api_key=self.config.api_key)
        elif LLM_AVAILABLE:
            # Try to get from environment
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
                self.client = None
        else:
            self.client = None
    
    def query_llm(self, board: MinesweeperBoard, move_history: List[Tuple[int, int, str]] = None) -> Tuple[int, int, str]:
        """
        Ask the LLM for the next move with enhanced prompting.
        
        Args:
            board: Current board state
            move_history: List of previous moves for context
            
        Returns:
            Tuple of (row, col, action) where action is "open" or "flag"
        """
        if not self.client:
            raise RuntimeError("OpenAI client not available. Check API key configuration.")
        
        # Get board analysis
        analysis = board.get_board_analysis()
        safe_moves = board.get_safe_moves()
        forced_mines = board.get_forced_mines()
        
        # Build move history context
        history_text = ""
        if move_history and len(move_history) > 0:
            recent_moves = move_history[-5:]  # Last 5 moves
            history_text = "\nRecent moves:\n"
            for i, (r, c, action) in enumerate(recent_moves):
                history_text += f"  Move {len(move_history) - 4 + i}: {action} at ({r}, {c})\n"
        
        # Enhanced prompt with strategic guidance
        prompt = f"""
You are playing Minesweeper on a {board.R}x{board.C} board with {board.M} mines.

{analysis}

{safe_moves and f"RECOMMENDED: Open one of these guaranteed safe cells: {safe_moves[:3]}" or ""}
{forced_mines and f"RECOMMENDED: Flag these guaranteed mines: {forced_mines[:3]}" or ""}

{history_text}

Current board state:
{board.player_view_str()}

STRATEGY GUIDELINES:
1. ALWAYS prioritize guaranteed safe moves over guesses
2. Flag cells that are guaranteed to be mines
3. When no guaranteed moves exist, prefer cells with fewer adjacent numbers
4. Start with corners or edges if no obvious moves
5. Avoid cells adjacent to high numbers (6-8) unless forced

Think step-by-step:
1. Analyze the current board state
2. Identify any guaranteed safe moves
3. Identify any guaranteed mines
4. Assess risk for remaining cells
5. Choose the most logical move

Return your next move as JSON: {{"row": R, "col": C, "action": "open"|"flag"}}
Use 0-based indices. Choose the most logical move based on the analysis above.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert Minesweeper player and logical reasoner. 
                        You excel at pattern recognition and deductive reasoning. 
                        Always prioritize guaranteed safe moves over random guesses.
                        Think step-by-step about the board state before making your move.
                        Always respond with valid JSON."""
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            txt = response.choices[0].message.content
            
            # Enhanced parsing with better error handling
            try:
                # Try to extract JSON from the response
                if "```json" in txt:
                    json_part = txt.split("```json")[1].split("```")[0]
                elif "```" in txt:
                    json_part = txt.split("```")[1]
                else:
                    json_part = txt
                
                # Clean up the JSON string
                json_part = json_part.strip()
                if json_part.startswith("```"):
                    json_part = json_part[3:]
                if json_part.endswith("```"):
                    json_part = json_part[:-3]
                
                move = json.loads(json_part)
                r, c, action = int(move["row"]), int(move["col"]), move["action"]
                
                # Validate the move
                if not (0 <= r < board.R and 0 <= c < board.C):
                    raise ValueError("Invalid coordinates")
                if action not in ("open", "flag"):
                    raise ValueError("Invalid action")
                if board.view[r][c] not in ("□", "⚑"):
                    raise ValueError("Cell already opened")
                    
                return r, c, action
                
            except Exception as e:
                print(f"LLM response parsing failed: {e}")
                print(f"Raw response: {txt}")
                
                # Fallback to algorithmic strategy
                return self._get_fallback_move(board)
                
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return self._get_fallback_move(board)
    
    def _get_fallback_move(self, board: MinesweeperBoard) -> Tuple[int, int, str]:
        """
        Get a fallback move using algorithmic strategies when LLM fails.
        
        Args:
            board: Current board state
            
        Returns:
            Tuple of (row, col, action)
        """
        # 1. Check for guaranteed safe moves (highest priority)
        safe_moves = board.get_safe_moves()
        if safe_moves:
            r, c = safe_moves[0]
            return r, c, "open"
        
        # 2. Check for guaranteed mines
        forced_mines = board.get_forced_mines()
        if forced_mines:
            r, c = forced_mines[0]
            return r, c, "flag"
        
        # 3. Basic risk analysis
        best_cells = []
        min_risk = float('inf')
        
        for r in range(board.R):
            for c in range(board.C):
                if board.view[r][c] == "□":
                    adjacent_numbers = 0
                    adjacent_sum = 0
                    
                    for dr, dc in board.NEIGHB:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < board.R and 0 <= nc < board.C and 
                            board.view[nr][nc].isdigit()):
                            adjacent_numbers += 1
                            adjacent_sum += int(board.view[nr][nc])
                    
                    # Calculate risk score (lower is better)
                    if adjacent_numbers > 0:
                        risk_score = adjacent_sum / adjacent_numbers + adjacent_numbers * 0.5
                    else:
                        risk_score = 0.1  # Very low risk for isolated cells
                    
                    if risk_score < min_risk:
                        min_risk = risk_score
                        best_cells = [(r, c)]
                    elif risk_score == min_risk:
                        best_cells.append((r, c))
        
        if best_cells:
            r, c = random.choice(best_cells)
            return r, c, "open"
        
        # 4. Last resort: random unopened cell
        unopened = [(r, c) for r in range(board.R) for c in range(board.C)
                    if board.view[r][c] == "□"]
        if unopened:
            r, c = random.choice(unopened)
            return r, c, "open"
        
        # 5. No moves available (shouldn't happen)
        return None, None, None
    
    def play_game(self, rows: int, cols: int, mines: int, seed: Optional[int] = None) -> GameResult:
        """
        Play a complete game from start to finish.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            mines: Number of mines
            seed: Random seed for reproducibility
            
        Returns:
            GameResult with outcome and statistics
        """
        # Create board
        board = MinesweeperBoard(rows, cols, mines, rng=random.Random(seed))
        start_time = time.time()
        moves = 0
        move_history = []

        # Game loop
        while not board.finished and moves < self.config.max_moves_per_game:
            try:
                # Get next move from LLM
                r, c, action = self.query_llm(board, move_history)
                
                if r is None:  # No moves available
                    break
                
                # Execute move
                result = board.step(r, c, action)
                move_history.append((r, c, action))
                moves += 1
                
                # Check for game over
                if result == "hit-mine":
                    break
                    
            except Exception as e:
                print(f"Error during game: {e}")
                break

        # Calculate results
        duration = time.time() - start_time
        outcome = "win" if board.win else "loss"
        
        return GameResult(
            game_id=0,  # Will be set by caller
            rows=rows,
            cols=cols,
            mines=mines,
            moves=moves,
            outcome=outcome,
            duration_s=round(duration, 3),
            rng_seed=seed or 0
        )
    
    def play_multiple_games(self, n_games: int, rows: int, cols: int, mines: int) -> List[GameResult]:
        """
        Play multiple games and return results.
        
        Args:
            n_games: Number of games to play
            rows: Number of rows
            cols: Number of columns
            mines: Number of mines
            
        Returns:
            List of GameResult objects
        """
        results = []
        
        for game_id in range(1, n_games + 1):
            seed = random.randrange(2**32)
            result = self.play_game(rows, cols, mines, seed)
            result.game_id = game_id
            results.append(result)
            
            # Print progress
            wins = sum(1 for r in results if r.outcome == "win")
            print(f"Game {game_id}: {result.outcome.upper()} in {result.moves} moves "
                  f"({wins}/{game_id} wins, {wins/game_id*100:.1f}%)")
        
        return results


def create_llm_agent(api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> LLMMinesweeperAgent:
    """
    Factory function to create an LLM agent.
    
    Args:
        api_key: OpenAI API key (if None, will try environment variable)
        model: Model to use
        
    Returns:
        Configured LLMMinesweeperAgent instance
    """
    config = LLMConfig(
        model=model,
        api_key=api_key,
        temperature=0.0,
        use_advanced_strategies=True
    )
    return LLMMinesweeperAgent(config)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_llm_agent()
    
    # Play a single game
    result = agent.play_game(9, 9, 10)
    print(f"Game result: {result.outcome} in {result.moves} moves ({result.duration_s}s)")
    
    # Play multiple games
    results = agent.play_multiple_games(5, 9, 9, 10)
    
    # Print summary
    wins = sum(1 for r in results if r.outcome == "win")
    avg_moves = sum(r.moves for r in results) / len(results)
    avg_duration = sum(r.duration_s for r in results) / len(results)
    
    print(f"\nSummary:")
    print(f"Win rate: {wins}/{len(results)} ({wins/len(results)*100:.1f}%)")
    print(f"Average moves: {avg_moves:.1f}")
    print(f"Average duration: {avg_duration:.3f}s") 