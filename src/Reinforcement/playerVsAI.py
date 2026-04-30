import os
import sys
import numpy as np


# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from GameBoard.GameBoard import Connect4Board
# from brains.ZeroBrain import ZeroBrain        
# from players.ZeroPlayer import ZeroPlayer    


current_file = os.path.abspath(__file__) 
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
src_path = os.path.join(root_path, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from GameBoard.GameBoard import Connect4Board
from Reinforcement.brains.ZeroBrain import ZeroBrain        
from Reinforcement.players.ZeroPlayer import ZeroPlayer

def play():
    
    print("Loading AI Model ...")
    zero_ai = ZeroBrain(8) 
    ai_player = ZeroPlayer(zero_ai)
    
    # setup game with human as player 1 and AI as player 2
    human_player = 1
    board = Connect4Board(first_player=human_player)

    print("\n--- Game Start! ---")
    
    while not board.isEnd:
        print(f"\nRound No: {board.round}")
        board.showBoard()
        
        if board.current_turn == human_player:
            # human move
            try:
                move = int(input(f"Your Turn (Player {human_player}). Enter column (0-6): "))
                if move not in board.validAction():
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Please enter a number between 0 and 6.")
                continue
        else:
            # AI move
            print("AI is thinking...")
            # tau=0 for deterministic move (best move), tau>0 for more exploration 
            move, policy = ai_player.act(board, tau=0)
            print(f"AI chose column: {move}")
            # print(f"AI Confidence: {np.round(policy, 3)}") # To see the confidence of each move (optional)

        board.insertColumn(move)

    # End
    board.showBoard()
    if board.winner == 0:
        print("\n--- Game Over: DRAW! ---")
    else:
        winner_name = "You" if board.winner == human_player else "AI"
        print(f"\n--- Game Over: {winner_name} WON! ---")

if __name__ == "__main__":
    play()