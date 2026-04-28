import os
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GameBoard.GameBoard import Connect4Board
from brains.ZeroBrain import ZeroBrain        
from players.ZeroPlayer import ZeroPlayer    

def play():
    
    print("Loading AI Model v6...")
    zero_ai = ZeroBrain(4) 
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
            # ใช้ tau=0 เพื่อให้ AI เลือกท่าที่มั่นใจที่สุด (Best Move) ไม่สุ่มเหมือนตอนเทรน
            move, policy = ai_player.act(board, tau=0)
            print(f"AI chose column: {move}")
            # print(f"AI Confidence: {np.round(policy, 3)}") # เปิดเพื่อดูความมั่นใจในแต่ละช่อง

        board.insertColumn(move)

    # จบเกม
    board.showBoard()
    if board.winner == 0:
        print("\n--- Game Over: DRAW! ---")
    else:
        winner_name = "You" if board.winner == human_player else "AI"
        print(f"\n--- Game Over: {winner_name} WON! ---")

if __name__ == "__main__":
    play()