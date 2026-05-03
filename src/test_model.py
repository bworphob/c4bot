import os
import sys
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from GameBoard.GameBoard import Connect4Board
from Reinforcement.playerVsAI_TRT import TRTBrainWrapper

def test_model():
    # Load model
    model_path = os.path.join(current_dir, 'Reinforcement', 'Models', 'model_v4.engine')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    brain = TRTBrainWrapper(model_path)
    print("Model loaded successfully.")

    # Test cases
    test_boards = [
        ("Empty board", Connect4Board(first_player=1)),
        ("Board with some moves", create_test_board()),
    ]

    for name, board in test_boards:
        print(f"\n--- Testing: {name} ---")
        board.showBoard()
        policy, value = brain.predict(board)
        print(f"Policy: {policy}")
        print(f"Value: {value}")

def create_test_board():
    board = Connect4Board(first_player=1)
    board.insertColumn(0)  # P1
    board.insertColumn(4)  # P2 (ลงไกลๆ)
    board.insertColumn(1)  # P1
    board.insertColumn(5)  # P2 (ลงไกลๆ)
    board.insertColumn(2)  # P1 -> ตอนนี้ P1 จ่อชนะที่คอลัมน์ 3
    return board

if __name__ == "__main__":
    test_model()