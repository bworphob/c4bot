import numpy as np
import copy

class Connect4Board:
    def __init__(self, first_player):
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.round = 1
        self.current_turn = first_player
        self.isEnd = False
        self.winner = 0

    def getBoard(self):
        return copy.deepcopy(self.board)

    def showBoard(self):
        print("\n" + "="*29)
        print(f" ROUND: {self.round:02d} | TURN: Player {self.current_turn}")
        print("-" * 29)
        print("  0   1   2   3   4   5   6")
        
        for row in self.board:
            symbols = []
            for cell in row:
                if cell == 1:
                    symbols.append('X')      # player 1
                elif cell == 2:
                    symbols.append('O')      # player 2 (AI)
                else:
                    symbols.append('.')      # empty
            
            
            row_str = " | ".join(symbols)
            print("| " + row_str + " |")
            
        print("-" * 29)
        print("="*29 + "\n")

    # def getStateAsPlayer(self):
    #     CH1 = np.zeros((6,7),dtype=np.int8) 
    #     CH1[self.board == 1] = 1
    #     CH2 = np.zeros((6,7),dtype=np.int8)
    #     CH2[self.board == 2] = 1
        
        
    #     CH3 = np.zeros((6,7),dtype=np.int8)
    #     if self.current_turn == 2:
    #         CH3 = np.ones((6,7),dtype=np.int8)
            
    #     board_stack = np.stack((CH1, CH2, CH3), axis=0)
    #     return board_stack
    
    def getStateAsPlayer(self):
        me = self.current_turn
        opp = 1 if me == 2 else 2

        # Layer 0: current player's pieces
        CH1 = np.zeros((6,7), dtype=np.float32)
        CH1[self.board == me] = 1.0

        # Layer 1: opponent's pieces
        CH2 = np.zeros((6,7), dtype=np.float32)
        CH2[self.board == opp] = 1.0

        # Layer 2: turn indicator
        CH3 = np.zeros((6,7), dtype=np.float32)
        if me == 2:  # set one-hot indicator when AI is current player
            CH3 = np.ones((6,7), dtype=np.float32)

        return np.stack((CH1, CH2, CH3), axis=0)

    def topRowInColumn(self, col):
        if col > 6 or col < 0:
            return -1

        for i in range(5, -1, -1):
            if self.board[i, col] == 0:
                return i

        return -1

    def checkEndGameFromInsert(self, row, col):
        # directions: horizontal, vertical, diagonal right, diagonal left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # 1. forward direction
            r = row + dr
            c = col + dc
            while r >= 0 and r < 6 and c >= 0 and c < 7:
                if self.board[r, c] == self.current_turn:
                    count = count + 1
                    r = r + dr
                    c = c + dc
                else:
                    break  # if different piece, stop counting this direction

            # 2. backward direction
            r = row - dr
            c = col - dc
            while r >= 0 and r < 6 and c >= 0 and c < 7:
                if self.board[r, c] == self.current_turn:
                    count = count + 1
                    r = r - dr
                    c = c - dc
                else:
                    break  # if different piece, stop counting this direction
            
            # if count >= 4, current player wins
            if count >= 4:
                self.isEnd = True
                return self.current_turn
                
        return 0

    def validAction(self):
        # which columns are not full yet? (topRowInColumn != -1)
        valid_list = []
        for c in range(7):
            if self.topRowInColumn(c) != -1:
                valid_list.append(c)
        return valid_list

    def insertColumn(self, col):
        targetRow = self.topRowInColumn(col)
        
        # col is full or game already ended, cannot insert
        if targetRow == -1 or self.isEnd:
            return False
        
        # insert coin to the board
        self.board[targetRow, col] = self.current_turn
        
        # check if this move wins the game
        self.winner = self.checkEndGameFromInsert(targetRow, col)
        
        # increase round count
        self.round = self.round + 1
        
        # if full board and no one wins, it's a draw, end the game
        if len(self.validAction()) == 0:
            self.isEnd = True

        # switch player turn
        if self.current_turn == 1:
            self.current_turn = 2
        else:
            self.current_turn = 1

        return True
