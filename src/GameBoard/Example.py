from GameBoard import Connect4Board

def play_test():
    # game = object
    game = Connect4Board(first_player=1)
    
    print("Welcome to Connect4Bot Test!")
    
    while not game.isEnd:
        game.showBoard()
        # available columns to play
        valid_columns = game.validAction()
        print(f"Can play in columns: {valid_columns}")
        
        try:
            col = int(input(f"Player {game.current_turn}, choose column (0-6): "))
            # check if the input column is in valid columns
            is_valid = False
            for v in valid_columns:
                if col == v:
                    is_valid = True
            
            # insert coin if valid
            if is_valid == True:
                game.insertColumn(col)
            else:
                print("!!! Column Full or Invalid !!!")
                # repeat the loop without switching player turn or increasing round count
                # because the player has to choose again until it's valid
        
        except ValueError:
            # check if the input is an integer
            print("!!! Error: Please enter only numbers 0-6 !!!")
        
        # -------------------------------------------

    # game ended, show final board and winner
    game.showBoard()
    
    if game.winner == 0:
        print("DRAW GAME!")
    else:
        print(f"CONGRATULATIONS! Player {game.winner} WINS!")

if __name__ == "__main__":
    play_test()





# from GameBoard import Connect4Board
    
# board = Connect4Board(first_player=1) # first_player = 1 or first_player = 2 
# print("Round No : {}".format(board.round))
# print("This is what board does look like")
# board.showBoard()
# print("List of valid action")
# print(board.validAction())
# print("state with current player ")
# print(board.getStateAsPlayer())
# x = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
# board.insertColumn(x)
# while(board.isEnd is not True):
#     print("Round No : {}".format(board.round))
#     print("This is what board does look like")
#     board.showBoard()
#     print("List of valid action")
#     print(board.validAction())
#     print("state with current player ")
#     print(board.getStateAsPlayer())
#     print("state for player 1")
#     print(board.getState(1).shape)
#     print("state for player 2")
#     print(board.getState(2).shape)
#     x = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
#     board.insertColumn(x)
# print("Winner is {}".format(board.winner))