import pygame
import sys
import random
import tensorflow as tf
tf.config.run_functions_eagerly(False) 
import keras as kr
import numpy as np
import os
from collections import deque as dq
from keras import layers , optimizers, models, losses
from keras.losses import MeanSquaredError # type: ignore

print("目前使用的 Python 解譯器：", sys.executable)

#minimax開關
USE_MINIMAX = True
NN_VS_MINIMAX = True
#簡易NN模型
model_path = "OOXX_model.h5"

if os.path.exists(model_path):
    model =kr.models.load_model(model_path)
    model.compile(optimizer='Adam', loss=MeanSquaredError())
    print("已載入模型")
else:
    print("未找到模型，建立新模型")
    model = kr.Sequential([
        kr.layers.Input(shape=(9,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss=MeanSquaredError())

#stage3 滑動窗
buf_states = dq(maxlen=50000)
buf_labels = dq(maxlen=50000)

#stage2
if os.path.exists("OOXX_training_data.npz"): 
    data = np.load("OOXX_training_data.npz")
    buf_labels.extend(data["labels"])
    buf_states.extend(data["states"])
    x0 = np.array(buf_states, dtype=np.float32).reshape(-1, 9)
    y0 = np.array(buf_labels, dtype=np.float32).reshape(-1, 9)
    if x0.shape[0] > 0:
        model.fit(x0,y0,epochs=10, batch_size=32)
        model.save(model_path)
        print("模型已儲存為 OOXX_model.h5")

        print("載入筆數：", len(buf_states))
    else:
        print("沒有找到訓練資料，這將是模型的第一次學習！")

#minimax (evaluate)
def evaluate(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "":
            return 10 if board[i][0] == "X" else -10
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] != "":
            return 10 if board[0][i] == "X" else -10
    if board[0][0] == board[1][1] == board[2][2] != "":
        return 10 if board[0][0] == "X" else -10
    if board[0][2] == board[1][1] == board[2][0] != "":
        return 10 if board[0][2] == "X" else -10
    return 0

#minimax main
def minimax(board, depth, is_maximizing):
    score = evaluate(board)
    if score != 0 or all(cell != "" for row in board for cell in row) or depth == 0:
        return score
    if is_maximizing:
        best = float('-inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    board[r][c] = "X"
                    val = minimax(board, depth-1, False)
                    board[r][c] = ""
                    best = max(best, val)
        return best
    else:
        best = float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    board[r][c] = "O"
                    val = minimax(board, depth-1, True)
                    board[r][c] = ""
                    best = min(best, val)
        return best

 #minimax getbestmove
def get_best_move(board, max_depth=4):
    best_val = float('-inf')
    best_move = None
    for r in range(3):
        for c in range(3):
            if board[r][c] == "":
                board[r][c] = "X"
                move_val = minimax(board, max_depth-1, False)
                board[r][c] = ""
                if move_val > best_val:
                    best_val = move_val
                    best_move = (r, c)
    return best_move

#AI stage 1
def board_to_input(board):
    mapping = {"X":1, "O":-1, "":0}
    return np.array([mapping[cell]for row in board for cell in row])

if NN_VS_MINIMAX == False:

    pygame.init()
    Width, Height=300, 300
    Line_Width=15
    Line_color=(255,255,255)
    O_color=(255, 0, 0)
    X_color=(0,0,255)
    BackGround_color=(0,0,0)
    Circle_Radius=50
    X_width=25

    Screen=pygame.display.set_mode((Width, Height))
    pygame.display.set_caption("井字棋")

    board = []
    for i in range(3):   #內部迴圈為橫列 外圍為直排
        row = []
        for j in range(3):
            row.append("")
        board.append(row)

    def draw_lines():
        pygame.draw.line(Screen,Line_color,(0,Height//3),(Width,Height//3),Line_Width)
        pygame.draw.line(Screen,Line_color,(0,2*Height//3),(Width,2*Height//3),Line_Width)

        pygame.draw.line(Screen,Line_color,(Width//3,0),(Width//3,Height),Line_Width)
        pygame.draw.line(Screen,Line_color,(2*Width//3,0),(2*Width//3,Height),Line_Width)

    def draw_o(row,col):
        pygame.draw.circle(Screen,O_color,(col*Width//3 + Width//6,row*Height//3 + Height//6),Circle_Radius,Line_Width)

    def draw_x(row,col):
        pygame.draw.line(Screen,X_color,(col*Width//3+10,row*Height//3+10),(col*Width//3+Width//3-10,row*Height//3+Height//3-10),X_width) #左上到右下
        pygame.draw.line(Screen,X_color,(col*Width//3+10,row*Height//3+Height//3-10),(col*Width//3+Width//3-10,row*Height//3+10),X_width) #左下到右上

    def check_winner():
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2] !="" :
                pygame.draw.line(Screen,Line_color,(0,row*Height//3+Height//6),(Width,row*Height//3+Height//6),Line_Width)
                return True
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != "":
                pygame.draw.line(Screen, Line_color, (col * Width // 3 + Width // 6, 0), (col * Width // 3 + Width // 6, Height), Line_Width)
                return True
        if board[0][0] == board[1][1] == board[2][2] != "":
            pygame.draw.line(Screen, Line_color, (0, 0), (Width, Height), Line_Width)
            return True
        if board[0][2] == board[1][1] == board[2][0] != "":
            pygame.draw.line(Screen, Line_color, (Width, 0), (0, Height), Line_Width)
            return True
        return False

    def check_draw():
        for row in board:
            if "" in row: 
                return False
        return True

    def main():
     #AI stage2
        current_player = "X"
        game_over = False
        Screen.fill(BackGround_color)
        draw_lines()
        pygame.display.update()
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()                
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    mouse_X = event.pos[0]
                    mouse_Y = event.pos[1]

                    clicked_row = mouse_Y // (Height//3)
                    clicked_col = mouse_X // (Width//3)

                    if 0 <= clicked_row < 3 and 0 <= clicked_col < 3:
                        if board[clicked_row][clicked_col] == "":
                            board[clicked_row][clicked_col] = current_player
                            if current_player == "X":
                                board[clicked_row][clicked_col] = "X"
                                draw_x(clicked_row,clicked_col)
                                pygame.display.update()

                                flat_board = board_to_input(board) #stage3
                                action_index = clicked_row*3+clicked_col
                                label =[0]*9
                                label[action_index] = 1
                                buf_states.append(flat_board)
                                buf_labels.append(label) 

                                if check_winner():
                                    print(f"player{current_player} wins!" )
                                    game_over = True
                                    break                            
                                elif not game_over and check_draw():
                                    print("Draw!")
                                    game_over = True
                                else:
                                    current_player = "O"

            #AI 
            if current_player == "O":
                empty_cells = []
                for row in range(3):
                    for col in range(3):
                        if board[row][col] == "":
                            empty_cells.append((row,col))

                if empty_cells:                                     #AI從這裡開始 stage1
                    input_data = board_to_input(board).reshape(1,9)
                    predictions = model.predict(input_data, verbose=2)[0]
                    legal_moves=[(row, col) for row in range(3) for col in range(3) if board[row][col]==""]
                    if legal_moves:
                        best_move = max(legal_moves, key =lambda move:predictions[move[0]*3+move[1]])
                        #stage2
                        flat_board = board_to_input(board)
                        action_index = best_move[0]*3+best_move[1]
                        label = [0]*9
                        label[action_index] = 1
                        buf_states.append(flat_board)
                        buf_labels.append(label)
                    #stage1
                    board[best_move[0]][best_move[1]] = "O"
                    draw_o(best_move[0],best_move[1])
                    pygame.display.update()
                    if check_winner():
                        print(f"player{current_player} wins!")
                        game_over = True
                    elif check_draw():
                        print("Draw!")
                        game_over = True
                    current_player = "X"
        pygame.display.update()

        if game_over:   #stage2
            if os.path.exists("OOXX_training_data.npz"):
                old = np.load("OOXX_training_data.npz")
                old_states = np.atleast_2d(old['states'])
                old_labels = np.atleast_2d(old['labels'])
                all_states = np.array(buf_states, dtype=np.float32)
                all_states = all_states.reshape(-1,9)
                all_labels = np.array(buf_labels, dtype=np.float32)
                all_labels = all_labels.reshape(-1, 9)                
            else:
                all_states = np.array(buf_states)
                all_labels = np.array(buf_labels)
            np.savez("OOXX_training_data.npz", states=all_states,labels=all_labels)
            print("訓練資料已累積處存")
            print("這一局記錄筆數：", len(buf_states))
    if __name__ == "__main__":
        main()
else:
    num_games = 200 #可自行自定義局數
    stats = {"X":0, "O":0,"draw":0}
    history_states=[]
    history_labels=[]
    for _ in range(num_games):
        board = []
        for i in range(3):   #內部迴圈為橫列 外圍為直排
            row = []
            for j in range(3):
                row.append("")
            board.append(row)
        
        legal_moves = [(r,c) for r in range(3) for c in range(3) if board[r][c]==""]
        first = random.choice(legal_moves)
        board[first[0]][first[1]] = "X"
        current_player = "O"

        game_over = False
        history = []
        while not game_over:
            if current_player =="X": #(minimax)
                state = board_to_input(board)
                move = get_best_move(board)
                board[move[0]][move[1]] = "X"
                result = evaluate(board)

                label = [0]*9
                label[move[0]*3+move[1]] = 1
                buf_states.append(state)
                buf_labels.append(label)

                if result == 10:
                    stats["X"] +=1
                    game_over=True
                elif all(cell !='' for row in board for cell in row):
                    stats["draw"]+=1
                    game_over = True
                else:
                    current_player = "O"
            else:                    
                empty_cells = []
                for row in range(3):
                    for col in range(3):
                        if board[row][col] == "":
                            empty_cells.append((row,col))

                if empty_cells:                                     #AI從這裡開始 stage1
                    input_data = board_to_input(board).reshape(1,9)
                    predictions = model.predict(input_data, verbose=2)[0]
                    legal_moves=[(row, col) for row in range(3) for col in range(3) if board[row][col]==""]
                    best_move = max(legal_moves, key =lambda move:predictions[move[0]*3+move[1]])
                    #stage2
                    flat_board = board_to_input(board)
                    action_index = best_move[0]*3+best_move[1]
                    label = [0]*9
                    label[action_index] = 1
                    buf_states.append(flat_board)
                    buf_labels.append(label)
                    #stage1
                    board[best_move[0]][best_move[1]] = "O" 
                    history.append(('O',best_move))

                    result = evaluate(board)
                    if result == -10:
                        stats["O"] += 1
                        game_over = True
                    elif all(cell != "" for row in board for cell in row):
                        stats["draw"] += 1
                        game_over = True
                    else:
                        current_player = "X"
            if os.path.exists("OOXX_training_data.npz"):
                old = np.load("OOXX_training_data.npz")
                old_states = np.atleast_2d(old['states'])
                old_labels = np.atleast_2d(old['labels'])
                all_states = np.array(buf_states, dtype=np.float32)
                all_states = all_states.reshape(-1,9)
                all_labels = np.array(buf_labels, dtype=np.float32)
                all_labels = all_labels.reshape(-1, 9)                
            else:
                all_states = np.array(buf_states)
                all_labels = np.array(buf_labels)
            np.savez("OOXX_training_data.npz", states=all_states, labels=all_labels)
        total = stats['X'] + stats['O'] + stats['draw']
        print(f"minimax{stats['X']}勝/NN{stats['O']}勝/和{stats['draw']}局/勝率{(stats['O']/total):.2f} ")        
                