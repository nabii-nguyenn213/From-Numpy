import pygame
from NeuralNetworkModel import *

#constant
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Board:
    
    def __init__(self):
        self.board = self.make_board()
    
    def make_board(self):
        self.row = 28
        self.col = 28
        board = []
        for i in range(self.row):
            r = []
            for j in range(self.col):
                r.append(0)
            board.append(r)
        return board
    
    def fill_board(self, coors):
        for coor in coors:
            self.board[coor[0]][coor[1]] = 1
    
    def tonumpy(self):
        return np.array(self.board)
    
    def print_board(self, board):
        board = board.tonumpy()
        for i in board:
            print(i)

    def clear_board(self):
        self.board = self.make_board()


class PygameInterface:
    
    def __init__(self):
        self.board = Board()
        self.coor = []
        self.y_pred = None
        self.model = DeepNeuralNetwork( Layer(28*28),
                                        Layer(64, activation_func='relu'),
                                        Layer(64, activation_func='relu'),
                                        Layer(10, activation_func='softmax', train_bias=False),
                                        init='xavier',
                                        uniform=True
                                    )
    
    def init(self):
        pygame.init()
        win_size = (1000, 700)
        self.window = pygame.display.set_mode(win_size)
        self.window.fill('cadetblue4')
        pygame.display.set_caption("Handwritten Digit Recognition")
    
    def fit(self, x_train, y_train, lr = 0.01, batch_size = 32, epochs = 1000):
        if self.model.is_file_empty('bias.npy') or self.model.is_file_empty('weights.npy'):
            print('training process')
            self.model.fit(x_train, y_train, lr = lr, batch_size = batch_size,  epochs = epochs)
            print('done training process')
            print('--------------------------------------------------------------------------------------')
            
        y_predict = self.model.predict(x_train)
        print('accuracy train :', self.model.accuracy(y_train, y_predict))
        y_predict_one_hot = self.model.one_hot(pd.Series(y_predict))
        y_train_one_hot = self.model.one_hot(y_train)
        print('cross-entropy loss :', self.model.cross_entropy_loss(y_train_one_hot.values, y_predict_one_hot.values))
        print('--------------------------------------------------------------------------------------')
    
    def run(self):
        running = True
        while running:
            self.mouse_x, self.mouse_y = pygame.mouse.get_pos()
            self.draw_grid()
            self.button()
            self.draw_digit_predict()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if pygame.mouse.get_pressed()[0]:
                    self.reset_draw_digit_predict()
                    if (14 < self.mouse_x < 686) and (14 < self.mouse_y < 686):
                        self.draw()
                        self.board.fill_board(self.coor)
        
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.reset_draw_digit_predict()
                    if (870 <= self.mouse_x <= 970) and (640 <= self.mouse_y <= 670): # *clear button
                        self.clear_board()
            pygame.display.flip()
    
    def button(self):
        img = self.board.tonumpy()
        img = img.reshape(784, 1)
        self.y_pred = self.model.predict_single_point(img)
        
        font = pygame.font.SysFont('sans', 25)
        
        clear_text = font.render('clear', True, BLACK)
        
        pygame.draw.rect(self.window, WHITE, (870, 640, 100, 30)) # *clear button
        self.window.blit(clear_text, (895, 640))
        
        pygame.draw.rect(self.window, WHITE, (750, 50, 200, 100)) # *predict box
        predict_number = font.render(str(self.y_pred), True, BLACK)
        predict_text = font.render('predict :', True, BLACK)
        self.window.blit(predict_text, (765, 80))
        self.window.blit(predict_number, (850 , 82))
    
    def draw(self):
        col = (self.mouse_x - 14) // 24
        row = (self.mouse_y - 14) // 24
        col1 = max(col - 1, 0)
        row1 = max(row - 1, 0)
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col), (14 + 24 * row), 24, 24))
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col1), (14 + 24 * row1), 24, 24))
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col1), (14 + 24 * row), 24, 24))
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col), (14 + 24 * row1), 24, 24))
        
        if (row, col) not in self.coor:
            self.coor.append((row, col))
        if (row1, col1) not in self.coor:
            self.coor.append((row1, col1))
        if (row1, col) not in self.coor:
            self.coor.append((row1, col))
        if (row, col1) not in self.coor:
            self.coor.append((row, col1))
    
    def reset_draw_digit_predict(self):
        pygame.draw.rect(self.window, 'cadetblue4', (780, 200, 250, 350))

    
    def draw_digit_predict(self):
        data = self.board.tonumpy()
        data = data.reshape(784, 1)
        output = self.model(data)
        
        i = 50
        font = pygame.font.SysFont('sans', 30)
        
        
        zero = font.render('0 : ' + str(round(float(output[0]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(zero, (750, 250 - i))
        
        one = font.render('1 : ' + str(round(float(output[1]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(one, (750, 285 - i))
        
        two = font.render('2 : ' + str(round(float(output[2]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(two, (750, 320 - i))
        
        three =font.render('3 : ' + str(round(float(output[3]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(three, (750, 355 - i))
        
        four = font.render('4 : ' + str(round(float(output[4]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(four, (750, 390 - i))
        
        five = font.render('5 : ' + str(round(float(output[5]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(five, (750, 425 - i))
        
        six = font.render('6 : ' + str(round(float(output[6]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(six, (750, 460 - i))
        
        seven = font.render('7 : ' + str(round(float(output[7]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(seven, (750, 495 - i))
        
        eight = font.render('8 : ' + str(round(float(output[8]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(eight, (750, 530 - i))
        
        nine = font.render('9 : ' + str(round(float(output[9]) * 100, 3)) + '%', True, BLACK)
        self.window.blit(nine, (750, 565 - i))
    
    def clear_board(self):
        self.window.fill('cadetblue4')
        self.draw_grid()
        self.coor = []
        self.board.clear_board()
        
    
    def draw_grid(self):
        for r in range(self.board.row+1):
            pygame.draw.line(self.window, WHITE, (r * 24 + 14, 14), (r * 24 + 14, 686))
            pygame.draw.line(self.window, WHITE, (14, r * 24 + 14), (686, r * 24 + 14))