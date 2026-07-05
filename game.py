'''
Huligutta (Goats and Tigers)
file: game.py
Description: GUI of the game using TKinter
'''

__author__ = "Clyde James Felix"
__email__ = "cjfelix.hawaii.edu"
__status__ = "Dev"

from tkinter import *
from tkinter import messagebox
import os
import numpy as np
import sys
import time
from huligutta import *
from functions import *
from PIL import ImageTk, Image
from random import randint, choice
from sys import platform
try:
    from qlearning import QLearningAgent
    QLEARNING_AVAILABLE = True
except ImportError:
    QLEARNING_AVAILABLE = False
    print("Warning: Q-learning module not available. Install required dependencies.")

# Board parts

numPosition = 23 
if platform == 'darwin':
    # Mac
    boardSize = 500
    fontSize = 15
    buttonStyle = 30
elif platform == 'win32' or platform == 'cygwin':
    # Windows
    boardSize = 500
    fontSize = 12
    buttonStyle = 0
elif  platform == 'linux':
    # Linux
    pass

boardSize = 500
Board().clearBoard()
possible_pos = ['a1','a2','a3','b0','b1','b2','b3','b4','c1','c2','c3','c4','d1','d2','d3','d4','e1','e2','e3','e4','f1','f2','f3']
root = os.path.abspath('images')
# Game modes:
class Main():

    def __init__(self,mode):

        tigerPhoto =  root + '/tiger.gif'
        goatPhoto = root + '/goat.gif'
            
        self.window = Tk()        
        self.window.title('Huligutta (Goats & Tigers)')
        self.window.geometry(str(boardSize) + 'x' + str(boardSize+20))
        self.window.resizable(0,0)
        # self.window.tk_focusFollowsMouse()
        # self.window.tk.call('tk', 'scaling', 17)
        self.tiger = PhotoImage(file=tigerPhoto)
        self.tiger = self.tiger.subsample(2,2)
        self.goat = PhotoImage(file=goatPhoto)
        self.goat = self.goat.subsample(2,2)
        self.canvas = Canvas(self.window,width=boardSize,height=boardSize)

        # For self.turn, Goat: False, Tiger: True
        self.turn = False
        self.move = False
        self.initialize_board()
        self.canvas.pack()
        
        self.location = ''
        self.turntext = StringVar()
        self.numGoats = StringVar()
        self.goatsEatentext = StringVar()
        self.selectedBtn = StringVar()
        self.goatEaten = 0
        self.goatCount = 0  # Current goats on board
        self.goatsPlaced = 0  # Total goats placed (including captured ones)
        self.moveCount = 0
        self.moveCount_prev = 0
        self.mode = mode
        
        # Initialize Q-learning agents if available
        self.tiger_agent = None
        self.goat_agent = None
        if QLEARNING_AVAILABLE:
            if mode in ['qlearningTiger', 'qlearningBoth']:
                self.tiger_agent = QLearningAgent(player='tiger', epsilon=0.0)  # No exploration during play
                # Try to load trained Q-table
                q_file = 'q_table_tiger_final.pkl'
                if not os.path.exists(q_file):
                    q_file = 'q_table_tiger_1000.pkl'
                if os.path.exists(q_file):
                    self.tiger_agent.load_q_table(q_file)
                    print(f"Loaded Q-table for tiger from {q_file}")
                else:
                    print(f"Warning: Q-table file not found. Tiger will play randomly.")
            if mode in ['qlearningGoat', 'qlearningBoth']:
                self.goat_agent = QLearningAgent(player='goat', epsilon=0.0)  # No exploration during play
                # Try to load trained Q-table
                q_file = 'q_table_goat_final.pkl'
                if not os.path.exists(q_file):
                    q_file = 'q_table_goat_1000.pkl'
                if os.path.exists(q_file):
                    self.goat_agent.load_q_table(q_file)
                    print(f"Loaded Q-table for goat from {q_file}")
                else:
                    print(f"Warning: Q-table file not found. Goat will play randomly.")
        self.turnDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.turntext).place(x=boardSize - 100,y= 25)
        self.selectedDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.selectedBtn).place(x=boardSize - 100,y = 50)
        self.goatDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.numGoats).place(x=10,y= 25)
        self.goatEatenDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.goatsEatentext).place(x= 10,y= 50)
        if self.goatsPlaced < 15:
            self.numGoats.set(f"Goats: {self.goatCount} on board / {self.goatsPlaced} placed (max 15)")
        else:
            self.numGoats.set(f"Goats: {self.goatCount} on board / {self.goatsPlaced} placed")
        self.goatsEatentext.set("Goats eaten: " + str(self.goatEaten))
        self.turnDisplay()
        self.selectedToggle()
        self.reportBugbtn = Button(self.window,text = 'Report Bug',command=lambda : self.report_bug()).place(x= 70,y = boardSize - 15,anchor=CENTER)

        # Buttons
        # self.btn1  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b0')).place(x=boardSize/2,y=boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn2  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('a1')).place(x=boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn3  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('a2')).place(x=boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn4  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('a3')).place(x=boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn5  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b1')).place(x=boardSize/2 - 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn6  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b2')).place(x=boardSize/2- 100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn7  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b3')).place(x=boardSize/2-135,y=boardSize/2+70,height=30,width=30,anchor=CENTER)
        # self.btn8  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b4')).place(x=boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn9  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c1')).place(x=boardSize/2 - 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn10 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c2')).place(x=boardSize/2 - 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn11 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c3')).place(x=boardSize/2 - 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn12 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c4')).place(x= boardSize/2 - 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn13 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d1')).place(x=boardSize/2 + 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn14 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d2')).place(x=boardSize/2 + 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn15 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d3')).place(x=boardSize/2 + 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn16 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d4')).place(x=boardSize/2 + 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn17 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e1')).place(x=boardSize/2 + 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn18 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e2')).place(x=boardSize/2+100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn19 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e3')).place(x=boardSize/2 + 135,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn20 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e4')).place(x=boardSize - boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)        
        # self.btn21 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('f1')).place(x=boardSize-boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn22 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('f2')).place(x=boardSize-boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn23 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('f3')).place(x=boardSize-boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)

    def initialize_board(self):
        # Draws the board
        self.canvas.create_rectangle(boardSize/10,boardSize/2 - 70,boardSize-boardSize/10,boardSize/2+70)    
        self.canvas.create_line(boardSize/10,boardSize/2,boardSize-boardSize/10,boardSize/2)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize/10,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize - boardSize/10,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/10,boardSize - boardSize/10,boardSize - boardSize/10,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize/2 - 80,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize/2 + 80,boardSize - boardSize/10)

        printAndLog('========================')
        attempts = textCount('Attempts: ')
        printAndLog('Attempts: ' +  str(attempts))
        printAndLog('Game Mode: '+ mode)
        printAndLog('========================')

    def button_position(self,pos):
        # Moves pieces in clicks

        # print('###### Debug ######')
        # print('location:', self.location)
        # print('Position clicked: ', pos)

        # Handles how goat positions move
        self.moveCount_prev = self.moveCount

        if mode == 'pvp':
            self.pvpMode(pos)
            self.selectedToggle()
            self.turnDisplay()
        elif mode == 'goatPlayer':
            self.goatMode(pos)
        elif mode == 'tigerPlayer':
            self.tigerMode(pos)
        elif mode == 'qlearningTiger':
            self.qlearningTigerMode(pos)
        elif mode == 'qlearningGoat':
            self.qlearningGoatMode(pos)
        elif mode == 'qlearningBoth':
            self.qlearningBothMode(pos)        

        if self.moveCount_prev != self.moveCount:
            self.update()
            self.collectData()
        self.window.mainloop()                

    def pvpMode(self,pos):

        # How goats move
        if self.turn == False:
            
            if Position(pos[0],pos[1]).content() == 'X':
                print('You must select any empty or goat positions')
            elif Position(pos[0],pos[1]).content() == () and self.move == False:
                if self.goatsPlaced < 15:
                    Goat(pos).place()
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.goatsPlaced += 1  # Increment total goats placed
                    self.turn = True
                else: 
                    print("Cannot place more goats. Maximum 15 goats can be placed.")

            elif Position(pos[0],pos[1]).content() == 'O':
                if self.goatsPlaced == 15:  # All 15 goats have been placed
                    if pos == self.location:
                        self.move = not self.move
                        # print('Debug: self.move ',self.move)                    
                # Select valid position to move
                    elif self.move == False: 
                        self.move = True
                        self.location = pos
                else:
                    print("Goats can only be moved if all 15 goats have been placed")
            else:
                
                if Goat(self.location).move(pos) == 1:
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.move = False
                    self.location = ''
                    self.turn = True
                    self.moveCount = self.moveCount + 1
            pass

        # How tigers move
        elif self.turn == True:
            # print('DEBUG: location',self.location)
            
            # print('DEBUG: possible moves',Piece(pos).possibleMoves())
            if Position(pos[0],pos[1]).content() != 'X' and self.move == False:
                print('You must select any current tiger positions') 

            elif Position(pos[0],pos[1]).content() == 'X':
                if pos == self.location:
                    self.move = not self.move
                elif self.location == '' or self.move == False or pos != self.location:
                    self.move = True
                    self.location = pos

            elif Position(pos[0],pos[1]).content() == () and self.move == True:
                # print('DEBUG: location',self.location)
                # print('DEBUG: secondAdjacent',Piece(self.location).secondAdjacent(pos))
                # print('DEBUG: Adjacent',Piece(self.location).adjacent(pos))
                # print('DEBUG: possiblemoves 2 ', Piece(self.location).possibleMoves())
                if pos in Position(self.location[0],self.location[1]).get_neighbors():
                    tigerMoveFlag = Tiger(self.location).move(pos)

                    if tigerMoveFlag == 1:
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1
                        return

                # TODO: error on capture message when Tiger moves to corner (i.e Tiger move to e4)
                elif pos in Tiger(self.location).possibleMoves():
                    if Tiger(self.location).capture(pos) == 1:

                        self.goatEaten = self.goatEaten + 1 
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1

                else:
                    print('Error on capture')
                    self.move = False
                    self.location = ''
                    self.turn = True
        self.selectedToggle()

    def goatMode(self,pos):

        if self.turn == False:
            if Position(pos[0],pos[1]).content() == 'X':
                print('You must select any empty or goat positions')
            elif Position(pos[0],pos[1]).content() == () and self.move == False:
                if self.goatsPlaced < 15:
                    Goat(pos).place()
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.goatsPlaced += 1  # Increment total goats placed
                    self.turn = True
                else: 
                    print("Cannot place more goats. Maximum 15 goats can be placed.")
            elif Position(pos[0],pos[1]).content() == 'O':
                if self.goatsPlaced == 15:  # All 15 goats have been placed
                    if pos == self.location:
                        self.move = not self.move
                        # print('Debug: self.move ',self.move)
                # Select valid position to move
                    elif self.move == False: 
                        self.move = True
                        self.location = pos
                else:
                    print("Goats can only be moved if all 15 goats have been placed")
            else:
                if Goat(self.location).move(pos) == 1:
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.move = False
                    self.location = ''
                    self.turn = True
                    self.moveCount = self.moveCount + 1
        self.selectedToggle()
        time.sleep(0.5)
        if self.turn == True:
            # Randomize tiger positions
            possibleCaptures = {}
            tigers = tigerPositions(Board().boardPositions)
            # print("DEBUG: tigers ", tigers)

            for tiger in tigers:
                # print('DEBUG: Position(tiger[0],tiger[1]).get_captures()', Position(tiger[0],tiger[1]).get_captures())
                if Position(tiger[0],tiger[1]).get_captures() != None:
                    for capture in Position(tiger[0],tiger[1]).get_captures():
                        possibleCaptures[tiger] = capture

            tigerPos = choice(tigers)
            while len(Tiger(tigerPos).possibleMoves()) == 0:
                if len(tigers) != 0:
                    tigers.remove(tigerPos)
                    tigerPos = choice(tigers)
                else:
                    return
            
            # print('DEBUG Tiger possible moves ',Tiger(tigerPos).possibleMoves())
            tigerChoice = choice(Tiger(tigerPos).possibleMoves())
            # print('DEBUG Tiger choice' , tigerChoice)
            
            if pos in tigerPos:
                del tigerChoice[tigerChoice.index(pos)]

            # print('###### Debug ######')
            # print('tiger to move:', tigerPos)
            # print('Possible moves: ', Tiger(tigerPos).possibleMoves())
            # print('Choice the tiger made: ',choice(Tiger(tigerPos).possibleMoves()))

            if len(possibleCaptures) != 0:
                tigerPos,where2go = choice(list(possibleCaptures.items()))
                # print('DEBUG possibleMOves', tigerPos,' ',where2go)
                # print('DEBUG TIGERCPOS', tigerPos)
                # print('DEBUG possibleCaptures[tigerPos] ', possibleCaptures[tigerPos])
                if Tiger(tigerPos).capture(where2go) == 1:

                    self.goatEaten = self.goatEaten + 1
                    self.move = False
                    self.location = ''
                    self.turn = False
                    self.moveCount = self.moveCount + 1
            elif tigerChoice in Position(tigerPos[0],tigerPos[1]).get_neighbors():
                tigerMoveFlag = Tiger(tigerPos).move(tigerChoice) 

                if tigerMoveFlag == 1:
                    self.move = False
                    self.turn = False
                    self.moveCount = self.moveCount + 1
                    return

            else:
                print('Error on capture')
                self.move = False
                self.location = ''
                self.turn = True

    def tigerMode(self,pos):

        if self.turn == True:
            # print('DEBUG: possible moves',Piece(pos).possibleMoves())
            if Position(pos[0],pos[1]).content() != 'X' and self.move == False:
                print('You must select any current tiger positions') 

            elif Position(pos[0],pos[1]).content() == 'X':
                if pos == self.location:
                    self.move = not self.move
                elif self.location == '' or self.move == False or pos != self.location:
                    self.move = True
                    self.location = pos

            elif Position(pos[0],pos[1]).content() == () and self.move == True:
                # print('DEBUG: location',self.location)
                # print('DEBUG: secondAdjacent',Piece(self.location).secondAdjacent(pos))
                # print('DEBUG: Adjacent',Piece(self.location).adjacent(pos))
                # print('DEBUG: possiblemoves 2 ', Piece(self.location).possibleMoves())
                if pos in Position(self.location[0],self.location[1]).get_neighbors():
                    tigerMoveFlag = Tiger(self.location).move(pos) 

                    if tigerMoveFlag == 1:
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1
                        
                elif pos in Piece(self.location).possibleMoves():
                    if Tiger(self.location).capture(pos) == 1:

                        self.goatEaten = self.goatEaten + 1 
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1

                else:
                    print('Error on capture')
                    self.move = False
                    self.location = ''
                    self.turn = True       
            self.selectedToggle()

        if self.turn == False:
            time.sleep(.2)
           # Randomize goat positions
            emptyPos = emptyPositions(Board().boardPositions)
            goats = goatPositions(Board().boardPositions)
            # print("DEBUG: goats ", goats)

            if self.numGoats == 15:
                goatPos = choice(goats)

                while len(Goat(goatPos).possibleMoves()) == 0:
                    if len(goats) != 0:
                        goats.remove(goatPos)
                    else:
                        return

                goatChoice = choice(Goat(goatPos).possibleMoves())
                # print('DEBUG goatChoice ',goatChoice)

                if pos in goatPos:
                    del goatChoice[goatChoice.index(pos)]

                # print('###### Debug ######')
                # print('goat to move:', goatPos)
                # print('Possible moves: ', Goat(goatPos).possibleMoves())
                # print('Choice the goat made: ',choice(Goat(goatPos).possibleMoves()))

                if goatChoice in Position(goatPos[0],goatPos[1]).get_neighbors():
                    goatMoveFlag = Goat(goatPos).move(goatChoice) 

                    if goatMoveFlag == 1:
                        self.move = False
                        self.turn = True
                        self.moveCount = self.moveCount + 1
                        return

                else:
                    print('Error')
                    self.move = False
                    self.location = ''
                    self.turn = False
            else:
                goatChoice = choice(emptyPos)
                Goat(goatChoice).place()
                self.move = False
                self.turn = True
    
    def execute_qlearning_action(self, action):
        """Execute an action chosen by Q-learning agent."""
        if action is None:
            return False
        
        action_type, from_pos, to_pos = action
        
        try:
            if action_type == 'capture':
                tiger = Tiger(from_pos)
                if tiger.capture(to_pos) == 1:
                    self.goatEaten += 1
                    self.turn = False  # After capture, it's goat's turn
                    self.moveCount += 1
                    self.update()  # Update display
                    return True
            elif action_type == 'move':
                if self.turn:  # Tiger move
                    tiger = Tiger(from_pos)
                    if tiger.move(to_pos) == 1:
                        self.turn = False  # After tiger move, it's goat's turn
                        self.moveCount += 1
                        self.update()  # Update display
                        return True
                else:  # Goat move
                    goat = Goat(from_pos)
                    if goat.move(to_pos) == 1:
                        self.turn = True  # After goat move, it's tiger's turn
                        self.moveCount += 1
                        self.update()  # Update display
                        return True
            elif action_type == 'place':
                if self.goatsPlaced < 15:
                    goat = Goat(to_pos)
                    goat.place()
                    self.goatsPlaced += 1  # Increment total goats placed
                    self.turn = True  # After placing goat, it's tiger's turn
                    self.update()  # Update display
                    return True
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
        return False
    
    def qlearningTigerMode(self, pos):
        """Mode where player plays as goat, Q-learning agent plays as tiger."""
        if not QLEARNING_AVAILABLE or self.tiger_agent is None:
            print("Q-learning not available, falling back to random")
            self.tigerMode(pos)
            return
        
        # Player's turn (goat)
        if self.turn == False:
            if Position(pos[0],pos[1]).content() == 'X':
                print('You must select any empty or goat positions')
            elif Position(pos[0],pos[1]).content() == () and self.move == False:
                if self.goatsPlaced < 15:
                    Goat(pos).place()
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.goatsPlaced += 1  # Increment total goats placed
                    self.turn = True
                    self.selectedToggle()
                    self.turnDisplay()
                    self.update()
                    # Q-learning agent's turn
                    self.qlearning_tiger_move()
                else: 
                    print("Cannot place more goats. Maximum 15 goats can be placed.")
            elif Position(pos[0],pos[1]).content() == 'O':
                if self.goatsPlaced == 15:  # All 15 goats have been placed
                    if pos == self.location:
                        self.move = not self.move
                    elif self.move == False: 
                        self.move = True
                        self.location = pos
                else:
                    print("Goats can only be moved if all 15 goats have been placed")
            else:
                if Goat(self.location).move(pos) == 1:
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.move = False
                    self.location = ''
                    self.turn = True
                    self.moveCount = self.moveCount + 1
                    self.selectedToggle()
                    self.turnDisplay()
                    self.update()
                    # Q-learning agent's turn
                    self.qlearning_tiger_move()
    
    def qlearning_tiger_move(self):
        """Make Q-learning agent move for tiger."""
        if not QLEARNING_AVAILABLE or self.tiger_agent is None:
            return
        
        # Don't move if it's not tiger's turn
        if not self.turn:
            return
        
        time.sleep(0.3)  # Small delay for visualization
        board_state = Board().boardPositions
        action = self.tiger_agent.choose_action(board_state, is_tiger_turn=True, training=False)
        
        if action:
            if self.execute_qlearning_action(action):
                self.selectedToggle()
                self.turnDisplay()
                self.collectData()
    
    def qlearningGoatMode(self, pos):
        """Mode where player plays as tiger, Q-learning agent plays as goat."""
        if not QLEARNING_AVAILABLE or self.goat_agent is None:
            print("Q-learning not available, falling back to random")
            self.goatMode(pos)
            return
        
        # Player's turn (tiger)
        if self.turn == True:
            if Position(pos[0],pos[1]).content() != 'X' and self.move == False:
                print('You must select any current tiger positions') 
            elif Position(pos[0],pos[1]).content() == 'X':
                if pos == self.location:
                    self.move = not self.move
                elif self.location == '' or self.move == False or pos != self.location:
                    self.move = True
                    self.location = pos
            elif Position(pos[0],pos[1]).content() == () and self.move == True:
                if pos in Position(self.location[0],self.location[1]).get_neighbors():
                    tigerMoveFlag = Tiger(self.location).move(pos)
                    if tigerMoveFlag == 1:
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1
                        self.selectedToggle()
                        self.turnDisplay()
                        self.update()
                        # Q-learning agent's turn
                        self.qlearning_goat_move()
                        return
                elif pos in Tiger(self.location).possibleMoves():
                    if Tiger(self.location).capture(pos) == 1:
                        self.goatEaten = self.goatEaten + 1 
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1
                        self.selectedToggle()
                        self.turnDisplay()
                        self.update()
                        # Q-learning agent's turn
                        self.qlearning_goat_move()
                else:
                    print('Error on capture')
                    self.move = False
                    self.location = ''
                    self.turn = True
            self.selectedToggle()
    
    def qlearning_goat_move(self):
        """Make Q-learning agent move for goat."""
        if not QLEARNING_AVAILABLE or self.goat_agent is None:
            return
        
        time.sleep(0.3)  # Small delay for visualization
        board_state = Board().boardPositions
        action = self.goat_agent.choose_action(board_state, is_tiger_turn=False, training=False)
        
        if action:
            if self.execute_qlearning_action(action):
                self.update()
                self.collectData()
    
    def qlearningBothMode(self, pos):
        """Mode where both players are Q-learning agents (auto-play)."""
        # In this mode, user clicks don't matter, agents play automatically
        # This is mainly for demonstration
        if not QLEARNING_AVAILABLE:
            print("Q-learning not available")
            return
        
        # Auto-play mode - agents take turns automatically
        if self.turn:  # Tiger's turn
            self.qlearning_tiger_move()
        else:  # Goat's turn
            self.qlearning_goat_move()
                
    def update(self):
        # Updates the Screen
        numGoats = 0
        tigerPos = tigerPositions(Board().boardPositions)
        possibleMovesCount = 0
        for i in range(len(possible_pos)):
            if Position(possible_pos[i][0],possible_pos[i][1]).content() == ():
                self.change_button(possible_pos[i],'')
            elif Position(possible_pos[i][0],possible_pos[i][1]).content() == 'O':
                numGoats = numGoats + 1
                self.change_button(possible_pos[i],self.goat)
            else:
                self.change_button(possible_pos[i],self.tiger)
        
        for tiger in tigerPos:
            possibleMovesCount = possibleMovesCount + len(Piece(tiger).possibleMoves())

        self.goatCount = numGoats
        # Display: current goats on board / total goats placed
        if self.goatsPlaced < 15:
            self.numGoats.set(f"Goats: {numGoats} on board / {self.goatsPlaced} placed (max 15)")
        else:
            self.numGoats.set(f"Goats: {numGoats} on board / {self.goatsPlaced} placed")
        self.goatsEatentext.set("Goats eaten: " + str(self.goatEaten))
        self.turnDisplay()
        self.selectedToggle()
        self.window.update()
        # Endgame
        if possibleMovesCount == 0:
            printAndLog('Goat wins')
            messagebox.showinfo("Game Over","Goat wins") 
            return

        if self.goatEaten == 5:
            printAndLog('Tiger wins')
            messagebox.showinfo("Game Over", "Tiger wins") 
            return

    def collectData(self):
        printAndLog("Move: " + str(self.moveCount))
        printAndLog("--------------------")
        printAndLog("Goats: " + str(self.goatCount))
        tigers = tigerPositions(Board().boardPositions)
        printAndLog("Tigers positions: " + str(tigers))
        printAndLog(str(Board().boardPositions))
        editDistance = edit_distance(Board().boardPositions)
        printAndLog("Edit distance: " + str(editDistance))
        
    def selectedToggle(self):
        if self.move == True:
            self.selectedBtn.set('Selected:' + str(self.location))
        else:
            self.selectedBtn.set('')

    def turnDisplay(self):
        # Displays turn as text in the window
        if self.turn:
            self.turntext.set("Turn: Tiger")
        else:
            self.turntext.set("Turn: Goat")

    def change_button(self,pos,img):
        # change the images of the board pieces

        if pos == 'b0':
            self.btn1 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b0')).place(x=boardSize/2,y=boardSize/10,height=30,width=30,anchor=CENTER)
        if pos == 'a1':
            self.btn2 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('a1')).place(x=boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'a2':
            self.btn3 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('a2')).place(x=boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'a3':
            self.btn4 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('a3')).place(x=boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'b1':
            self.btn5 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b1')).place(x=boardSize/2 - 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)  
        if pos == 'b2':
            self.btn6 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b2')).place(x=boardSize/2- 100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'b3':
            self.btn7 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b3')).place(x=boardSize/2-135,y=boardSize/2+70,height=30,width=30,anchor=CENTER)
        if pos == 'b4':
            self.btn8 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b4')).place(x=boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER) 
        if pos == 'c1':
            self.btn9 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c1')).place(x=boardSize/2 - 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'c2':
            self.btn10 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c2')).place(x=boardSize/2 - 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'c3':        
            self.btn11 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c3')).place(x=boardSize/2 - 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'c4':
            self.btn12 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c4')).place(x= boardSize/2 - 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        if pos == 'd1':     
            self.btn13 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d1')).place(x=boardSize/2 + 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'd2':
            self.btn14 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d2')).place(x=boardSize/2 + 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'd3':      
            self.btn15 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d3')).place(x=boardSize/2 + 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'd4':
            self.btn16 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d4')).place(x=boardSize/2 + 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)  
        if pos == 'e1':  
            self.btn17 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e1')).place(x=boardSize/2 + 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'e2':
            self.btn18 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e2')).place(x=boardSize/2+100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'e3':
            self.btn19 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e3')).place(x=boardSize/2 + 135,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'e4':
            self.btn20 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e4')).place(x=boardSize - boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)        
        if pos == 'f1':
            self.btn21 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('f1')).place(x=boardSize-boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'f2':
            self.btn22 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('f2')).place(x=boardSize-boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'f3': 
            self.btn23 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('f3')).place(x=boardSize-boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)

    def report_bug(self):
        printAndLog('Bug reporting...')
        print('What to include for the report:')
        print('What happened?')
        print('What operating system you are running?')
        reason = input('Type the reason for the bug:')
        printAndLog('Bug reported \nReason: ' + reason)

    def start(self):
        # Sets up the game window
        # Game starts from scratch: 3 tigers at initial positions, no goats
        
        # Clear the board first
        Board().clearBoard()
        
        # Place 3 tigers at official initial positions
        Tiger('b0').place()
        Tiger('c1').place()
        Tiger('d1').place()
        
        # No goats placed initially - game starts with goat's turn to place first goat
        self.goatCount = 0
        self.goatsPlaced = 0  # Total goats placed (including captured)
        self.goatEaten = 0
        self.turn = False  # Goat's turn to place first goat
        self.move = False
        self.location = ''
        self.moveCount = 0
        
        # Special handling for tigerPlayer mode (if needed)
        if mode == 'tigerPlayer':
            # In tigerPlayer mode, one goat is placed automatically and tiger goes first
            empty = emptyPositions(Board().boardPositions)
            if empty:
                emptyPos = choice(empty)
                Goat(emptyPos).place()
                self.goatCount = 1
                self.turn = True  # Tiger's turn
        
        self.update()
        self.window.mainloop()        

if __name__ == "__main__":
    # Game mode:
        # Uncomment to choose the game mode

    # mode = 'pvp'          # Player vs. Player
    # mode = 'goatPlayer'   # You are Goat
    # mode = 'tigerPlayer'  # You are tiger
    # mode = 'qlearningTiger'  # You are Goat, Q-learning agent is Tiger
    # mode = 'qlearningGoat'   # You are Tiger, Q-learning agent is Goat
    # mode = 'qlearningBoth'   # Both players are Q-learning agents (auto-play)
    mode = 'goatPlayer'   # Default: You are Goat
    
    game = Main(mode)
    game.start()