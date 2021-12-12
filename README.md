# Chessposition-recognition
 The goal of this project is demonstrate a method for computers to recognize the state of chess board given an image. The method involves first finding the location of each piece in an image by identifying the chess grid using Canny Edge Detection and Hough Line Transform. The second half of the models takes each piece or empty square on the board and gives a prediction for its type using a ResNet18. Working together both parts are able to parse the state of board from an image. 

#Files 
Read the paper (Paper.pdf) for an explanation of the different components of the project. It also includes preliminary results and analysis. Additionally, it cites some areas to change/add to that would help flesh out and improve the project (I plan on getting to them at some point and recording the new results). 

advancedBoardDetector.py takes an image, identifies the chess board and returns 64 encodings to represent where each piece is located in the original image. 
baseline.py is the first CNN used to classify individual pieces; however, it is quite limited and only returns type instead of type and color.  
chessPieceIdentification.ipynb contains the final pre-trained ResNet18 that was tuned on a chess piece dataset. 
