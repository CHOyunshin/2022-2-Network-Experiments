# import socket module
from socket import *
import sys # in order to terminate the program
import time

serverSocket = socket(AF_INET, SOCK_STREAM)

# Assign a port number
serverPort = 7002

# Bind the socket to server and server port
# 윈도우 기준 cmd 키고 ipconfig 치면 바로 주소 확인가능 
serverSocket.bind(('172.20.10.4', serverPort)) 

# Listen to at most 1 connection at a time
serverSocket.listen(1)
while True:
    print('The server is ready to receive')
    clientSocket, client_addr = serverSocket.accept()
    print('Connected by', client_addr)     
    try:
        message = clientSocket.recv(4096).decode()       
        print('Received from', client_addr, message)
        filename = message
        f = open(filename)
        data_iter = 0
        with open(filename, 'rb') as f:
            # try:
            outputdata = f.read()
            while outputdata:
                serverSocket.send(outputdata)
                data_iter = data_iter+1
                outputdata = f.read(4096)
                print("파일 %s (seq : %d) is sent" %(filename, data_iter))
                time.sleep(5)
    except Exception as ex:
        print(ex)
serverSocket.close()