# import socket module
import sys 
from socket import *

serverSocket = socket(AF_INET,SOCK_STREAM) #IPv4, TCP socket 생성
#Assign a port number 
severPort = 6789

#Bind the socket to address and server port
serverSocket.bind(('',severPort))
#Listen to at most 1 connection at a time 
serverSocket.listen(1)

while 1:
	print('The server is ready to receive')
	# set up a new connection from the client 
	connectionSocket, addr = serverSocket.accept()
	try:
		# Receives the request message from the client 
		message = connectionSocket.recv(1024).decode()
		# Exact the path of the requested object from the message
		# The path is the second part of HTTP header, identified by [1]
		filename = message.split()[1]
		# Because the extracted path of the request includes
		# a character '\', reaf the path frrom the second character
		f = open(filename[1:])
		# Store the entire contenet of the requensted file in a temporart buffer 
		outputdata = f.read()
		# Send the HTTP request header line to the connection Socket 
		connectionSocket.send("HTTP/1.1 200 OK\r\n\r\n".encode())
		
		# Send the content of the requested file to the connection socket
		for i in range(0, len(outputdata)):
			connectionSocket.send(outputdata[i].encode())
		connectionSocket.send('\r\n'.encode())

		# close the client connection socket
		connectionSocket.close()
		
	except IOError:
			#send HTTP request message for file not found
			connectionSocket.send("HTTP/1.1 404 Not Found\r\n\r\n".encode())
			connectionSocket.send("<html><head></head><body><h1>404 Not Found</h1></body></html>\r\n".encode())
			# close the client connection socket
			connectionSocket.close()

serverSocket.close()
sys.exit() #Terminate the program after sending the corresponding data