import socket

def client_program():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("tcp-server", 9999))  # Connect to the server
    message = client.recv(1024).decode()  # Receive message from the server
    print(f"[CLIENT] Received: {message}")
    client.close()

if __name__ == "__main__":
    client_program()
