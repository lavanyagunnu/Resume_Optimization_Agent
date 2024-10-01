import socket
import threading

def handle_client(client_socket, client_address, message):
    print(f"[SERVER] Connected to {client_address}")
    client_socket.send(message.encode())  # Send message to the client
    client_socket.close()

def server_program():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9999))
    server.listen(3)  # Listen for up to 3 clients
    print("[SERVER] Waiting for connections...")

    messages = ["Message for Client 1", "Message for Client 2", "Message for Client 3"]
    clients = []

    for i in range(3):
        client_socket, client_address = server.accept()
        clients.append((client_socket, client_address))

    for i in range(3):
        client_socket, client_address = clients[i]
        threading.Thread(target=handle_client, args=(client_socket, client_address, messages[i])).start()

if __name__ == "__main__":
    server_program()
