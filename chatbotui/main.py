from http.server import HTTPServer, BaseHTTPRequestHandler
import re
from urllib.parse import parse_qs
import html

# HTML template for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .chat-container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .chat-history {
            margin-bottom: 20px;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .bot-message {
            background-color: #f5f5f5;
        }
        form {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 8px 16px;
            background-color: #2196f3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #1976d2;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>Python Chatbot</h1>
        <div class="chat-history">
            {chat_history}
        </div>
        <form method="POST">
            <input type="text" name="user_input" placeholder="Type your message..." required>
            <button type="submit">Send</button>
        </form>
    </div>
</body>
</html>
"""

def chatbot_response(user_input):
    user_input = user_input.lower()
    
    if re.search(r"\b(hello|hi)\b", user_input):
        return "Hello! How can I assist you today?"
    elif re.search(r"\bhow are you\b", user_input):
        return "I'm just a chatbot, but I'm doing great! How can I help?"
    elif re.search(r"\byour name\b", user_input):
        return "I'm your friendly chatbot! You can call me Chatbot."
    elif re.search(r"\b(bye|goodbye)\b", user_input):
        return "Goodbye! Have a great day!"
    elif re.search(r"\bhelp\b", user_input):
        return "Sure, I can help! Please let me know what you need help with."
    else:
        return "Sorry, I didn't understand that. Can you rephrase?"

class ChatbotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Display the chat interface without any messages
        html_content = HTML_TEMPLATE.format(chat_history="")
        self.wfile.write(html_content.encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        user_input = parse_qs(post_data)['user_input'][0]
        
        # Get chatbot response
        response = chatbot_response(user_input)
        
        # Create chat history HTML
        chat_history = f"""
        <div class="message user-message">
            <strong>You:</strong> {html.escape(user_input)}
        </div>
        <div class="message bot-message">
            <strong>Bot:</strong> {html.escape(response)}
        </div>
        """
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html_content = HTML_TEMPLATE.format(chat_history=chat_history)
        self.wfile.write(html_content.encode())

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ChatbotHandler)
    print(f"Server running at http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()