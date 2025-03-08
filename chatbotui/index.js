import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Configure Express
app.use(express.urlencoded({ extended: true }));
app.set('view engine', 'ejs');
app.set('views', join(__dirname, 'views'));

function chatbotResponse(userInput) {
  const input = userInput.toLowerCase();
  
  if (/\b(hello|hi)\b/.test(input)) {
    return "Hello! How can I assist you today?";
  } else if (/\bhow are you\b/.test(input)) {
    return "I'm just a chatbot, but I'm doing great! How can I help?";
  } else if (/\byour name\b/.test(input)) {
    return "I'm your friendly chatbot! You can call me Chatbot.";
  } else if (/\b(bye|goodbye)\b/.test(input)) {
    return "Goodbye! Have a great day!";
  } else if (/\bhelp\b/.test(input)) {
    return "Sure, I can help! Please let me know what you need help with.";
  } else {
    return "Sorry, I didn't understand that. Can you rephrase?";
  }
}

app.get('/', (req, res) => {
  res.render('index', { userInput: undefined, response: undefined });
});

app.post('/', (req, res) => {
  const userInput = req.body.userInput;
  const response = chatbotResponse(userInput);
  res.render('index', { userInput, response });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Chatbot server running at http://localhost:${port}`);
});