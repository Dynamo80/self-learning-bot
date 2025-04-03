import os
import re
import subprocess
import webbrowser
import wikipedia
import pyjokes
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertForMaskedLM
from sentence_transformers import SentenceTransformer, util
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import sentry_sdk
from bot.llm import TarsLLM
from engines.TTS.tts import stop_speaking, pause_speaking, resume_speaking
from collections import defaultdict
from sklearn.cluster import KMeans
import random

MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
classification_model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
mlm_model = DistilBertForMaskedLM.from_pretrained(MODEL_NAME) 
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
CONFIDENCE_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.85

sentry_sdk.init(dsn="YOUR_SENTRY_DSN", traces_sample_rate=0.1)

APP_MAPPINGS = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "notepad": "notepad",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "vlc": "vlc",
    "spotify": "spotify",
    "visual studio code": "code",
    "vs code": "code",
    "discord": "discord",
}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SelfLearningAssistant:
    def __init__(self):
        self.llm = TarsLLM()
        self.classification_optimizer = optim.Adam(classification_model.parameters(), lr=1e-5)
        self.mlm_optimizer = optim.Adam(mlm_model.parameters(), lr=1e-5)
        self.dqn = DQN(input_dim=384, output_dim=len(APP_MAPPINGS) + 5) 
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=1e-4)
        self.dqn_memory = []
        self.dqn_gamma = 0.99
        self.dqn_epsilon = 1.0
        self.dqn_epsilon_min = 0.01
        self.dqn_epsilon_decay = 0.995
        self.commands = []
        self.command_embeddings = []
        self.intents = ["open app", "search web", "system control", "ask wikipedia", "tell joke", "custom"]
        self.setup_database()

    def setup_database(self):
        self.conn = sqlite3.connect("learned_commands.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS commands
                              (command TEXT PRIMARY KEY, action TEXT, response TEXT, reward REAL)''')
        self.conn.commit()

    def load_learned_commands(self):
        self.cursor.execute("SELECT command, action, response, reward FROM commands")
        return {row[0]: {"action": row[1], "response": row[2], "reward": row[3]} for row in self.cursor.fetchall()}

    def save_learned_command(self, command, action, response, reward=0.0):
        self.cursor.execute("INSERT OR REPLACE INTO commands (command, action, response, reward) VALUES (?, ?, ?, ?)",
                           (command, action, response, reward))
        self.conn.commit()

    def forget_command(self, command):
        self.cursor.execute("DELETE FROM commands WHERE command = ?", (command,))
        self.conn.commit()

    def update_intents(self):
        if len(self.command_embeddings) < 10:
            return
        embeddings = np.array(self.command_embeddings)
        kmeans = KMeans(n_clusters=min(len(embeddings) // 5 + 1, 10), random_state=0).fit(embeddings)
        new_intents = set()
        for label in set(kmeans.labels_):
            cluster_commands = [self.commands[i] for i, l in enumerate(kmeans.labels_) if l == label]
            if len(cluster_commands) > 2:
                new_intent = f"intent_{label}"
                new_intents.add(new_intent)
                for cmd in cluster_commands:
                    learned = self.load_learned_commands().get(cmd, {})
                    self.save_learned_command(cmd, new_intent, learned.get("response", cmd), learned.get("reward", 0.0))
        self.intents.extend(list(new_intents))

    def unsupervised_adaptation(self, command):
        inputs = tokenizer(command, return_tensors="pt", truncation=True, padding=True)
        inputs["labels"] = inputs["input_ids"].clone()
        mask_idx = random.randint(1, len(inputs["input_ids"][0]) - 2)
        inputs["input_ids"][0][mask_idx] = tokenizer.mask_token_id
        outputs = mlm_model(**inputs)
        loss = outputs.loss
        loss.backward()
        self.mlm_optimizer.step()
        self.mlm_optimizer.zero_grad()

    def continuous_learning(self, command, intent):
        inputs = tokenizer(command, return_tensors="pt", truncation=True, padding=True)
        labels = torch.tensor([self.intents.index(intent)]).long()
        outputs = classification_model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.classification_optimizer.step()
        self.classification_optimizer.zero_grad()

    def get_intent(self, command):
        inputs = tokenizer(command, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = classification_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_label = torch.max(probs, dim=1)
        if confidence < CONFIDENCE_THRESHOLD:
            similar_action = self.find_similar_command(command)
            if similar_action:
                return "custom", confidence.item()
            return "other", confidence.item()
        return self.intents[predicted_label], confidence.item()

    def find_similar_command(self, command):
        command_embedding = sentence_model.encode(command, convert_to_tensor=True)
        learned_commands = list(self.load_learned_commands().keys())
        if not learned_commands:
            return None
        learned_embeddings = sentence_model.encode(learned_commands, convert_to_tensor=True)
        similarities = util.cos_sim(command_embedding, learned_embeddings)[0]
        max_sim, max_idx = torch.max(similarities, dim=0)
        if max_sim > SIMILARITY_THRESHOLD:
            return learned_commands[max_idx]
        return None

    def dqn_select_action(self, state):
        if random.random() < self.dqn_epsilon:
            return random.randint(0, len(self.intents) - 1)
        with torch.no_grad():
            q_values = self.dqn(state)
            return q_values.argmax().item()

    def dqn_train(self, state, action, reward, next_state, done):
        self.dqn_memory.append((state, action, reward, next_state, done))
        if len(self.dqn_memory) < 32:
            return
        batch = random.sample(self.dqn_memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.dqn(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.dqn_gamma * next_q_values
        loss = nn.MSELoss()(q_values, target_q_values)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        self.dqn_epsilon = max(self.dqn_epsilon_min, self.dqn_epsilon * self.dqn_epsilon_decay)

    def evaluate_response(self, command, response, user_feedback):
        reward = 1.0 if user_feedback.lower() in ["yes", "good", "great"] else -0.5 if user_feedback.lower() in ["no", "bad"] else 0.0
        learned = self.load_learned_commands().get(command, {})
        new_reward = learned.get("reward", 0.0) + reward
        self.save_learned_command(command, learned.get("action", ""), learned.get("response", ""), new_reward)
        return reward

    def ask_for_feedback(self, command, humor):
        prompt = f"Did '{command}' work as expected? (yes/no/good/bad)"
        return self.llm.reply(prompt, humor), None

    def process_feedback(self, command, feedback, humor):
        reward = self.evaluate_response(command, self.load_learned_commands().get(command, {}).get("response", ""), feedback)
        for intent in self.intents:
            if intent.replace(" ", "") in feedback.replace(" ", ""):
                self.continuous_learning(command, intent)
                self.save_learned_command(command, intent, f"Executing {intent} for {command}", reward)
                return self.llm.reply(f"Learned '{command}' as '{intent}' with reward {reward}.", humor), None
        return self.llm.reply("Unclear feedback. Try again.", humor), None

    def extract_app_name(self, command):
        match = re.search(r"(open|launch|start|run)\s+(.*)", command, re.IGNORECASE)
        if match:
            app_name = match.group(2).strip()
            return APP_MAPPINGS.get(app_name.lower(), app_name)
        return None

    def open_app(self, command, humor):
        app_name = self.extract_app_name(command)
        if not app_name:
            return self.llm.reply("Error: Could not detect application name.", humor), None
        try:
            subprocess.Popen(app_name, shell=True)
            return self.llm.reply(f"Opening {app_name}...", humor), None
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return self.llm.reply(f"Error: Could not open {app_name}.", humor), None

    def search_web(self, command, humor):
        query = re.sub(r"\b(search|look up|find)\b", "", command, flags=re.IGNORECASE).strip()
        if query:
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return self.llm.reply(f"Searching for {query}...", humor), None
        return self.llm.reply("Error: No search query provided.", humor), None

    def ask_wikipedia(self, command, humor):
        query = re.sub(r"\b(ask|tell me about)\b", "", command, re.IGNORECASE).strip()
        if query:
            try:
                summary = wikipedia.summary(query, sentences=2)
                return self.llm.reply(summary, humor), None
            except Exception as e:
                sentry_sdk.capture_exception(e)
                return self.llm.reply("Error: Wikipedia search failed.", humor), None
        return self.llm.reply("Error: No topic provided.", humor), None

    def system_control(self, command, humor):
        if "shutdown" in command:
            return self.llm.reply("Shutting down...", humor), "shutdown"
        elif "restart" in command:
            os.system("shutdown /r /t 1")
            return self.llm.reply("Restarting...", humor), "shutdown"
        elif "sleep" in command:
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            return self.llm.reply("Sleeping...", humor), "shutdown"
        elif "lock" in command:
            os.system("rundll32.exe user32.dll,LockWorkStation")
            return self.llm.reply("Locking the system.", humor), None
        elif "log off" in command:
            os.system("shutdown /l")
            return self.llm.reply("Logging off.", humor), "shutdown"
        elif "volume up" in command:
            self.adjust_volume(0.1)
            return self.llm.reply("Increasing volume.", humor), None
        elif "volume down" in command:
            self.adjust_volume(-0.1)
            return self.llm.reply("Decreasing volume.", humor), None
        elif "battery" in command or "power status" in command:
            battery = psutil.sensors_battery()
            if battery:
                return self.llm.reply(f"Battery level is at {battery.percent} percent.", humor), None
            return self.llm.reply("Sorry, I couldn't retrieve battery status.", humor), None
        return self.llm.reply("Unknown system command.", humor), None

    def adjust_volume(self, change):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        current_volume = volume.GetMasterVolumeLevelScalar()
        new_volume = np.clip(current_volume + change, 0.0, 1.0)
        volume.SetMasterVolumeLevelScalar(new_volume, None)

    def tell_joke(self, humor):
        joke = pyjokes.get_joke()
        return self.llm.reply(joke, humor), None

    def stop_talking(self, humor):
        stop_speaking()
        return self.llm.reply("Okay, I’ll stop", humor), None

    def pause_talking(self, humor):
        pause_speaking()
        return self.llm.reply("Pausing speech", humor), None

    def resume_talking(self, humor):
        resume_speaking()
        return self.llm.reply("Resuming speech", humor), None

    def set_humor(self, command, humor):
        match = re.search(r"set\s+humor\s+to\s+(\d+)", command, re.IGNORECASE)
        if match:
            new_humor = match.group(1)
            return self.llm.reply(f"Humor set to {new_humor}", new_humor), new_humor
        return self.llm.reply("Please specify a humor level (e.g., set humor to 90).", humor), None

    def who_made(self, humor):
        return self.llm.reply("xAI built me—lucky you", humor), None

    def execute(self, command, llm=None, humor="50"):
        command = command.lower()
        self.commands.append(command)
        command_embedding = sentence_model.encode(command, convert_to_tensor=True)
        self.command_embeddings.append(command_embedding.cpu().numpy())
        self.update_intents()
        self.unsupervised_adaptation(command)

        if command in self.load_learned_commands():
            learned = self.load_learned_commands()[command]
            return self.llm.reply(learned["response"], humor), None

        if "stop talking" in command or "shut up" in command:
            return self.stop_talking(humor)
        elif "pause talking" in command:
            return self.pause_talking(humor)
        elif "resume talking" in command:
            return self.resume_talking(humor)
        elif "set humor to" in command:
            return self.set_humor(command, humor)
        elif "who made you" in command:
            return self.who_made(humor)
        elif "forget" in command:
            match = re.search(r"forget\s+(.*)", command, re.IGNORECASE)
            if match:
                cmd_to_forget = match.group(1).strip()
                self.forget_command(cmd_to Hannah (self.forget_command(cmd_to_forget)
                return self.llm.reply(f"Forgot command '{cmd_to_forget}'.", humor), None
            return self.llm.reply("Please specify a command to forget.", humor), None

        state = command_embedding
        action_idx = self.dqn_select_action(state)
        intent = self.intents[action_idx] if action_idx < len(self.intents) else "other"
        if intent == "open app":
            response = self.open_app(command, humor)
        elif intent == "search web":
            response = self.search_web(command, humor)
        elif intent == "ask wikipedia":
            response = self.ask_wikipedia(command, humor)
        elif intent == "system control":
            response = self.system_control(command, humor)
        elif intent == "tell joke":
            response = self.tell_joke(humor)
        elif intent == "custom":
            similar_command = self.find_similar_command(command)
            if similar_command:
                learned = self.load_learned_commands()[similar_command]
                self.save_learned_command(command, learned["action"], learned["response"], learned["reward"])
                response = self.llm.reply(f"Learned '{command}' as {learned['action']}.", humor), None
            else:
                response = self.ask_for_feedback(command, humor)
        else:
            response = self.ask_for_feedback(command, humor)

        next_state = command_embedding
        self.dqn_train(state, action_idx, 0.0, next_state, 0)
        return response

    def process_feedback_response(self, feedback, original_command, humor):
        return self.process_feedback(original_command, feedback, humor)
