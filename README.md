🧠 Synthetic Cognitive Substrate (V1.0)
Standard LLMs (like ChatGPT or Claude) are passive "language cortices"—they sit inert until prompted, predict the next likely token, and immediately lose their state when the session ends. They have no continuous memory, no active agency, and no internal check on whether they are hallucinating.

This project is a Continuous Active-Inference Engine. It transforms a static LLM into an autonomous, self-auditing artificial mind. It formulates hypotheses, writes Python code to test them in a local sandbox, uses a decoupled "Inner Critic" to ruthlessly audit the empirical results, and saves verified algorithms into a permanent tool library.

⚙️ The Cognitive Architecture
The system mimics a functional biological brain using modular Python components and Pydantic structured schemas:

The Brainstem (Homeostasis): The system has an internal epistemic_energy budget. Testing hypotheses costs energy; discovering verified truths replenishes it. If the system hallucinates or fails, it loses energy.

The Thalamus (Global Workspace): An attention router that looks at energy levels, past repetition, and memory saturation to broadcast a single urgent directive to the rest of the brain.

Prefrontal Cortex (Planner): Generates concrete mathematical/engineering hypotheses and writes executable Python code to test them.

Actuator (Sandbox): Runs the AI's Python code locally on your CPU (e.g., signal processing, telemetry analysis, market simulations) to force a collision with empirical reality.

Second-Order Monitor (Inner Critic): A ruthlessly prompted metacognitive layer that reads the sandbox terminal output. If the code crashed, faked data, or failed its prediction, the Critic rejects the thought.

Hippocampus (Consolidation): Periodically enters a "Sleep Cycle" to distill core engineering principles, update its first-person Self-Model, and purge repetitive thoughts using a deterministic Jaccard similarity filter.

🚀 Features
Zero-Hallucination Verification: Claims are only accepted into long-term memory if the local Python execution proves them mathematically or empirically.

Cumulative Tool Building: Verified code is automatically saved into a skills/ directory as reusable .py modules. The brain literally builds its own software library over time.

Live Web Dashboard: Includes a visualizer.html interface to watch the brain's attention routing, energy levels, and memory consolidation in real-time.

Anti-Echo-Chamber Logic: Python-native Jaccard word-similarity checks prevent the model from ruminating on the same philosophical phrases (the "Markov Blanket trap").

🛠 Installation & Setup
1. Clone the repository and set up a virtual environment:

Bash
git clone https://github.com/yourusername/synthetic-cognitive-substrate.git
cd synthetic-cognitive-substrate
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
2. Install dependencies:

Bash
pip install google-genai pydantic python-dotenv yfinance numpy scipy
3. Set your API Key:
Create a .env file in the root directory and add your Gemini API key:

Code snippet
GEMINI_API_KEY="your_api_key_here"
🧠 Running the Brain
To experience the full cognitive loop, you should run the engine and the visual dashboard side-by-side.

Terminal 1 (Start the Visualizer):

Bash
python3 -m http.server 8000
# Open http://localhost:8000/visualizer.html in your browser
Terminal 2 (Start the Engine):

Bash
python brain.py
Watch the terminal as the Planner writes code, the Sandbox executes it, and the Inner Critic audits the terminal output. Verified skills will automatically appear in your skills/ folder, and the visualizer will update live.

🔮 Roadmap & The Next Frontier
Currently, the engine has mastered building local signal-processing filters (Savitzky-Golay, MAD outlier detection, CVaR risk estimators).

To define the next grand objective and push this epistemic engine toward high-stakes, real-world complexity, we will refer to the Missing Interesting Future document. This will guide our transition from a local mathematical sandbox into a continuous intelligence layer capable of parsing live internet telemetry, autonomous operations, and complex multi-agent market dynamics.

🤝 Contributing
This is an experimental blueprint for autonomous AI. Pull requests addressing memory graph optimization, expanded sensory web-scrapers, or continuous analog LIF (Leaky Integrate-and-Fire) spike-triggers are highly encouraged.

“The way to figure out what to work on is by working. If you're not sure what to work on, guess. But pick something and get going.” — Paul Graham
