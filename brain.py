import os
import sys
import json
import time
import subprocess
import urllib.request
from typing import Dict, Any, Type, Set
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load API key and environment variables
load_dotenv()

# =====================================================================
# 0. ANTIGRAVITY AGENT RUNTIME ADAPTER (antigravity-preview-latest)
# =====================================================================

MODEL_NAME = "antigravity-preview-latest"
STATE_FILE = "world_state.json"
SKILLS_DIR = "skills"

os.makedirs(SKILLS_DIR, exist_ok=True)

class AntigravityAgentRuntime:
    """
    Runtime adapter for 'antigravity-preview-latest' with automatic fallback
    to Google GenAI SDK or local OpenAI-compatible proxy endpoints.
    """
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        self.api_key = (
            os.environ.get("ANTIGRAVITY_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = os.environ.get("ANTIGRAVITY_BASE_URL", "http://localhost:4242/v1")

    def dispatch_structured_step(
        self,
        agent_role: str,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.6,
        max_retries: int = 5
    ) -> BaseModel:
        # 1. Native google.antigravity SDK if installed
        try:
            from google.antigravity import Agent  # type: ignore
            agent = Agent(model=self.model, role=agent_role)
            res = agent.run(prompt=prompt, response_schema=schema, temperature=temperature)
            if isinstance(res, schema):
                return res
            return schema.model_validate_json(res if isinstance(res, str) else json.dumps(res))
        except (ImportError, AttributeError):
            pass

        # 2. Google GenAI SDK bridge
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=f"[AGENT_ROLE: {agent_role}]\n{prompt}",
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=temperature
                        )
                    )
                    raw_text = response.text.strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    elif raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                    return schema.model_validate_json(raw_text.strip())
                except Exception as ex:
                    if attempt < max_retries - 1 and ("503" in str(ex) or "429" in str(ex) or "UNAVAILABLE" in str(ex)):
                        time.sleep(2 ** attempt + 2)
                        continue
                    # Fallback to active flash model if preview cluster is temporarily unreachable
                    try:
                        fallback_model = "gemini-3.5-flash-lite"
                        response = client.models.generate_content(
                            model=fallback_model,
                            contents=f"[TARGET_MODEL: {self.model} | ROLE: {agent_role}]\n{prompt}",
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=schema,
                                temperature=temperature
                            )
                        )
                        raw_text = response.text.strip()
                        if raw_text.startswith("```json"):
                            raw_text = raw_text[7:]
                        elif raw_text.startswith("```"):
                            raw_text = raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]
                        return schema.model_validate_json(raw_text.strip())
                    except Exception as inner_ex:
                        raise inner_ex
        except ImportError:
            pass

        # 3. Standard HTTP JSON endpoint
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"Role: {agent_role}. Output strict JSON matching schema: {json.dumps(schema.model_json_schema())}"
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return schema.model_validate_json(content)


ag_runtime = AntigravityAgentRuntime(model=MODEL_NAME)


# =====================================================================
# 1. DETERMINISTIC ANTI-ECHO-CHAMBER FILTER (Python Math, Not LLM)
# =====================================================================

def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Calculates word-level overlap between two strings to prevent the
    agent from saving repetitive rewordings of the same principle.
    """
    stop_words = {"the", "a", "an", "is", "of", "and", "to", "in", "by", "with", "that", "for", "across", "its"}
    words_a: Set[str] = {w.strip(".,;:()").lower() for w in text_a.split()} - stop_words
    words_b: Set[str] = {w.strip(".,;:()").lower() for w in text_b.split()} - stop_words
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a.intersection(words_b))
    union = len(words_a.union(words_b))
    return intersection / union


def deduplicate_principles(principles: list[str], threshold: float = 0.35) -> list[str]:
    """Purges near-duplicate sentences from existing world_state.json."""
    unique = []
    for p in principles:
        if all(compute_jaccard_similarity(p, existing) < threshold for existing in unique):
            unique.append(p)
    return unique


# =====================================================================
# 2. STRUCTURED SCHEMAS (Empirical Verification & Tool Evolution)
# =====================================================================

class CounterfactualPlan(BaseModel):
    skill_name: str = Field(
        description="Short snake_case name for the reusable Python module (e.g., 'kalman_anomaly_detector', 'orderbook_entropy_estimator')."
    )
    hypothesis: str = Field(
        description="A concrete, testable engineering or statistical hypothesis (e.g., signal filtering, network graph resilience, anomaly detection, or market regime shifts)."
    )
    expected_prediction: str = Field(
        description="Exact quantitative metric predicted BEFORE running the code (e.g., RMSE reduction > 25%, precision > 0.85, p-value < 0.01)."
    )
    python_verification_code: str = Field(
        description="Self-contained Python code (using numpy, scipy, math, or yfinance) that defines a reusable function and runs a benchmark proving the prediction."
    )
    confidence_score: float = Field(description="Initial confidence (0.0 to 1.0).")

class MetacognitiveAudit(BaseModel):
    prediction_matched_reality: bool = Field(
        description="Did the actual terminal output match the pre-execution prediction?"
    )
    is_genuinely_novel: bool = Field(
        description="Is this a distinct algorithm/domain from recent verified claims (True), or a repetitive variation of an old test (False)?"
    )
    identified_blindspots: str = Field(
        description="Technical code review: critique sample size, edge cases, prediction error, or repetition."
    )
    calibrated_certainty: float = Field(
        description="Objective quality score (0.0 to 1.0). Must be < 0.50 if execution failed, faked numbers, or repeated past topics."
    )
    decision: str = Field(description="Strictly 'ACCEPT' (if >= 0.75) or 'REJECT' (if < 0.75).")

class ConsolidationUpdate(BaseModel):
    synthesized_engineering_principle: str = Field(
        description="A concise, practical mathematical or systems-engineering rule distilled from recent benchmarks (avoid philosophical jargon)."
    )
    updated_system_summary: str = Field(
        description="Clear, grounded technical summary of the agent's current library of verified skills, accuracy rate, and active engineering bottlenecks."
    )


# =====================================================================
# 3. PERSISTENT STATE & AUTOMATIC ECHO-CHAMBER CLEANUP
# =====================================================================

def load_world_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            
            # Automatically clean up the Cycle 82 Markov Blanket echo chamber
            old_principles = state.get("core_principles", [])
            cleaned_principles = deduplicate_principles(old_principles, threshold=0.35)
            if len(cleaned_principles) < len(old_principles):
                print(f"[*] Echo-Chamber Filter: Pruned {len(old_principles) - len(cleaned_principles)} repetitive principles from memory.")
                state["core_principles"] = cleaned_principles

            state["objective"] = (
                "Autonomously design, benchmark, and evolve a library of verified Python algorithms "
                "across signal processing, statistical anomaly detection, network optimization, and empirical telemetry."
            )
            state.setdefault("verified_skills", [])
            state.setdefault("epistemic_energy", 100)
            return state

    return {
        "cycle_count": 0,
        "epistemic_energy": 100,
        "active_region": "IDLE",
        "thalamus_broadcast": "System Boot: Initializing cumulative tool-building & empirical benchmark loop.",
        "objective": "Autonomously design, benchmark, and evolve a library of verified Python algorithms.",
        "self_model": "Autonomous algorithm discovery & verification pipeline. Evaluates hypotheses via code execution and builds a persistent library of reusable Python modules.",
        "core_principles": [
            "Every accepted hypothesis must produce a reusable, benchmarked Python module.",
            "Semantic repetition (>35% word overlap) is automatically blocked by the memory filter."
        ],
        "verified_skills": [],
        "verified_beliefs": [],
        "failed_hypotheses": [],
        "last_prediction": "",
        "last_sandbox_output": "",
        "last_inner_dialogue": ""
    }

def save_world_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def set_active_region(state: Dict[str, Any], region: str):
    state["active_region"] = region
    save_world_state(state)


# =====================================================================
# 4. TASK ROUTER (Enforcing Domain Diversity)
# =====================================================================

def thalamus_router(state: Dict[str, Any]) -> str:
    set_active_region(state, "THALAMUS_ROUTER")
    cycles = state["cycle_count"]
    energy = state["epistemic_energy"]
    existing_skills = state.get("verified_skills", [])

    if cycles > 0 and cycles % 4 == 0:
        broadcast = "URGENCY [0.98] - MEMORY CONSOLIDATION: Summarize verified skill performance and prune redundant heuristics."
    elif energy <= 40:
        broadcast = f"URGENCY [0.95] - LOW ENERGY ({energy}%): Build a clean, deterministic numerical optimization or signal-filtering module."
    else:
        engineering_domains = [
            "DOMAIN 1 (Signal Processing & Filtering): Build and benchmark a 1D Kalman Filter or Savitzky-Golay filter vs. raw noisy telemetry.",
            "DOMAIN 2 (Graph & Network Topology): Simulate cascading load failures on a scale-free network vs. random Erdős–Rényi graph and compute routing resilience.",
            "DOMAIN 3 (Statistical Regime Change Detection): Build a CUSUM (Cumulative Sum) or Bayesian change-point detector to identify sudden variance shifts in time-series data.",
            "DOMAIN 4 (Compression & Information Bottlenecks): Benchmark delta-encoding + zlib compression ratios on structured time-series vs. white noise.",
            "DOMAIN 5 (Monte Carlo Risk & Tail Estimation): Compare Conditional Value-at-Risk (CVaR) under Student-t fat tails vs. Gaussian assumptions."
        ]
        target = engineering_domains[cycles % len(engineering_domains)]
        broadcast = (
            f"URGENCY [0.85] - SKILL EVOLUTION: Existing skills ({len(existing_skills)}): {existing_skills[-5:]}. "
            f"FORBIDDEN from repeating Markov blanket philosophy. Next task: {target}"
        )

    state["thalamus_broadcast"] = broadcast
    save_world_state(state)
    return broadcast


# =====================================================================
# 5. PLANNER, SANDBOX, CRITIC, & SKILL PERSISTENCE
# =====================================================================

def prefrontal_planner(state: Dict[str, Any]) -> CounterfactualPlan:
    set_active_region(state, "PREFRONTAL_PLANNER")
    prompt = f"""
    You are an Autonomous Algorithm Designer and Systems Engineer running on {MODEL_NAME}.
    
    [CURRENT DIRECTIVE]:
    {state['thalamus_broadcast']}

    [EXISTING SKILLS IN LIBRARY (DO NOT DUPLICATE)]:
    {json.dumps(state.get('verified_skills', []))}

    [RECENT VERIFIED BENCHMARKS]:
    {json.dumps(state['verified_beliefs'][-3:])}

    [RECENT FAILED ATTEMPTS]:
    {json.dumps(state['failed_hypotheses'][-2:])}

    DIRECTIVE:
    1. Propose a concrete algorithmic hypothesis matching the CURRENT DIRECTIVE. Do NOT use philosophical buzzwords or repeat Markov blanket tests.
    2. State a quantitative pre-execution prediction (exact error bound, speedup, or statistical threshold).
    3. Write clean, self-contained Python code (using numpy/scipy/math) with a reusable function and a `__main__` benchmark block that prints the exact metrics.
    """
    return ag_runtime.dispatch_structured_step(
        agent_role="Algorithm_Planner",
        prompt=prompt,
        schema=CounterfactualPlan,
        temperature=0.6
    )

def execute_in_sandbox(state: Dict[str, Any], code_str: str) -> str:
    set_active_region(state, "ACTUATOR_SANDBOX")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_str],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = f"SUCCESS_OUTPUT:\n{result.stdout.strip()}" if result.returncode == 0 else f"EXECUTION_ERROR:\n{result.stderr.strip()}"
    except Exception as e:
        output = f"SANDBOX_CRASH: {str(e)}"

    state["last_sandbox_output"] = output
    save_world_state(state)
    return output

def inner_critic_monitor(state: Dict[str, Any], plan: CounterfactualPlan, exec_output: str) -> MetacognitiveAudit:
    set_active_region(state, "INNER_CRITIC")
    prompt = f"""
    You are an Automated Code & Benchmark Auditor running on {MODEL_NAME}.

    DIRECTIVE: {state['thalamus_broadcast']}
    EXISTING SKILLS: {json.dumps(state.get('verified_skills', []))}
    PROPOSED SKILL NAME: {plan.skill_name}
    HYPOTHESIS: {plan.hypothesis}
    PREDICTION: {plan.expected_prediction}
    CODE:
    {plan.python_verification_code}

    SANDBOX OUTPUT:
    {exec_output}

    STRICT AUDIT RULES:
    1. If SANDBOX OUTPUT contains EXECUTION_ERROR or SANDBOX_CRASH -> REJECT (score 0.1).
    2. If the hypothesis repeats "Markov blanket" or philosophical prose -> REJECT (score 0.1).
    3. Did the empirical numerical output match the pre-execution PREDICTION? If not -> REJECT (score 0.35).
    4. Only ACCEPT (score >= 0.75) if the algorithm is mathematically sound, non-trivial, and verified by the output.
    """
    return ag_runtime.dispatch_structured_step(
        agent_role="Code_Auditor",
        prompt=prompt,
        schema=MetacognitiveAudit,
        temperature=0.1
    )

def consolidate_memory(state: Dict[str, Any]):
    set_active_region(state, "HIPPOCAMPUS_SLEEP")
    print("\n[🌙 CONSOLIDATION] Reviewing verified skills and updating engineering principles...")
    prompt = f"""
    Review the system's recent algorithmic benchmarks:
    - Cycle: {state['cycle_count']}
    - Verified Skills Built: {json.dumps(state.get('verified_skills', []))}
    - Recent Verified Claims: {json.dumps(state['verified_beliefs'][-4:])}
    - Current Principles: {json.dumps(state['core_principles'])}

    Tasks:
    1. Distill ONE concise, practical engineering/mathematical rule from the recent benchmarks (strictly distinct from Current Principles; zero philosophical jargon).
    2. Write a grounded technical status summary of what tools this codebase has built and verified so far.
    """
    update: ConsolidationUpdate = ag_runtime.dispatch_structured_step(
        agent_role="Memory_Consolidator",
        prompt=prompt,
        schema=ConsolidationUpdate,
        temperature=0.3
    )

    new_p = update.synthesized_engineering_principle
    # Deterministic Python similarity guard
    if all(compute_jaccard_similarity(new_p, existing) < 0.35 for existing in state["core_principles"]):
        state["core_principles"].append(new_p)
        state["core_principles"] = state["core_principles"][-6:]
        print(f"    [+] Added Novel Engineering Principle: {new_p}")
    else:
        print(f"    [!] Blocked Redundant Principle (failed Jaccard novelty check): {new_p}")

    # Sleep restores energy
    state["epistemic_energy"] = min(100, state["epistemic_energy"] + 35)
    state["self_model"] = update.updated_system_summary
    save_world_state(state)


# =====================================================================
# 6. MAIN EXECUTION LOOP
# =====================================================================

def run_cognitive_cycle():
    state = load_world_state()
    state["cycle_count"] += 1
    cycle = state["cycle_count"]
    state["epistemic_energy"] = max(0, state["epistemic_energy"] - 5)

    print(f"\n==================== CYCLE {cycle} | MODEL: {MODEL_NAME} | ENERGY: {state['epistemic_energy']}% ====================")

    broadcast = thalamus_router(state)
    print(f"[1. ROUTER]: {broadcast}")

    if "MEMORY CONSOLIDATION" in broadcast:
        consolidate_memory(state)

    print("\n[2. PLANNER]: Designing algorithm & benchmark...")
    plan = prefrontal_planner(state)
    state["last_prediction"] = plan.expected_prediction
    save_world_state(state)
    print(f"    Skill Module: {plan.skill_name}.py")
    print(f"    Hypothesis:   {plan.hypothesis}")
    print(f"    Prediction:   {plan.expected_prediction}")

    print("\n[3. SANDBOX]: Running benchmark on CPU...")
    exec_output = execute_in_sandbox(state, plan.python_verification_code)
    print(f"    Output:\n    {exec_output[:300]}")

    print("\n[4. AUDITOR]: Checking prediction accuracy & code novelty...")
    audit = inner_critic_monitor(state, plan, exec_output)
    print(f"    Review: {audit.identified_blindspots}")
    print(f"    Score:  {audit.calibrated_certainty} | Decision: {audit.decision}")

    if audit.decision == "ACCEPT" and audit.calibrated_certainty >= 0.75:
        state["epistemic_energy"] = min(100, state["epistemic_energy"] + 25)
        
        # Save verified code as a permanent, reusable Python module in skills/
        safe_name = "".join(c for c in plan.skill_name if c.isalnum() or c == "_").lower() or f"skill_{cycle}"
        skill_path = os.path.join(SKILLS_DIR, f"{safe_name}.py")
        with open(skill_path, "w") as sf:
            sf.write(f'"""\nVerified in Cycle {cycle} (Score: {audit.calibrated_certainty})\nHypothesis: {plan.hypothesis}\n"""\n\n')
            sf.write(plan.python_verification_code)

        if safe_name not in state["verified_skills"]:
            state["verified_skills"].append(safe_name)

        print(f"\n[✅ VERIFIED & SAVED]: Module written to '{skill_path}'! Energy: {state['epistemic_energy']}%.")
        state["verified_beliefs"].append({
            "cycle": cycle,
            "skill": safe_name,
            "claim": plan.hypothesis,
            "prediction": plan.expected_prediction,
            "empirical_output": exec_output,
            "certainty": audit.calibrated_certainty
        })
    else:
        state["epistemic_energy"] = max(0, state["epistemic_energy"] - 10)
        print(f"\n[❌ REJECTED]: Benchmark failed audit. Energy: {state['epistemic_energy']}%.")
        state["failed_hypotheses"].append({
            "cycle": cycle,
            "rejected_claim": plan.hypothesis,
            "sandbox_output": exec_output,
            "flaw": audit.identified_blindspots
        })

    state["last_inner_dialogue"] = audit.identified_blindspots
    set_active_region(state, "IDLE")


if __name__ == "__main__":
    print(f"[*] Booting Evolution Pipeline ({MODEL_NAME})...")
    for _ in range(5):
        run_cognitive_cycle()
        time.sleep(3)
