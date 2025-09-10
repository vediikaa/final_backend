# main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import pickle
import threading
from typing import List, Optional, Any, Dict

# Import your RL environment
from environment_ag_20220402 import Env

# ---- Load model at startup ----
with open("dqn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---- Initialize environment (shared env instance) ----
NETWORK_FILE = "simple_network.txt"
TRAIN_FILE = "simple_train.txt"
MAX_DELAY = 600

env = Env(NETWORK_FILE, TRAIN_FILE, MAX_DELAY)

# Use the same dimensions used in training
STATE_SIZE = 2844
ACTION_SIZE = 50

# Lock to avoid concurrent env mutations (Env is stateful)
env_lock = threading.Lock()

app = FastAPI(title="DQN Inference + Simulation API")


class StateInput(BaseModel):
    state: List[float]   # the state vector (length must match STATE_SIZE)


@app.post("/predict")
def predict(input_data: StateInput):
    if len(input_data.state) != STATE_SIZE:
        raise HTTPException(status_code=400, detail=f"state length must be {STATE_SIZE}")
    state_array = np.array(input_data.state, dtype=np.float32).reshape(1, -1)
    q_values = model.predict(state_array)
    action = int(np.argmax(q_values))
    return {"q_values": q_values.tolist(), "chosen_action": action}


@app.get("/simulate")
def simulate_episode(
    max_steps: int = Query(5000, description="absolute max steps to run (safety)"),
    max_logged_steps: int = Query(1000, description="maximum number of steps to include in the returned log"),
    log_every: int = Query(1, description="save every `log_every`-th step in the log (1 = every step)"),
    include_q: bool = Query(False, description="include q_values in each logged step (adds ~50 floats per step)")
):
    """
    Run a greedy-policy episode (model picks argmax Q) and return:
      - summary: steps, total_reward, done
      - actions_taken (full sequence)
      - logs: list of step entries { step, action, reward, state, [q_values] }

    Notes:
      - The `state` vector returned at each step is a Python list of length STATE_SIZE.
      - Large responses are possible; tune `max_logged_steps` and `log_every` to reduce payload.
    """
    # run with lock to prevent concurrent modification of env
    with env_lock:
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        actions_taken: List[int] = []
        logs: List[Dict[str, Any]] = []

        # Basic validation for initial state length
        if getattr(state, "__len__", None) is None or len(state) != STATE_SIZE:
            # If state is not the expected shape, still proceed but warn
            raise HTTPException(status_code=500, detail=f"Reset state length mismatch: expected {STATE_SIZE}, got {len(state) if hasattr(state, '__len__') else 'unknown'}")

        while not done and steps < max_steps:
            # reshape for model input
            state_input = np.reshape(state, [1, STATE_SIZE])
            q_values = model.predict(state_input)  # shape (1, ACTION_SIZE)
            action = int(np.argmax(q_values))

            # step env
            next_state, reward, done = env.step(action)

            # bookkeeping
            total_reward += reward
            steps += 1
            actions_taken.append(action)

            # decide whether to log this step
            if len(logs) < max_logged_steps and (steps % max(1, log_every) == 0):
                entry: Dict[str, Any] = {
                    "step": steps,
                    "action": action,
                    "reward": reward,
                    "state": state.tolist() if isinstance(state, np.ndarray) else list(state),
                }
                if include_q:
                    entry["q_values"] = q_values[0].tolist()
                logs.append(entry)

            # move to next
            state = next_state

            # safety early exit if env.step never returns terminal (shouldn't normally happen)
            if steps >= max_steps:
                break

    # return everything JSON-serializable
    return {
        "summary": {
            "steps": steps,
            "total_reward": total_reward,
            "done": bool(done)
        },
        # actions_taken can be large, but user asked for step-by-step logs; we include it as summary too
        "actions_taken_count": len(actions_taken),
        "actions_taken_sample": actions_taken[:max(100, len(actions_taken))],  # small sample; full actions are represented in logs
        "logs_returned": len(logs),
        "logs": logs
    }
