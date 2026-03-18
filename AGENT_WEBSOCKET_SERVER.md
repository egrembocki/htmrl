# Agent WebSocket Server

Real-time bidirectional communication between Python RL agents and web visualization clients.

## Quick Start

### 1. Start the Server

```bash
python run_agent_server.py

# Or with custom host/port:
python run_agent_server.py --host 0.0.0.0 --port 9000 --env CartPole-v1
```

### 2. Connect Web Client

The web visualization client connects to `ws://localhost:8765` and receives:
- **Ready message** on connection
- **Episode start** with initial observation when `start_episode` is sent
- **Step results** with action, reward, and status after each step
- **Episode summary** when episode ends

## WebSocket Protocol

### Message Types

#### Client → Server

**start_episode** - Start a new episode
```json
{"type": "start_episode"}
```

**reset** - Alias for start_episode
```json
{"type": "reset"}
```

**step** - Run one timestep with inputs
```json
{"type": "step", "inputs": {"obs_0": 0.5, "obs_1": -0.2}}
```

**stop** - Stop the current episode
```json
{"type": "stop"}
```

**status** - Query episode state
```json
{"type": "status"}
```

#### Server → Client

**ready** - Server is ready for connections
```json
{"type": "ready", "message": "Server ready"}
```

**episode_start** - New episode initialized
```json
{
  "type": "episode_start",
  "episode": 1,
  "obs": [0.05, -0.12, 0.08, 0.02],
  "inputs": {"position_input": [0, 1, 0, ...], "velocity_input": [...]}
}
```

**step_result** - Action and metrics from a step
```json
{
  "type": "step_result",
  "step": 42,
  "action": 1,
  "reward": 1.0,
  "terminated": false,
  "truncated": false,
  "total_reward": 42.0,
  "episode_done": false
}
```

**error** - Error response
```json
{"type": "error", "message": "Error description"}
```

## Architecture

- **Per-client state**: Each WebSocket connection maintains independent episode state
- **Multiple clients**: Server handles multiple concurrent visualization clients
- **Real-time learning**: Agent learns (or not) based on transitions provided
- **JSON communication**: All messages use JSON for web compatibility

## Integration with Python Code

```python
from psu_capstone.agent_layer.agent_server import AgentWebSocketServer
from psu_capstone.agent_layer.agent import Agent

# Build your agent
agent = Agent(brain=..., adapter=..., policy_mode="brain")

# Create and run server
server = AgentWebSocketServer(agent, host="0.0.0.0", port=8765)
server.run()  # Blocks until shutdown
```

## Files

- `agent_server.py` - Core WebSocket server implementation
- `run_agent_server.py` - CLI entry point
- `websocket_client_example.js` - Example JavaScript client (see repo root)

## Requirements

- Python 3.11+
- `websockets>=13.0` (added to pyproject.toml)
- All psu_capstone dependencies
