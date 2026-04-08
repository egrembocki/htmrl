import { Environment, Observation, StepResult } from '../types';

export class FrozenLake implements Environment {
  name = 'FrozenLake-v1';
  private size = 5;
  private agentPos = { x: 0, y: 0 };
  private goalPos = { x: 4, y: 4 };
  private holes = [
    { x: 1, y: 1 },
    { x: 2, y: 2 },
    { x: 3, y: 3 },
    { x: 1, y: 3 },
    { x: 3, y: 1 },
  ];

  reset(): Observation {
    this.agentPos = { x: 0, y: 0 };
    return this.getStateMap();
  }

  private getStateMap(): Observation {
    return {
      'Agent X': this.agentPos.x,
      'Agent Y': this.agentPos.y,
      'Goal X': this.goalPos.x,
      'Goal Y': this.goalPos.y,
    };
  }

  step(action: number): StepResult {
    // 0: Up, 1: Right, 2: Down, 3: Left
    const nextPos = { ...this.agentPos };
    if (action === 0) nextPos.y = Math.max(0, this.agentPos.y - 1);
    else if (action === 1) nextPos.x = Math.min(this.size - 1, this.agentPos.x + 1);
    else if (action === 2) nextPos.y = Math.min(this.size - 1, this.agentPos.y + 1);
    else if (action === 3) nextPos.x = Math.max(0, this.agentPos.x - 1);

    // Check holes
    const hitHole = this.holes.some(hole => hole.x === nextPos.x && hole.y === nextPos.y);
    
    // In FrozenLake, falling into a hole ends the episode
    if (hitHole) {
      this.agentPos = nextPos;
      return {
        observation: this.getStateMap(),
        reward: 0,
        done: true,
        truncated: false,
        info: { status: 'fell_in_hole' },
      };
    }

    this.agentPos = nextPos;
    const reachedGoal = this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y;
    const reward = reachedGoal ? 1 : 0;
    const done = reachedGoal;

    return {
      observation: this.getStateMap(),
      reward,
      done,
      truncated: false,
      info: {},
    };
  }

  render(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const cellSize = Math.min(width, height) / this.size;
    const offsetX = (width - cellSize * this.size) / 2;
    const offsetY = (height - cellSize * this.size) / 2;

    // Draw background (Ice)
    ctx.fillStyle = '#f0f9ff';
    ctx.fillRect(offsetX, offsetY, cellSize * this.size, cellSize * this.size);

    // Draw grid
    ctx.strokeStyle = '#bae6fd';
    ctx.lineWidth = 1;
    for (let i = 0; i <= this.size; i++) {
      ctx.beginPath();
      ctx.moveTo(offsetX + i * cellSize, offsetY);
      ctx.lineTo(offsetX + i * cellSize, offsetY + this.size * cellSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(offsetX, offsetY + i * cellSize);
      ctx.lineTo(offsetX + this.size * cellSize, offsetY + i * cellSize);
      ctx.stroke();
    }

    // Draw holes
    ctx.fillStyle = '#0c4a6e';
    this.holes.forEach(hole => {
      ctx.beginPath();
      ctx.arc(offsetX + hole.x * cellSize + cellSize / 2, offsetY + hole.y * cellSize + cellSize / 2, cellSize / 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw goal (Frisbee/Gift)
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.moveTo(offsetX + this.goalPos.x * cellSize + cellSize / 2, offsetY + this.goalPos.y * cellSize + cellSize / 4);
    ctx.lineTo(offsetX + this.goalPos.x * cellSize + cellSize * 3/4, offsetY + this.goalPos.y * cellSize + cellSize * 3/4);
    ctx.lineTo(offsetX + this.goalPos.x * cellSize + cellSize / 4, offsetY + this.goalPos.y * cellSize + cellSize * 3/4);
    ctx.closePath();
    ctx.fill();

    // Draw agent
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(offsetX + this.agentPos.x * cellSize + cellSize / 2, offsetY + this.agentPos.y * cellSize + cellSize / 2, cellSize / 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  getObservationLabels(): string[] {
    return ['Agent X', 'Agent Y', 'Goal X', 'Goal Y'];
  }

  getActionSpace(): number {
    return 4;
  }
}
