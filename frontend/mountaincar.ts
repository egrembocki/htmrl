import { Environment, Observation, StepResult } from '../types';

export class MountainCar implements Environment {
  name = 'MountainCar-v0';
  
  private min_position = -1.2;
  private max_position = 0.6;
  private max_speed = 0.07;
  private goal_position = 0.5;
  private goal_velocity = 0;

  private force = 0.001;
  private gravity = 0.0025;

  private position = -0.5;
  private velocity = 0;

  reset(): Observation {
    this.position = Math.random() * 0.2 - 0.6;
    this.velocity = 0;
    return this.getStateMap();
  }

  private getStateMap(): Observation {
    return {
      'Position': this.position,
      'Velocity': this.velocity,
    };
  }

  step(action: number): StepResult {
    // 0: Left, 1: Nothing, 2: Right
    this.velocity += (action - 1) * this.force + Math.cos(3 * this.position) * (-this.gravity);
    this.velocity = Math.max(-this.max_speed, Math.min(this.max_speed, this.velocity));
    
    this.position += this.velocity;
    this.position = Math.max(this.min_position, Math.min(this.max_position, this.position));

    if (this.position === this.min_position && this.velocity < 0) {
      this.velocity = 0;
    }

    const done = this.position >= this.goal_position && this.velocity >= this.goal_velocity;
    const reward = -1.0;

    return {
      observation: this.getStateMap(),
      reward,
      done,
      truncated: false,
      info: {},
    };
  }

  render(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const scale = width / (this.max_position - this.min_position);
    const ground_y = height * 0.8;

    // Draw terrain
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x <= width; x++) {
      const pos = this.min_position + (x / width) * (this.max_position - this.min_position);
      const y = ground_y - Math.sin(3 * pos) * 50;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw goal flag
    const goal_x = (this.goal_position - this.min_position) * scale;
    const goal_y = ground_y - Math.sin(3 * this.goal_position) * 50;
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(goal_x, goal_y);
    ctx.lineTo(goal_x, goal_y - 30);
    ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(goal_x, goal_y - 30);
    ctx.lineTo(goal_x + 15, goal_y - 22);
    ctx.lineTo(goal_x, goal_y - 15);
    ctx.fill();

    // Draw car
    const car_x = (this.position - this.min_position) * scale;
    const car_y = ground_y - Math.sin(3 * this.position) * 50;
    const angle = Math.atan(Math.cos(3 * this.position) * 3);

    ctx.save();
    ctx.translate(car_x, car_y);
    ctx.rotate(-angle);
    
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(-10, -15, 20, 10);
    
    // Wheels
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.arc(-6, -5, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(6, -5, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.restore();
  }

  getObservationLabels(): string[] {
    return ['Position', 'Velocity'];
  }

  getActionSpace(): number {
    return 3; // 0: Left, 1: None, 2: Right
  }
}
