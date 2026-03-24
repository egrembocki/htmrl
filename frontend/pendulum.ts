import { Environment, Observation, StepResult } from '../types';

export class Pendulum implements Environment {
  name = 'Pendulum-v1';
  
  private max_speed = 8;
  private max_torque = 2.0;
  private dt = 0.05;
  private g = 10.0;
  private m = 1.0;
  private l = 1.0;

  private state: [number, number] = [0, 0]; // [theta, theta_dot]

  reset(): Observation {
    const high = [Math.PI, 1];
    this.state = [
      (Math.random() * 2 - 1) * high[0],
      (Math.random() * 2 - 1) * high[1],
    ];
    return this.getStateMap();
  }

  private getStateMap(): Observation {
    const [theta, theta_dot] = this.state;
    return {
      'cos(theta)': Math.cos(theta),
      'sin(theta)': Math.sin(theta),
      'theta_dot': theta_dot,
    };
  }

  step(action: number): StepResult {
    // Action is discrete in our simplified version for consistency
    // 0: -max_torque, 1: 0, 2: +max_torque
    const u = (action - 1) * this.max_torque;
    
    let [th, thdot] = this.state;
    const g = this.g;
    const m = this.m;
    const l = this.l;
    const dt = this.dt;

    const costs = this.angleNormalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2;

    let newthdot = thdot + (3 * g / (2 * l) * Math.sin(th) + 3.0 / (m * l ** 2) * u) * dt;
    newthdot = Math.max(-this.max_speed, Math.min(this.max_speed, newthdot));
    let newth = th + newthdot * dt;

    this.state = [newth, newthdot];

    return {
      observation: this.getStateMap(),
      reward: -costs,
      done: false, // Pendulum is usually truncated, not done
      truncated: false,
      info: {},
    };
  }

  private angleNormalize(x: number): number {
    return ((x + Math.PI) % (2 * Math.PI)) - Math.PI;
  }

  render(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const [theta] = this.state;
    const centerX = width / 2;
    const centerY = height / 2;
    const length = Math.min(width, height) * 0.4;

    const endX = centerX + Math.sin(theta) * length;
    const endY = centerY + Math.cos(theta) * length;

    // Draw pivot
    ctx.fillStyle = '#94a3b8';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
    ctx.fill();

    // Draw arm
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw weight
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(endX, endY, 12, 0, Math.PI * 2);
    ctx.fill();
  }

  getObservationLabels(): string[] {
    return ['cos(theta)', 'sin(theta)', 'theta_dot'];
  }

  getActionSpace(): number {
    return 3; // -2, 0, 2 torque
  }
}
