import { Environment, Observation, StepResult } from '../types';

export class CartPole implements Environment {
  name = 'CartPole-v1';
  
  // Physics constants
  private gravity = 9.8;
  private masscart = 1.0;
  private masspole = 0.1;
  private total_mass = this.masspole + this.masscart;
  private length = 0.5; // actually half the pole's length
  private polemass_length = this.masspole * this.length;
  private force_mag = 10.0;
  private tau = 0.02; // seconds between state updates
  private kinematics_integrator = 'euler';

  // Angle at which to fail the episode
  private theta_threshold_radians = (12 * 2 * Math.PI) / 360;
  private x_threshold = 2.4;

  // State: [x, x_dot, theta, theta_dot]
  private state: [number, number, number, number] = [0, 0, 0, 0];
  private steps_beyond_done = -1;

  reset(): Observation {
    this.state = [
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
    ];
    this.steps_beyond_done = -1;
    return this.getStateMap();
  }

  private getStateMap(): Observation {
    return {
      'Cart Position': this.state[0],
      'Cart Velocity': this.state[1],
      'Pole Angle': this.state[2],
      'Pole Angular Velocity': this.state[3],
    };
  }

  step(action: number): StepResult {
    let [x, x_dot, theta, theta_dot] = this.state;
    const force = action === 1 ? this.force_mag : -this.force_mag;
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);

    const temp = (force + this.polemass_length * theta_dot * theta_dot * sintheta) / this.total_mass;
    const thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0 / 3.0 - (this.masspole * costheta * costheta) / this.total_mass));
    const xacc = temp - (this.polemass_length * thetaacc * costheta) / this.total_mass;

    if (this.kinematics_integrator === 'euler') {
      x = x + this.tau * x_dot;
      x_dot = x_dot + this.tau * xacc;
      theta = theta + this.tau * theta_dot;
      theta_dot = theta_dot + this.tau * thetaacc;
    } else {
      // semi-implicit euler
      x_dot = x_dot + this.tau * xacc;
      x = x + this.tau * x_dot;
      theta_dot = theta_dot + this.tau * thetaacc;
      theta = theta + this.tau * theta_dot;
    }

    this.state = [x, x_dot, theta, theta_dot];

    const done =
      x < -this.x_threshold ||
      x > this.x_threshold ||
      theta < -this.theta_threshold_radians ||
      theta > this.theta_threshold_radians;

    let reward = 0;
    if (!done) {
      reward = 1.0;
    } else if (this.steps_beyond_done === -1) {
      this.steps_beyond_done = 0;
      reward = 1.0;
    } else {
      this.steps_beyond_done += 1;
      reward = 0.0;
    }

    return {
      observation: this.getStateMap(),
      reward,
      done,
      truncated: false,
      info: {},
    };
  }

  render(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const [x, , theta] = this.state;
    
    const world_width = this.x_threshold * 2;
    const scale = width / world_width;
    const carty = height * 0.7; // 70% down
    const cartx = x * scale + width / 2.0;

    const cartwidth = 50.0;
    const cartheight = 30.0;

    // Draw track
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, carty);
    ctx.lineTo(width, carty);
    ctx.stroke();

    // Draw cart
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(cartx - cartwidth / 2, carty - cartheight / 2, cartwidth, cartheight);

    // Draw pole
    const pole_len = scale * (this.length * 2);
    const pole_top_x = cartx + Math.sin(theta) * pole_len;
    const pole_top_y = carty - Math.cos(theta) * pole_len;

    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cartx, carty);
    ctx.lineTo(pole_top_x, pole_top_y);
    ctx.stroke();

    // Draw axle
    ctx.fillStyle = '#9ca3af';
    ctx.beginPath();
    ctx.arc(cartx, carty, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  getObservationLabels(): string[] {
    return ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'];
  }

  getActionSpace(): number {
    return 2; // 0: Left, 1: Right
  }
}
