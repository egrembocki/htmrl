import { Environment, Observation, StepResult } from '../types';

export class LunarLander implements Environment {
  name = 'LunarLander-v3';
  
  // Physics constants
  private gravity = -0.001;
  private main_engine_power = 0.0015;
  private side_engine_power = 0.0006;
  
  // State: [x, y, x_vel, y_vel, angle, angular_vel, left_leg, right_leg]
  private state = {
    x: 0,
    y: 0.8,
    x_vel: 0,
    y_vel: 0,
    angle: 0,
    angular_vel: 0,
    left_leg: 0,
    right_leg: 0
  };

  private stepCount = 0;
  private maxSteps = 1000;

  reset(): Observation {
    this.state = {
      x: (Math.random() - 0.5) * 0.2,
      y: 0.8,
      x_vel: (Math.random() - 0.5) * 0.01,
      y_vel: (Math.random() - 0.5) * 0.01,
      angle: (Math.random() - 0.5) * 0.1,
      angular_vel: (Math.random() - 0.5) * 0.01,
      left_leg: 0,
      right_leg: 0
    };
    this.stepCount = 0;
    return this.getStateMap();
  }

  private getStateMap(): Observation {
    return {
      'x_position': this.state.x,
      'y_position': this.state.y,
      'x_velocity': this.state.x_vel,
      'y_velocity': this.state.y_vel,
      'angle': this.state.angle,
      'angular_velocity': this.state.angular_vel,
      'left_leg_contact': this.state.left_leg,
      'right_leg_contact': this.state.right_leg,
    };
  }

  step(action: number): StepResult {
    // action: 0: Do nothing, 1: Fire left engine, 2: Fire main engine, 3: Fire right engine
    
    // Apply gravity
    this.state.y_vel += this.gravity;
    
    // Apply engines
    if (action === 2) { // Main engine
      this.state.x_vel -= Math.sin(this.state.angle) * this.main_engine_power;
      this.state.y_vel += Math.cos(this.state.angle) * this.main_engine_power;
    } else if (action === 1) { // Left engine (fires right)
      this.state.angular_vel += 0.001;
    } else if (action === 3) { // Right engine (fires left)
      this.state.angular_vel -= 0.001;
    }

    // Update position
    this.state.x += this.state.x_vel;
    this.state.y += this.state.y_vel;
    this.state.angle += this.state.angular_vel;

    // Ground contact
    let done = false;
    let reward = 0;
    
    if (this.state.y <= 0) {
      this.state.y = 0;
      this.state.y_vel = 0;
      this.state.x_vel = 0;
      this.state.angular_vel = 0;
      done = true;
      
      // Check if landing was soft and upright
      const speed = Math.sqrt(this.state.x_vel**2 + this.state.y_vel**2);
      if (speed < 0.01 && Math.abs(this.state.angle) < 0.1 && Math.abs(this.state.x) < 0.1) {
        reward = 100; // Success
      } else {
        reward = -100; // Crash
      }
    }

    this.stepCount++;
    if (this.stepCount >= this.maxSteps) {
      done = true;
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
    // Draw sky
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Draw stars
    ctx.fillStyle = '#ffffff';
    for (let i = 0; i < 50; i++) {
        const x = (Math.sin(i * 123.45) * 0.5 + 0.5) * width;
        const y = (Math.cos(i * 678.90) * 0.5 + 0.5) * height * 0.7;
        ctx.fillRect(x, y, 1, 1);
    }

    // Draw ground
    ctx.fillStyle = '#334155';
    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(0, height - 20);
    ctx.lineTo(width / 2 - 40, height - 20); // Landing pad
    ctx.lineTo(width / 2 + 40, height - 20);
    ctx.lineTo(width, height - 20);
    ctx.lineTo(width, height);
    ctx.fill();

    // Draw landing pad
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 2;
    ctx.strokeRect(width / 2 - 40, height - 22, 80, 2);

    // Draw lander
    const landerX = width / 2 + this.state.x * (width / 2);
    const landerY = height - 20 - this.state.y * (height - 40);
    
    ctx.save();
    ctx.translate(landerX, landerY);
    ctx.rotate(this.state.angle);

    // Lander body
    ctx.fillStyle = '#94a3b8';
    ctx.fillRect(-10, -15, 20, 15);
    
    // Legs
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-10, 0);
    ctx.lineTo(-15, 5);
    ctx.moveTo(10, 0);
    ctx.lineTo(15, 5);
    ctx.stroke();

    ctx.restore();
  }

  getObservationLabels(): string[] {
    return [
      'x_position', 'y_position', 'x_velocity', 'y_velocity',
      'angle', 'angular_velocity', 'left_leg_contact', 'right_leg_contact'
    ];
  }

  getActionSpace(): number {
    return 4; // 0: Do nothing, 1: Fire left engine, 2: Fire main engine, 3: Fire right engine
  }
}
