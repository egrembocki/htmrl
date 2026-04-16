import { Environment, Observation, StepResult } from '../types';

export class GymTrading implements Environment {
  name = 'gym_trading_env';
  
  private price = 100.0;
  private ownedShares = 0.0;
  private balance = 10000.0;
  private stepCount = 0;
  private maxSteps = 500;

  reset(): Observation {
    this.price = 100.0;
    this.ownedShares = 0.0;
    this.balance = 10000.0;
    this.stepCount = 0;
    return this.getStateMap();
  }

  private getStateMap(): Observation {
    return {
      'price': this.price,
      'owned_shares': this.ownedShares,
      'balance': this.balance,
    };
  }

  step(action: number): StepResult {
    // action: 0: Hold, 1: Buy, 2: Sell
    const oldPrice = this.price;
    // Simple random walk for price
    this.price += (Math.random() - 0.5) * 2.0;
    this.price = Math.max(1, this.price);

    let reward = 0;
    if (action === 1) { // Buy
      if (this.balance >= this.price) {
        this.balance -= this.price;
        this.ownedShares += 1;
      }
    } else if (action === 2) { // Sell
      if (this.ownedShares > 0) {
        this.balance += this.price;
        this.ownedShares -= 1;
        reward = this.price - oldPrice; // Profit/loss on the share sold
      }
    }

    this.stepCount++;
    const done = this.stepCount >= this.maxSteps;
    
    // Total portfolio value change as reward? Or just balance change.
    // Let's use portfolio value change.
    const portfolioValue = this.balance + this.ownedShares * this.price;
    // Actually, let's just use a simple reward for now.
    
    return {
      observation: this.getStateMap(),
      reward: reward,
      done,
      truncated: false,
      info: { portfolio_value: portfolioValue },
    };
  }

  render(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    // Draw a simple price chart
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Display current stats
    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(`Price: $${this.price.toFixed(2)}`, 20, 30);
    ctx.fillText(`Shares: ${this.ownedShares}`, 20, 55);
    ctx.fillText(`Balance: $${this.balance.toFixed(2)}`, 20, 80);
    
    const portfolioValue = this.balance + this.ownedShares * this.price;
    ctx.fillStyle = '#2563eb';
    ctx.fillText(`Total Value: $${portfolioValue.toFixed(2)}`, 20, 110);

    // Draw a simple "trading floor" or something
    ctx.fillStyle = '#334155';
    ctx.fillRect(width / 2 - 50, height - 100, 100, 60);
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    ctx.fillText("EXCHANGE", width / 2 - 35, height - 65);
  }

  getObservationLabels(): string[] {
    return ['price', 'owned_shares', 'balance'];
  }

  getActionSpace(): number {
    return 3; // 0: Hold, 1: Buy, 2: Sell
  }
}
