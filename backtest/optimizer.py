from itertools import product
import numpy as np
from backtest.backtester import run_backtest

def grid_search():
    # Mini-Grid (nur Beispiel)
    tfs = ["1m"]
    limits = [1500]
    best = None
    for tf, lim in product(tfs, limits):
        curve = run_backtest("BTC/USDT", tf, lim)
        if not curve: continue
        final = curve[-1]
        sharpe = np.mean(np.diff(curve)) / (np.std(np.diff(curve)) + 1e-9)
        score = final + 0.2 * sharpe
        if best is None or score > best[0]:
            best = (score, final, sharpe, tf, lim)
    return best

if __name__ == "__main__":
    print(grid_search())