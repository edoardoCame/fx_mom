"""
Test rapido della strategia momentum commodities su dati reali.
"""
import pandas as pd
from core.commodity_momentum import generate_commodity_momentum_signals

def main():
    df = pd.read_parquet('data/commodities_extended_data.parquet')
    signals = generate_commodity_momentum_signals(
        df,
        lookback_days=60,
        top_n=3,
        rebalance_weekday=4
    )
    print(signals.head(15))
    print(signals.describe())
    # Verifica: almeno una settimana tutta cash se tutti asset negativi
    all_zero_weeks = (signals.sum(axis=1) == 0).resample('W').sum()
    print('Settimane tutte cash:', (all_zero_weeks == 5).sum())

if __name__ == "__main__":
    main()
