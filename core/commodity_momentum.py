"""
Modulo: commodity_momentum.py
Strategia momentum top n ranking su commodities, vettorizzata, senza lookahead bias.
"""
import pandas as pd
import numpy as np

def generate_commodity_momentum_signals(
    price_data: pd.DataFrame,
    lookback_days: int = 60,
    top_n: int = 3,
    rebalance_weekday: int = 4  # 4 = Friday
) -> pd.DataFrame:
    """
    Genera segnali momentum per commodities senza lookahead bias.
    Ogni venerdì calcola i rendimenti cumulativi su lookback_days,
    seleziona le top n con rendimento positivo, assegna pesi proporzionali.
    Se nessuna è positiva, resta in cash.
    I pesi sono applicati per tutta la settimana successiva.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Dati OHLCV con colonne tipo GOLD_Close, SILVER_Close, ...
    lookback_days : int
        Periodo di lookback per rendimento cumulativo
    top_n : int
        Numero massimo di commodities da selezionare
    rebalance_weekday : int
        Giorno di ribilanciamento (default: 4 = venerdì)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con pesi giornalieri per ogni commodity
    """
    close_cols = [c for c in price_data.columns if c.endswith('_Close')]
    asset_names = [c.replace('_Close', '') for c in close_cols]
    weight_cols = [f"{a}_weight" for a in asset_names]
    signals = pd.DataFrame(0.0, index=price_data.index, columns=weight_cols)
    close_prices = price_data[close_cols].copy()
    close_prices.columns = asset_names
    # Trova tutti i venerdì
    rebalance_dates = price_data.index[price_data.index.weekday == rebalance_weekday]
    for i, date in enumerate(rebalance_dates):
        if price_data.index.get_loc(date) < lookback_days:
            continue  # skip se non c'è abbastanza storia
        # Calcola rendimento cumulativo su lookback
        window = close_prices.loc[:date].iloc[-lookback_days:]
        returns = window.iloc[-1] / window.iloc[0] - 1
        # Filtra solo asset con rendimento positivo
        positive = returns[returns > 0]
        if len(positive) == 0:
            # Tutto cash: pesi zero
            period_start = date
            period_end = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else price_data.index[-1]
            period_range = price_data.loc[period_start:period_end].index
            signals.loc[period_range, :] = 0.0
            continue
        # Seleziona top n
        top = positive.sort_values(ascending=False).head(top_n)
        # Pesi proporzionali ai rendimenti (normalizzati)
        weights = top / top.sum()
        # Applica pesi per la settimana successiva
        period_start = date
        # Escludi il giorno di ribilanciamento successivo dal periodo di applicazione dei pesi
        if i+1 < len(rebalance_dates):
            period_end = rebalance_dates[i+1]
            period_range = price_data.loc[period_start:period_end].index[:-1]  # escludi il prossimo rebalance
        else:
            period_end = price_data.index[-1]
            period_range = price_data.loc[period_start:period_end].index
        for asset, w in weights.items():
            signals.loc[period_range, f"{asset}_weight"] = w
        # Tutti gli altri a zero (già di default)
    return signals
