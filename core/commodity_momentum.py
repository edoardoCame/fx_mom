"""
Modulo: commodity_momentum.py
Strategia momentum top n ranking su commodities, completamente vettorizzata, senza lookahead bias.
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
    
    LOGICA:
    - Ogni venerdì, calcola il rendimento cumulativo degli ultimi lookback_days
    - Seleziona le top_n commodities con rendimento positivo più alto
    - Assegna pesi proporzionali ai rendimenti
    - Applica i pesi dal LUNEDÌ successivo
    
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
    # Estrai prezzi di chiusura
    close_cols = [c for c in price_data.columns if c.endswith('_Close')]
    asset_names = [c.replace('_Close', '') for c in close_cols]
    close_prices = price_data[close_cols].copy()
    close_prices.columns = asset_names
    
    # Inizializza DataFrame dei pesi
    weight_cols = [f"{asset}_weight" for asset in asset_names]
    signals = pd.DataFrame(0.0, index=price_data.index, columns=weight_cols)
    
    # Identifica i giorni di ribilanciamento
    rebalance_dates = price_data.index[price_data.index.weekday == rebalance_weekday]
    
    # Per ogni data di ribilanciamento
    for i, rebal_date in enumerate(rebalance_dates):
        # Verifica che ci siano abbastanza dati storici
        rebal_idx = price_data.index.get_loc(rebal_date)
        if rebal_idx < lookback_days:
            continue
            
        # Calcola rendimento cumulativo: (prezzo_oggi / prezzo_lookback_giorni_fa) - 1
        # Usa dati fino al giorno PRIMA del ribilanciamento per evitare lookahead
        end_price = close_prices.iloc[rebal_idx - 1]  # Prezzo del giorno precedente
        start_price = close_prices.iloc[rebal_idx - 1 - lookback_days]  # Prezzo lookback giorni prima
        
        # Calcola rendimenti cumulativi
        cumulative_returns = (end_price / start_price) - 1
        
        # Filtra solo asset con rendimento positivo
        positive_returns = cumulative_returns[cumulative_returns > 0]
        
        # Determina il periodo di applicazione dei pesi (dal primo giorno di trading DOPO il ribilanciamento)
        try:
            # Trova il primo giorno di trading dopo il ribilanciamento
            next_trading_day_idx = rebal_idx + 1
            while next_trading_day_idx < len(price_data.index):
                next_trading_day = price_data.index[next_trading_day_idx]
                break
            else:
                continue  # Non ci sono giorni di trading successivi
                
            # Trova la fine del periodo (prossimo ribilanciamento o fine dati)
            if i + 1 < len(rebalance_dates):
                next_rebal_date = rebalance_dates[i + 1]
                end_idx = price_data.index.get_loc(next_rebal_date)
            else:
                end_idx = len(price_data.index)
            
            # Periodo di applicazione: dal giorno dopo il ribilanciamento fino al prossimo ribilanciamento (escluso)
            period_slice = slice(next_trading_day_idx, end_idx)
            
        except (IndexError, KeyError):
            continue
        
        # Reset tutti i pesi a zero per questo periodo
        signals.iloc[period_slice, :] = 0.0
        
        # Se ci sono asset con rendimento positivo, assegna i pesi
        if len(positive_returns) > 0:
            # Seleziona top n asset
            top_assets = positive_returns.nlargest(top_n)
            
            # Calcola pesi proporzionali ai rendimenti (normalizzati)
            weights = top_assets / top_assets.sum()
            
            # Applica i pesi
            for asset, weight in weights.items():
                col_idx = weight_cols.index(f"{asset}_weight")
                signals.iloc[period_slice, col_idx] = weight
    
    return signals


def generate_equal_weighted_commodity_momentum_signals(
    price_data: pd.DataFrame,
    lookback_days: int = 60,
    top_n: int = 3,
    rebalance_weekday: int = 4  # 4 = Friday
) -> pd.DataFrame:
    """
    Genera segnali momentum EQUI-PESATI per commodities senza lookahead bias.
    
    LOGICA:
    - Ogni venerdì, calcola il rendimento cumulativo degli ultimi lookback_days
    - Seleziona le top_n commodities con rendimento positivo più alto
    - Assegna PESO UGUALE a tutte le commodities selezionate
    - Applica i pesi dal LUNEDÌ successivo
    
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
        DataFrame con pesi giornalieri equi-pesati per ogni commodity
    """
    # Estrai prezzi di chiusura
    close_cols = [c for c in price_data.columns if c.endswith('_Close')]
    asset_names = [c.replace('_Close', '') for c in close_cols]
    close_prices = price_data[close_cols].copy()
    close_prices.columns = asset_names
    
    # Inizializza DataFrame dei pesi
    weight_cols = [f"{asset}_weight" for asset in asset_names]
    signals = pd.DataFrame(0.0, index=price_data.index, columns=weight_cols)
    
    # Identifica i giorni di ribilanciamento
    rebalance_dates = price_data.index[price_data.index.weekday == rebalance_weekday]
    
    # Per ogni data di ribilanciamento
    for i, rebal_date in enumerate(rebalance_dates):
        # Verifica che ci siano abbastanza dati storici
        rebal_idx = price_data.index.get_loc(rebal_date)
        if rebal_idx < lookback_days:
            continue
            
        # Calcola rendimento cumulativo usando dati fino al giorno PRIMA del ribilanciamento
        end_price = close_prices.iloc[rebal_idx - 1]  # Prezzo del giorno precedente
        start_price = close_prices.iloc[rebal_idx - 1 - lookback_days]  # Prezzo lookback giorni prima
        
        # Calcola rendimenti cumulativi
        cumulative_returns = (end_price / start_price) - 1
        
        # Filtra solo asset con rendimento positivo e dati validi
        positive_returns = cumulative_returns[cumulative_returns > 0]
        
        # Determina il periodo di applicazione dei pesi (dal primo giorno di trading DOPO il ribilanciamento)
        try:
            # Trova il primo giorno di trading dopo il ribilanciamento
            next_trading_day_idx = rebal_idx + 1
            while next_trading_day_idx < len(price_data.index):
                next_trading_day = price_data.index[next_trading_day_idx]
                break
            else:
                continue  # Non ci sono giorni di trading successivi
                
            # Trova la fine del periodo (prossimo ribilanciamento o fine dati)
            if i + 1 < len(rebalance_dates):
                next_rebal_date = rebalance_dates[i + 1]
                end_idx = price_data.index.get_loc(next_rebal_date)
            else:
                end_idx = len(price_data.index)
            
            # Periodo di applicazione: dal giorno dopo il ribilanciamento fino al prossimo ribilanciamento (escluso)
            period_slice = slice(next_trading_day_idx, end_idx)
            
        except (IndexError, KeyError):
            continue
        
        # Reset tutti i pesi a zero per questo periodo
        signals.iloc[period_slice, :] = 0.0
        
        # Se ci sono asset con rendimento positivo, assegna PESI EQUI-PESATI
        if len(positive_returns) > 0:
            # Seleziona top n asset
            top_assets = positive_returns.nlargest(top_n)
            
            # Assegna peso uguale a tutti gli asset selezionati
            equal_weight = 1.0 / len(top_assets)
            
            # Applica i pesi
            for asset in top_assets.index:
                col_idx = weight_cols.index(f"{asset}_weight")
                signals.iloc[period_slice, col_idx] = equal_weight
    
    return signals


def calculate_portfolio_performance(signals: pd.DataFrame, price_data: pd.DataFrame) -> pd.Series:
    """
    Calcola la performance del portafoglio basata sui segnali.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Segnali di peso per ogni asset
    price_data : pd.DataFrame
        Dati dei prezzi
        
    Returns:
    --------
    pd.Series
        Serie temporale della equity del portafoglio
    """
    # Estrai prezzi di chiusura
    close_cols = [c for c in price_data.columns if c.endswith('_Close')]
    close_prices = price_data[close_cols].copy()
    close_prices.columns = [c.replace('_Close', '') for c in close_cols]
    
    # Calcola rendimenti giornalieri
    daily_returns = close_prices.pct_change().fillna(0)
    
    # Allinea le colonne dei segnali con quelle dei rendimenti
    signals_aligned = signals.copy()
    signals_aligned.columns = [c.replace('_weight', '') for c in signals_aligned.columns]
    
    # Calcola rendimento del portafoglio
    # Usa i pesi del giorno per i rendimenti del giorno (no shift aggiuntivo)
    portfolio_returns = (signals_aligned * daily_returns).sum(axis=1)
    
    # Calcola equity cumulativa
    equity = (1 + portfolio_returns).cumprod()
    equity.name = 'Portfolio_Equity'
    
    return equity
