#!/usr/bin/env python3
"""
Script per scaricare dati forex estesi dal 2000 ad oggi
Utilizzo: python3 download_extended_data.py
"""

import sys
sys.path.append('core')

from data_loader import download_extended_forex_data
import warnings
warnings.filterwarnings('ignore')

def main():
    """Download extended forex data from 2000 to present."""
    
    print("🚀 DOWNLOAD DATI FOREX ESTESI (2000-2025)")
    print("="*50)
    
    try:
        # Download data dal 2000 ad oggi
        extended_data = download_extended_forex_data(
            start_date="2000-01-01",
            end_date="2025-12-31", 
            save_path="data/forex_extended_data.parquet"
        )
        
        print(f"\n🎯 DOWNLOAD COMPLETATO!")
        print(f"   • File salvato: data/forex_extended_data.parquet")
        print(f"   • Periodo: {extended_data.index.min().date()} - {extended_data.index.max().date()}")
        print(f"   • Giorni: {len(extended_data):,}")
        print(f"   • Coppie forex: {len([col for col in extended_data.columns if '_Close' in col])}")
        
        print(f"\n📋 COME USARE I NUOVI DATI:")
        print(f"   1. Nel notebook, cambia:")
        print(f"      data = load_forex_data('data/forex_synchronized_data.parquet')")
        print(f"      con:")
        print(f"      data = load_forex_data('data/forex_extended_data.parquet')")
        print(f"\n   2. Esegui il notebook normalmente")
        print(f"      Il sistema userà automaticamente i dati dal 2000!")
        
    except Exception as e:
        print(f"❌ ERRORE durante il download: {str(e)}")
        print(f"\n🔧 POSSIBILI SOLUZIONI:")
        print(f"   • Verifica connessione internet")
        print(f"   • Installa yfinance: pip install yfinance")
        print(f"   • Riprova più tardi (Yahoo Finance può avere limiti)")
        
if __name__ == "__main__":
    main()