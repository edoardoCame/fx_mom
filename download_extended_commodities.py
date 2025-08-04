#!/usr/bin/env python3
"""
Script per scaricare dati commodities estesi dal 2000 ad oggi
Utilizzo: python3 download_extended_commodities.py
"""

import sys
sys.path.append('core')

from data_loader import download_extended_commodities_data
import warnings
warnings.filterwarnings('ignore')

def main():
    """Download extended commodity data from 2000 to present."""
    print("🚀 DOWNLOAD DATI COMMODITIES ESTESI (2000-2025)")
    print("="*50)
    try:
        extended_data = download_extended_commodities_data(
            start_date="2000-01-01",
            end_date="2025-12-31",
            save_path="data/commodities_extended_data.parquet"
        )
        print(f"\n🎯 DOWNLOAD COMPLETATO!")
        print(f"   • File salvato: data/commodities_extended_data.parquet")
        print(f"   • Periodo: {extended_data.index.min().date()} - {extended_data.index.max().date()}")
        print(f"   • Giorni: {len(extended_data):,}")
        print(f"   • Commodities: {len([col for col in extended_data.columns if '_Close' in col])}")
        print(f"\n📋 COME USARE I NUOVI DATI:")
        print(f"   1. Nel notebook, carica:")
        print(f"      data = pd.read_parquet('data/commodities_extended_data.parquet')")
        print(f"   2. Esegui le tue analisi normalmente")
    except Exception as e:
        print(f"❌ ERRORE durante il download: {str(e)}")
        print(f"\n🔧 POSSIBILI SOLUZIONI:")
        print(f"   • Verifica connessione internet")
        print(f"   • Installa yfinance: pip install yfinance")
        print(f"   • Riprova più tardi (Yahoo Finance può avere limiti)")

if __name__ == "__main__":
    main()
