---
applyTo: '**'
---

# User Memory

## User Preferences
- Programming languages: Python
- Code style preferences: Seguire lo stile esistente nel progetto, nomi chiari e snake_case
- Development environment: VS Code su macOS, shell zsh
- Communication style: Concisa, spiegazioni tecniche solo se necessario

## Project Context
- Current project type: Trading system/data pipeline
- Tech stack: Python, pandas, parquet, Yahoo Finance API (da integrare per commodities)
- Architecture patterns: Moduli separati per data loading, signal generation, backtest
- Key requirements: Scaricare dati delle commodities più famose e liquide da Yahoo Finance, salvarle in parquet in /data, mantenere coerenza con il modulo forex

## Coding Patterns
- Preferred patterns and practices: Funzioni modulari, salvataggio dati in parquet, uso di pandas
- Code organization preferences: Moduli in /core, dati in /data
- Testing approaches: Test manuali e notebook
- Documentation style: Commenti inline e docstring

## Context7 Research History
- Context7 non ha fornito documentazione aggiornata specifica per commodities Yahoo Finance, quindi si userà la libreria yfinance come per il forex.
- Pattern di download: scaricare dati reali con yfinance, concatenare, sincronizzare, salvare in parquet (come in download_extended_forex_data).
- Simboli commodities usati: GC=F (Gold), SI=F (Silver), CL=F (Crude Oil WTI), BZ=F (Brent), NG=F (NatGas), HG=F (Copper), PL=F (Platinum), PA=F (Palladium), ZC=F (Corn), ZS=F (Soybeans), ZW=F (Wheat), KC=F (Coffee), SB=F (Sugar), CT=F (Cotton).

## Conversation History
- 4 agosto 2025: Richiesta di creare modulo per scaricare commodities da Yahoo Finance, salvarle in parquet in /data, ispirandosi al modulo forex
- 4 agosto 2025: Analisi codebase: il download forex reale avviene tramite yfinance in download_extended_forex_data (core/data_loader.py). Nessun download reale commodities esistente, solo dati sintetici.
- 4 agosto 2025: Implementata funzione download_extended_commodities_data in core/data_loader.py e script download_extended_commodities.py. File dati: data/commodities_extended_data.parquet.

## Notes
- Aggiornare la memoria dopo ogni step
- Seguire pattern di download dati forex (yfinance, concat, parquet)
- Nuovo modulo commodities: funzione download_extended_commodities_data, script download_extended_commodities.py, file dati data/commodities_extended_data.parquet
- Istruzioni d’uso: importare e caricare con pd.read_parquet('data/commodities_extended_data.parquet') nei notebook
