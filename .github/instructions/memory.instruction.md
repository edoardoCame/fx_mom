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

## Strategia Momentum Commodities (agosto 2025)
- Feature: Strategia momentum top n ranking su commodities, vettorizzata, senza lookahead bias
- Ribilanciamento settimanale ogni venerdì, pesi solo su asset con rendimento cumulativo positivo
- Se nessuna commodity ha rendimento positivo, si resta in cash
- Periodo di lookback configurabile
- Implementazione in modulo separato, test rigoroso su dati reali

## Coding Patterns
- Preferred patterns and practices: Funzioni modulari, salvataggio dati in parquet, uso di pandas
- Code organization preferences: Moduli in /core, dati in /data
- Testing approaches: Test manuali e notebook
- Documentation style: Commenti inline e docstring

## Context7 Research History
- Context7 non ha fornito documentazione aggiornata specifica per commodities Yahoo Finance, quindi si usa la libreria yfinance come per il forex.
- Pattern di download: scaricare dati reali con yfinance, concatenare, sincronizzare, salvare in parquet (come in download_extended_forex_data).
- Dal 4 agosto 2025: la funzione commodities ora scarica una lista estesa di future Yahoo Finance (agricoli, energetici, metalli, softs, livestock, lumber, ecc), validata da Wikipedia e pattern Yahoo (SYMBOL=F).
- Edge case gestiti: commodities senza dati o con pochi dati vengono segnalate nel notebook.
- Notebook aggiornato: mostra lista asset disponibili, gestisce errori di caricamento, verifica robustezza database.

## Conversation History
- 4 agosto 2025: Richiesta di creare modulo per scaricare commodities da Yahoo Finance, salvarle in parquet in /data, ispirandosi al modulo forex
- 4 agosto 2025: Analisi codebase: il download forex reale avviene tramite yfinance in download_extended_forex_data (core/data_loader.py). Nessun download reale commodities esistente, solo dati sintetici.
- 4 agosto 2025: Implementata funzione download_extended_commodities_data in core/data_loader.py e script download_extended_commodities.py. File dati: data/commodities_extended_data.parquet.

- 4 agosto 2025: Problema segnalato: equity curve piatta (zero rendimento) nella strategia momentum commodities. Obiettivo: identificare e correggere la causa, validare la soluzione. Prossimi step: ricerca Context7 su bug comuni strategie momentum pandas, debug pipeline dati e segnali, correzione logica se necessario.

## Notes
- Aggiornare la memoria dopo ogni step
- Seguire pattern di download dati forex (yfinance, concat, parquet)
- Nuovo modulo commodities: funzione download_extended_commodities_data, script download_extended_commodities.py, file dati data/commodities_extended_data.parquet
- Istruzioni d’uso: importare e caricare con pd.read_parquet('data/commodities_extended_data.parquet') nei notebook

Prossimo step: Esplorare struttura file commodities_extended_data.parquet, configurare ambiente Python, analizzare dati per strategia momentum
