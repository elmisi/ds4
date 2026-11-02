# P2B — costo della scansione vittime: ottimizzazione archiviata

Data:2026-10-30. Misurazione preliminare completata. **Nessuna riscrittura LRU e nessuna promozione.**

Autorizzazione: «ok, procedi», dopo proposta di misurare separatamente la scansione. Runtime base0ba37bd99cf3ada39423ab58a23b1e780e6f6358. Scratch `/home/alessandro/projects/ds4-p2b-20261030-231327`, branch `research/p2b-victim-profile-20261030-231327`. Patch opt-in DS4_CUDA_SSD_VICTIM_PROFILE, richiede anche SSD_CACHE_PROFILE. Nessuna modifica a politica, protezioni, scelta vittima, sync CUDA o copie.

| Misura decode64K/256token | Risultato |
| --- | ---: |
| Ricerche vittime | 2374 |
| Slot visitati | 18664388 |
| Slot per ricerca | 7862 |
| Scansione totale | 12,217546ms |
| Scansione per token | 0,047725ms |
| Load per token | 25,116712ms |
| Scansione / load | 0,1900% |
| Scansione / token completo | 0.0476% |

Il ciclo percorre tutti i7862slot a ogni miss nel decode osservato; il costo resta minimo. Eliminarlo interamente recupererebbe circa0,048ms/token, contro circa100ms/token totali. Questa è una stima ideale ricavata dagli intervalli, NON uno speedup misurato; aggiornare un'altra struttura cache avrebbe costi propri. Non esistono dati per giustificare la complessità di una riscrittura LRU su questo carico.

Metodo: due passaggi a65536token popolati/262144allocati,256token teacherforced, cache80GiB, stesso1017 NVMe/Engram Lexar, topk1/stage-sync0/graphs0/affinità normale. Entrambi con profiling loader/token e hook dump logits. A: nuovi contatori spenti. B: nuovi contatori accesi.256vettori completi per braccio,129280F32,132382720byte, finiti e byte identici; anche A è identico al dump originale produzione P2A. Nessun timing pulito e nessun claim throughput.

Associazione dei log verificata:10240record cache/vittime,40layer per ciascuna delle256posizioni65536–65791; searches=misses per record, visite entro limiti capacità, nessun record mancante. Stessi2374miss della precedente diagnostica. Minimo MemAvailable10.56GiB; zero swap processo, nessun incremento pswpin/pswpout, nessun memory-floor/timeout. Quota90%, checker conservativo60% mantenuto.

Strumentazione: due letture di orologio per ricerca; conteggio visite ricavato dall'indice finale del loop, senza incremento per iterazione. Il tempo comprende chiamate orologio e piccola contabilità: una stima con overhead, sufficiente a dimostrare che non è il collo di bottiglia. Tutta la scansione rimane inclusa in load_ms: NON sommare scan_ms e load_ms.

Build: ricompilato solo ds4_cuda.o in worktree separato, nice10/MemoryMax9G, flag archsm_121a; riutilizzati gli altri oggetti runtime e il benchmark dump P1. Link con gencode esplicito per evitare modulo predefinitosm_75; inventario finale di tutti i9cubin sm_121a in cubins.txt. Primo tentativo link scartato prima dell'inferenza. Nessun oggetto o binario produzione sovrascritto. Patch unicamente diagnostica; nessuna modifica ai kernel o lifetime, nessun test sanitizer funzionale presentato come eseguito.

Decisione: **P2B archiviato per beneficio massimo marginale** sul carico misurato. I circa25ms del loader sono quasi interamente fuori dalla ricerca vittime, ma restano letture, staging, copie e attese da distinguere: non chiamarli tutti tempo disco. Non estrapolare questa percentuale a modelli/budget diversi.

Prossima strada dentro P2: P2C, **solo schema stream/eventi e audit lifetime** per capire quale caricamento possa sovrapporsi al calcolo shared-expert single-host. Nessuna rimozione di barriere o implementazione overlap sulla base del solo profilo. In alternativa, resta disponibile P3A80/82GiB come esperimento configurazione indipendente. Nessuno dei due avviato.

Servizio ripristinato tramite dgx-ctl con configurazione originale, senza profiling/dump; raw e scratch preservati, nessun commit/push.

Artifact: `/home/alessandro/projects/ds4-engram-external/speed-bench/p2b-20261030-231327`. initial.json, victim-profile.patch, build-command.json/build.log, object-hashes.json, cubins.txt, execute.py, summarize-victims.py, victim-summary.json, profile-summary.json, gate-pass e runs.json. Provenienza benchmark effettivo in runtime-provenance.json dei raw; file modello e fixture invariati rispetto a P2A, checksum GGUF storico non ricalcolato.

Raw:
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p2b-20261030-231327-gate-A`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p2b-20261030-231327-gate-B`
