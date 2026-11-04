# P2A — profiling del caricamento esperti a64K

Data:2026-10-30. **Diagnostica completata, nessuna modifica di produzione.** Finestra autorizzata dal proprietario: «vai con il terzo».

Runtime0ba37bd99cf3ada39423ab58a23b1e780e6f6358; checkpoint1017, pesi/esperti NVMe ed Engram Lexar. Cache80GiB, contesto popolato65536/allocato262144,256token teacherforced, topk1/stage-sync0/graphs0, affinità normale. Flag P1 assente.

Due passaggi isolati, entrambi con hook dump P1 già verificato: A profiling spento, B DS4_CUDA_V41_DECODE_PROFILE=1 e DS4_CUDA_SSD_CACHE_PROFILE=1. Hook immutato, oggetti CUDA identici a produzione; nessuna compilazione.

Gate:256 vettori completi×129280F32,132382720byte per braccio; finiti e byte identici. Profiling completo:256token,10240record layer e10240cache, esattamente40layer per posizione65536–65791, tutti ok. Parser esistente più verifica indipendente posizione/layer.

| Intervallo nel decode | ms/token |
| --- | ---: |
| Cache prefetch_wait_ms | 0.001030 |
| Cache reuse_sync_ms | 0.017660 |
| Cache prepare_ms | 0.042636 |
| Cache load_ms | 25.097366 |
| Cache finish_ms | 0.086833 |
| Cache total_ms | 25.246559 |

Tempo token medio100,429657ms, mediano95,779845ms. Cache totale25,246559ms (~25,14% del token), di cui load25,097366ms (~24,99%). Attesa Engram0=1,298851ms, Engram1=0,000243ms; submit/serial0,005025ms. Gli intervalli del token contengono quelli del loader: non sommarli come lavoro indipendente.

Miss=2374, frazione3,863932% sugli esperti selezionati; circa9.273miss/token. Letture logiche esperti22,006302GiB nei256token = 88.025MiB/token. Byte logici del loader, non letture fisiche NVMe o banda saturata.

Interpretazione: Engram non domina questo carico, mentre il loader esperti occupa circa un quarto del tempo. prepare (~0,043ms) è piccolo, ma la scelta delle vittime avviene dentro load: NON si può dedurre che la scansione LRU sia trascurabile. load include ricerca slot, letture, upload e completamento upload. Anche reuse_sync piccolo non prova che le letture si possano sovrapporre senza nuovi vincoli.

Prossimo sottoprogetto scelto: **P2B, sola misurazione preliminare del costo della scansione vittime**. Contare scansioni/slot visitati e separarne il tempo dal resto di load in un hook opt-in isolato, prima di progettare strutture LRU. Nessuna riscrittura LRU autorizzata dai soli dati presenti. P2C resta sospeso: progettare overlap solo dopo aver capito quale parte dei25ms sia sovrapponibile e i relativi lifetime.

Confronto storico: il vecchio29,92ms di load e4,619% miss era di altro checkpoint/budget; qui25,10ms e3,864% NON dimostrano un miglioramento causale. I timing P1 a4K e questi a64K hanno frontiere diverse, quindi nessun confronto A/B di velocità.

Il profiling non è una misura di throughput. Inoltre entrambi i run includono dump logits: i CSV vengono conservati come diagnostica, non come timing pulito. I tempi A/B e la loro differenza non giustificano speedup. Campi prefill/whole-layer-prefetch a0 nel riepilogo significano record non emessi da questi flag, NON che il prefill o il loader non abbiano lavorato.

Minimo MemAvailable10.19GiB; nessuno swap processo o incremento contatori swap, entrambi i runner exit0 e senza stop. Quota90%, controllo conservativo60% mantenuto. OS page cache non controllata; nessuna lettura integrale dei341GiB per checksum.

Servizio ripristinato tramite dgx-ctl con configurazione originale; nessun profiling o dump abilitato nel processo servizio. Nessun commit/push.

Artifact: `/home/alessandro/projects/ds4-engram-external/speed-bench/p2a-20261030-225956`. Provenienza aggiornata initial.json; hook/oggetti/cubin da P1; execute.py, runs.json, profile-summary.json, derived.json. SHA effettivo benchmark per ogni raw in runtime-provenance.json. Identità GGUF da path/dimensione/mtime precedenti e checksum storico, senza nuovo hash integrale.

Raw:
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p2a-20261030-225956-gate-A`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p2a-20261030-225956-gate-B`
