# P4B — affinità ai core veloci: risultato neutro

Data: 2026-10-30. **Screening concluso, nessuna promozione.** Secondo esperimento nell’ordine preparazione → P1 → P4B → P2A.

Confronto A: scheduling normale (0–19); B: taskset5–9,15–19. Capacity/frequenze sysfs riconfermate: core scelti3,9GHz, gli altri2,808GHz. Affinità campionata ogni secondo sui thread reali: baseline0–19, candidato5–9,15–19; otto thread osservati per run. Nessuna modifica al servizio per imporre affinità.

Gate numerico4K: entrambi i dump256×129280 float32=132382720byte, completi, finiti e identici byte per byte. Riutilizzato hook P1 immutato in binario separato. Misure temporali con ds4-bench originale, senza dump/profiling.

| Ordine | Prefill t/s | Decode t/s | Steady decode t/s |
| --- | --- | --- | --- |
| timing-1-A | 98.70 | 9.98 | 10.07 |
| timing-1-B | 98.62 | 9.96 | 10.05 |
| timing-2-B | 99.67 | 9.96 | 10.04 |
| timing-2-A | 99.63 | 9.96 | 10.04 |

AB: decode−0,20%; BA:0,00%. Media/mediana A9,97t/s, B9,96t/s (−0,10%; circa100,30 contro100,40ms/token). Range A9,96–9,98, B9,96–9,96. Prefill medio A99,165, B99,145t/s (−0,02%); range A98,70–99,63, B98,62–99,67. Il tempo prefill+decode ricavato dai CSV è circa67,0s in entrambi; nessun guadagno utile. Wall runner70,03–72,05s, con startup/campionamento, media71,05s A e71,03s B. CSV arrotondato a0,01t/s: percentuali minime non sono significative.

Minimo MemAvailable sei run: 11.98GiB. Nessuno swap del processo né incremento pswpin/pswpout durante i run. Tutti i runner exit0, nessun memory-floor/timeout.

Configurazione invariata:1017, pesi/esperti NVMe, Engram Lexar, cache80GiB (72,88dinamici,7862slot), contesto allocato262144/popolato4096,256token teacherforced. Topk1, stage-sync0, graphs0; ordinamento decode P1 assente in entrambi. Quota checker conservativo60% mantenuto:90% disponibile durante tutta la suite. Page cache OS non controllata: nessuna dichiarazione disco freddo.

Decisione: neutro allo screening4K, scheduling OS mantenuto. Nessun run lungo, recall o generazione libera: nessun candidato viene promosso. Non si esclude un effetto su carichi diversi, ma questi dati non giustificano modifiche alla produzione né ulteriori prove automatiche.

Servizio fermato/ripristinato tramite dgx-ctl nella finestra autorizzata. Binari runtime identici a P1; nessuna ricompilazione o patch CUDA. Raw originali e strumenti storici preservati.

Artifact: `/home/alessandro/projects/ds4-engram-external/speed-bench/p4b-20261030-224823`. Contiene initial.json (HEAD/argv/env/hash), hook e hash oggetti da P1, execute.py, runs.json, affinity-summary.json. In ciascun raw: comando, vero hash ds4-bench in runtime-provenance.json, risultati, risorse e affinity.json. command.json del runner nei bracci B registra invece il wrapper taskset: per identità modello usare runtime-provenance.json.

Raw:
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p4b-20261030-224823-gate-A`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p4b-20261030-224823-gate-B`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p4b-20261030-224823-timing-1-A`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p4b-20261030-224823-timing-1-B`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p4b-20261030-224823-timing-2-B`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p4b-20261030-224823-timing-2-A`

Prossimo punto: P2A, profiling del loader attuale a64K/262K e256token, separato dai timing. Non avviato in questa finestra.
