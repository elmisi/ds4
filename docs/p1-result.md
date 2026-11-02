# P1 — letture ordinate nel decode: risultato neutro

Data: 2026-10-30. Finestra autorizzata in chat dal proprietario. **Nessuna promozione.**

Gate4K: 256 vettori completi da129280 float32 per braccio,132382720 byte ciascuno; tutti finiti e byte identici. Hook solo nel benchmark separato, patch e oggetti registrati nei raw. Timing con binario originale senza dump/profiling.

| Run | Prefill t/s | Decode t/s | Steady t/s |
| --- | --- | --- | --- |
| timing-1-A | 99.77 | 9.97 | 10.06 |
| timing-1-B | 99.70 | 9.98 | 10.06 |
| timing-2-B | 99.68 | 9.97 | 10.05 |
| timing-2-A | 99.60 | 9.98 | 10.06 |

Coppia AB: +0,10% throughput decode; coppia BA: −0,10%. Media e mediana A/B9,975t/s (range9,97–9,98). Circa100,25ms/token in entrambi. Risoluzione CSV0,01t/s: risultato entro il rumore, ben sotto soglia2%. Prefill circa41,06–41,12s; decode25,65–25,68s; somma circa66,71–66,80s. Wall runner70,05–70,06s, inclusi startup e campionamento.

Minimo MemAvailable su sei run: 11.98GiB. Nessuno swap del modello o incremento contatori swap, nessun abort del runner. Cache80GiB effettivi:72,88 dinamici,7862slot. Context alloc262144, popolato4096,256token teacherforced. Graphs0,topk1,stage-sync0. Esperti NVMe/Engram Lexar, checkpoint1017. Page cache OS non controllata: nessun claim disco freddo.

Il sorgente runtime ds4_cuda.cu:29138 applica il ramo ordered tramite presenza DS4_CUDA_SSD_SORT_SELECTED_READS; il wrapper passa la variabile solo a B e cancella le DS4 ereditate. Nessun contatore di ramo aggiunto: attivazione verificata da sorgente/env, non da tracing dinamico. Tutti i cubin inventariati sm_121a.

Decisione: concluso neutro nello screening4K. Nessuna estensione64K/254K, recall o output generato: non necessari per archiviare un candidato senza beneficio; nessuna dichiarazione di validazione lunga. Servizio ripristinato senza flag candidato.

Correzioni del piano: dump env assente nel binario produzione; hook separato necessario. Il runner non espande{run} nei valori env: path assoluto. Nessuna modifica agli strumenti storici.

Provenienza:
- Artifact preparazione/hook/script: `/home/alessandro/projects/ds4-engram-external/speed-bench/p1-20261030-190044`.
- `initial.json`, `provenance.json`, `object-hashes.json`, `dump-hook.patch`, `cubins.txt`, `runs.json`.
- Identita GGUF registrata per path/dimensione/mtime, checksum1017 da storico; nessun nuovo hash integrale dei341GiB.
- Quota prima/dopo ogni run PROCEED90%; checker runner conservativo60% mantenuto.

Raw:
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p1-20261030-190044-gate-A`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p1-20261030-190044-gate-B`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p1-20261030-190044-timing-1-A`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p1-20261030-190044-timing-1-B`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p1-20261030-190044-timing-2-B`
- `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/p1-20261030-190044-timing-2-A`

Prossimo punto pianificato: P4B affinità CPU, da avviare su richiesta; P2A resta successivo.
