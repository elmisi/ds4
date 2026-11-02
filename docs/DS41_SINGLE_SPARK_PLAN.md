# Piano prestazioni V4.1 su un singolo DGX Spark

Data: 2026-10-30. Stato: **P1/P4B NEUTRI; P2A COMPLETATO; P2B ARCHIVIATO; AUDIT P2C COMPLETATO**.

Obiettivo: migliorare le prestazioni di `ds4-ds41` con Q2 1017, contesto
262144, esperti su NVMe ed Engram su Lexar, preservando qualità, stabilità
e capacità di contesto. Il parallelismo tra sessioni è escluso per scelta
del proprietario. Questo piano non autorizza interruzioni del servizio,
deploy, download o ripresa automatica degli esperimenti storici.

## Come riprendere senza rifare l'indagine

Leggere questo documento e la scheda del punto scelto. Consultare gli altri
report solo quando richiamati. Non ricominciare con una ricognizione generale.

Ordine proposto: **preparazione → P1 → P4B → P2A → decisione**.
P3, P4A e P5 sono successivi; non eseguire tutta la matrice automaticamente.
P2B/P2C e P3B richiedono progettazione CUDA, non sono semplici cambi di flag.
Un agente esecutore può completare preparazione, P1, P4B, P2A e i riepiloghi;
deve fermarsi davanti a divergenze numeriche o decisioni di sincronizzazione.

Primo messaggio utile in una sessione futura:

> Riprendi docs/DS41_SINGLE_SPARK_PLAN.md dalla preparazione e dal primo punto
> non concluso. Riusa storico e harness, conserva il servizio fino a esplicita
> autorizzazione a fermarlo. Non ripetere esperimenti già conclusi.

Se viene autorizzata una finestra di test, eseguire un solo punto per volta,
registrarne l'esito e ripristinare il servizio secondo l'autorizzazione ricevuta.
Non chiedere nuovamente permessi già concessi per quella finestra.

## Baseline osservata e correzioni rispetto alla chat

- Worktree runtime: `/home/alessandro/projects/ds4-engram-external`.
- Branch: `feature/external-engram-gguf`, HEAD osservato `0ba37bd9`;
  upstream locale `ee7e8405`. Verificare nuovamente prima dei test.
- Launcher: `/home/alessandro/elmisi/amygdala/scripts/start-ds4-ds41.sh`.
- Main GGUF: `/home/alessandro/projects/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
- Engram: `/mnt/ds4-models/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
- `--cuda --ssd-streaming --ssd-streaming-cache-experts 80GB --ctx 262144`.
- KV disco: `~/.cache/ds4/deepseek-v4.1-flash-kv`, budget 65536 MiB,
  continued interval 20480. Occupazione osservata 46,88 GiB; non cancellare.
- `DS4_CUDA_DECODE_GRAPHS=0`, `DS4_CUDA_V41_TOPK_BATCH=1`,
  `DS4_CUDA_V41_PREFILL_STAGE_SYNC=0`; queue/Q8 sperimentali assenti.
- Il launcher esporta `DS4_CUDA_ENGRAM_READERS=8`, ma il codice attuale ha
  `TOKEN_READERS=4` in `ds4_engram.c`. La vecchia variabile è assente anche
  dalle stringhe del binario attivo: **non usare il messaggio readers=8 come
  prova di otto reader effettivi**. Nessuna modifica al launcher è stata fatta.
- Startup reale: KV 1,57 GiB + buffers 8,68 GiB + modello residente 9,37 GiB
  + cache dinamica 72,88 GiB + riserva prefill 7,12 GiB = 99,62 GiB pianificati.
  Il budget `80GB` include la riserva. Non sono 80 GiB di sola cache dinamica.
- V4.1 alloca un graph completo per sessione; la condivisione del workspace
  descritta genericamente nel server non si applica automaticamente qui.
  La precedente risposta in chat su 9 GiB di solo KV era errata.
- Sistema: circa 110/121 GiB usati, circa 10 GiB disponibili, swap disabilitato.
  Sono fotografie, non limiti garantiti. RSS e nvidia-smi non rappresentano
  da soli tutta la memoria unificata; includere MemAvailable e PSI.
- Lexar: ext4/noatime, UAS 20 Gbit/s. Engram usa POSIX_FADV_RANDOM.
- CPU veloci osservate: `5-9,15-19`, max 3,9 GHz; altre max 2,808 GHz.
  Governor già performance, affinità servizio `0-19`.
- Nel breve campione live di decode: NVMe circa 0,46–1,11 GB/s, Lexar
  0,56–0,91 MB/s; letture fisiche del processo circa 1,10 GB/s su 5 secondi.
  GPU 67 °C, NVMe circa 56 °C. Non sono misure isolate né prova di saturazione.


Aggiornamento 2026-11-02: branch rebasato su upstream890898cf e nuova build
validata; [report completo](REBASE_DS41_20261102.md). Il servizio resta sulla
build precedente fino all'attivazione. Per i prossimi esperimenti usare come
base il codice rebased e nuovi riferimenti numerici; i risultati P1–P2B restano
storici e non sono percentuali da sommare al nuovo runtime.

## Risultati acquisiti da non reinventare

| Tema | Evidenza e decisione | Fonte |
| --- | --- | --- |
| Split NVMe e Lexar | Il 20 ottobre split favorita di 0,16–0,20 t/s nei tre confronti; mantenerla | `~/bench-ds41-20261020-0908.md` |
| Letture ordinate prefill | −6,81% tempo medio su 30.475 token, due run per modalità, output identico; già attive | [SSD_ORDERED_READS.md](SSD_ORDERED_READS.md) |
| Top-k batch | Gate numerici e recall lungo superati; già attivo, vecchi guadagni non sommabili ai nuovi | [PREFILL_TOPK_BATCH.md](PREFILL_TOPK_BATCH.md) |
| Pool Engram storico | Profilo nativo: attesa Engram 0,02313 ms/token, carico esperti 29,92 ms/token; precedente al rebase e al 1017 | [ENGRAM_NATIVE_GB10.md](ENGRAM_NATIVE_GB10.md) |
| Cache 64 contro 72 | Vecchie prove mostrano beneficio decode circa 5,45%, non universale sul tempo totale; non equivalgono a 80 contro 84 | `~/projects/ds4-ds41/DGX_DS41_RESEARCH.md` |
| Queue | Circa +0,068% nella coppia conclusa, numericamente esatta: non promossa | `~/projects/ds4-ds41/speed-bench/dgx_ds41/RESUME.md` |
| Q8 cached-input | Nessun beneficio convincente, non promosso | stesso checkout storico, `DGX_DS41_RESEARCH.md` |
| Stage-sync | Due coppie pulite +1,38% e +11,17%, media +6,38%; coppia con swap esclusa | [PREFILL_STAGE_SYNC.md](PREFILL_STAGE_SYNC.md) |
| KV persistente | Fix e roundtrip lungo già validati; non ripetere audit da zero | [ENGRAM_RESUME.md](ENGRAM_RESUME.md) |

Il medium sweep CUDA per append intermedi è già attivo. Il vecchio NEXT
«4608 token» è un caso da caratterizzare, non una funzione mancante.
Prefetch dell'intero layer è oggi opt-in (`DS4_CUDA_ENABLE_SSD_PREFETCH`);
non confonderlo con l'ordinamento delle sole letture selezionate.

## Preparazione comune

### Autorizzazione e isolamento

1. Confermare il punto richiesto e leggere eventuali nuove istruzioni locali.
2. Verificare quota prima/dopo ogni passo limitato:
   `python3 /home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/check_quota.py --threshold 50`.
   STOP se sotto 50%, sconosciuta, invalida, futura o vecchia oltre 600 secondi.
   La soglia 60% in vecchi documenti è superata; un runner più conservativo
   può fermarsi prima. Non aggirare il controllo.
3. Leggere `git status`, HEAD, unit, PID, argv e sole variabili DS4/CUDA del
   processo. Verificare che binario su disco e `/proc/PID/exe` coincidano.
   Salvare hash binario, sorgenti/patch, fixture e identità dei GGUF. Riutilizzare
   checksum GGUF già verificati con dimensione/data coerenti; evitare una lettura
   completa dei 341 GiB mentre il servizio sta usando i dischi.
4. Il servizio attivo non va fermato né caricato con richieste sperimentali.
   Per GPU/bench serve una finestra autorizzata; lifecycle solo con `dgx-ctl`.
   Il runner rifiuta processi modello concorrenti: non disabilitare il controllo.
5. Modifiche CUDA in worktree scratch da HEAD runtime, con ramo dedicato.
   Non modificare gli esperimenti sospesi in `ds4-ds41`, non committare su main,
   non rebasare showcase, non chiudere/mergiare PR #9/#10.
6. Build solo se necessaria: `nice -n 10 systemd-run --user --scope -p MemoryMax=9G -- make -j2 cuda-spark`
   nel worktree scratch. Controllare tutti i cubin con `cuobjdump --list-elf`:
   attesi sm_121a. Non sostituire binari in produzione per fare un confronto.

### Harness e trappole già verificate

- Runner esistente: `~/projects/ds4-ds41/speed-bench/dgx_ds41/run.py`.
  Crea directory raw nuove, rifiuta sovrascrittura, controlla processi concorrenti,
  imposta graphs=0 e FORCE_SNAPSHOT=1 e rimuove le altre variabili DS4 ereditate.
  **Passare top-k e stage-sync esplicitamente in ogni braccio.**
- Il runner registra Git del proprio checkout storico. Aggiungere in ogni run
  `runtime-provenance.json` con commit/patch del runtime, hash binario effettivo,
  modello, Engram, fixture, contesti popolato/allocato, opzione variata.
- Il runner richiama il quota checker senza `--threshold`: ispezionare il default.
  Se va adeguato a 50, farlo in una copia isolata preservando percorsi raw/checker
  e registrando la patch. Non modificare silenziosamente gli strumenti storici.
- `prefill_stage_suite.py` importa `engram_pool_suite`, assente nel checkout
  osservato. Inoltre fissa cache72GB. **Non lanciarla così com'è.** Preferire
  invocazioni singole del runner o una copia corretta e parametrizzata.
- `compare_decode_logits.py` stampa diagnostiche: exit 0 NON significa uguaglianza.
  Esigere dimensioni/vettori attesi, valori finiti e `cmp` esatto per P1–P4.
- `prefill_telemetry.py --process ds4-bench` permette campionamento del processo
  reale. Con wrapper/sanitizer il PID del runner può appartenere al wrapper:
  non attribuire al modello RSS/swap del processo padre.
- Il replay storico è fissato a 4096 token di append e usa vecchi artefatti.
  Non usarlo come evidenza per 4608 né con snapshot di diverso modello/checkpoint.

### Comando base per una singola prova

Da eseguire SOLO dopo autorizzazione della finestra, a GPU libera e con un
`ds4-bench` di provenienza verificata. Il comando richiede un benchmark con hook di dump verificato: il binario
produzione del 25 ottobre NON implementa `DS4_BENCH_DECODE_LOGITS_FILE`.
Il runner NON espande `{run}` nelle variabili ambiente: usare path assoluto. Usa un nome nuovo ogni volta.

```bash
python3 /home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/run.py \
  --timeout 1800 --min-available-gib 8 \
  --env DS4_CUDA_V41_TOPK_BATCH=1 \
  --env DS4_CUDA_V41_PREFILL_STAGE_SYNC=0 \
  --env DS4_BENCH_DECODE_LOGITS_FILE=/PERCORSO/ASSOLUTO/DEL/RUN/decode.f32 \
  p1-baseline-4k-UNIQUE -- \
  /home/alessandro/projects/ds4-engram-external/ds4-bench \
  --model /home/alessandro/projects/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --engram-model /mnt/ds4-models/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 80GB \
  --prompt-file /home/alessandro/projects/ds4-engram-external/speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 4096 --ctx-alloc 262144 \
  --gen-tokens 256 --teacher-forced-decode --csv '{run}/bench.csv'
```

Sostituire UNIQUE con un identificatore mai usato. Per il candidato P1
aggiungere `--env DS4_CUDA_SSD_SORT_SELECTED_READS=1` prima del nome del run.
Per il baseline lasciare la variabile ASSENTE; `=0` la abilita comunque.
Per timing rimuovere il dump logits e qualsiasi profiling. Nessuna directory
KV di produzione è usata da questo comando. Controllare lunghezza reale prompt
e CSV: 4096 è contesto popolato del benchmark, 262144 è capacità allocata.

### Misure e criteri comuni

- Gate corto prima dei tempi: stessa sequenza teacher-forced, almeno 256 vettori
  completi F32; con vocabolario129280 sono 132382720 byte. Confermare il numero
  emesso dal percorso di dump, non ignorare file vuoti/incompleti o NaN.
- Per P1–P4: differenza numerica ⇒ STOP candidato e diagnosi; non allargare
  tolleranze per ottenere PASS. Il baseline va prodotto col 1017 corrente:
  gli hash dei vecchi Q2 non sono oracoli direttamente intercambiabili.
- Dopo il gate, due coppie temporali AB e BA come screening. Se promettente,
  aggiungere una terza coppia; non ripetere indefinitamente un risultato neutro.
- Soglie di progetto, NON risultati misurati: candidato promettente se almeno
  +3% sulla metrica primaria in media e stesso segno nelle prime due coppie.
  Risultato <2% ⇒ neutro; 2–3% o segni discordanti ⇒ inconclusivo, una sola
  coppia aggiuntiva prima di archiviarlo. Per memory-only P3B vale il suo gate.
- Riportare tempi singoli, mediana, range e tempo dell'intera richiesta. Non
  sommare percentuali. Non fare claim sul disco freddo senza misurarlo; non
  usare drop_caches. Run profilati e dump non entrano nella media dei tempi.
- MemAvailable floor iniziale8GiB; nessuno swap del modello, OOM, errori CUDA,
  reset USB o peggioramento persistente PSI memoria. Scartare l'intera coppia
  contaminata e registrare perché. Non abbassare il floor per far passare P3/P5.
- Per promozione: confronto numerico lungo a circa254K reali e append entro262144,
  recall16/16 con fixture esistente, risposta coding e uso tool, stabilità
  memoria. Riutilizzare gate già conclusi SOLO per stesso binario/configurazione
  e medesimo percorso. La prova corta serve a evitare lavoro lungo inutile.
- Prima del deploy preparare diff configurazione, evidenza, limiti e rollback;
  applicare al servizio solo quando autorizzato. Ripristinare la baseline
  salvata, senza reset della KV persistente o modifiche globali a swap/cache OS.

## P1 Letture ordinate anche nel decode

**Difficoltà: esecuzione semplice. Nessuna modifica CUDA.**

Ipotesi: ordinare i miss degli esperti per offset file riduce il costo I/O
del decode. Il prefill è già ordinato e non è l'oggetto del confronto.

- Sorgenti: `ds4_cuda.cu`, `routed_moe_launch` passa `n_tokens > 1` a
  `cuda_stream_selected_cache_begin_load_tensor`; `cuda_stream_selected_cache_begin_load`
  consente override con `DS4_CUDA_SSD_SORT_SELECTED_READS`.
- A: variabile assente. B: `DS4_CUDA_SSD_SORT_SELECTED_READS=1`.
  `DS4_CUDA_DISABLE_SSD_SORT_SELECTED_READS` deve essere assente in entrambi.
- Tenere cache80, ctx262144, topk1, stage-sync0, graphs0, stesso GGUF e fixture.
- Eseguire gate4K con comando base, poi screening4K. Se passa, validare64K e
  circa254K. Aggiungere output generato rappresentativo prima della promozione.
- Metrica primaria: ms/token decode, più steady decode se disponibile;
  secondarie: byte letti/token, miss e costo loader in una prova profilata separata.
- Controllare che B entri davvero nel ramo ordered con `n_tokens=1`: sorgente
  più log/contatore isolato se necessario, non inferirlo dalla sola presenza env.
- Possibile beneficio piccolo quando manca un solo esperto. Neutro ⇒ archiviare.
- Rollback: togliere la variabile, non impostarla a0.
- Deliverable: `p1-result.md`, comandi/provenienza, logits confronto, coppie e decisione.

## P2 Caricamento e gestione degli esperti

**P2A: profiling circoscritto; P2B/P2C: progettazione CUDA.**

### P2A Attribuire il costo sul runtime corrente

Il vecchio29,92ms/token includeva I/O, trasferimenti e gestione cache; non è
puro tempo NVMe e non va trasferito automaticamente al checkpoint1017.

1. Una prova diagnostica con baseline attuale, 64K popolati/262K allocati e
   256 token teacher-forced; aggiungere `DS4_CUDA_V41_DECODE_PROFILE=1` e
   `DS4_CUDA_SSD_CACHE_PROFILE=1`. Confermare flag e campi nei sorgenti prima.
2. Usare `python3 speed-bench/engram_profile_summary.py RUN/stderr.log --frontier 65536`.
   Separare prefetch_wait, reuse_sync, prepare, load, finish, miss e byte/token.
   Gli intervalli possono essere annidati: non sommarli indiscriminatamente.
3. Confrontare diagnostica con tempi puliti P1; nessun claim di throughput
   basato sul profiling. Esigere che il profiling non cambi il gate numerico.
4. Scegliere un solo sottoprogetto. Se CPU/victim selection è trascurabile,
   non implementare P2B. Se attesa caricamento non domina, rivedere la priorità.

### P2B Ridurre il costo di scelta delle vittime

- Punto: scansione lineare `g_stream_expert_slots` per ogni miss nel loader.
- Prima misurare separatamente durata e numero di scansioni: `load_ms` include
  altro e non basta per attribuire il costo alla ricerca LRU.
- Progettare una struttura che preservi protezione slot in uso/prefetch,
  aggiornamento hit, invalidazione e gestione degli errori. Documentare il
  comportamento in caso di timestamp uguali e assenza di vittime disponibili.
- Patch opt-in isolata; test host per invarianti, failure path e riuso;
  CUDA sanitizer sul percorso toccato, gate numerici corti/lunghi, timing pulito.
- STOP e revisione specialistica se non è chiaro quando uno slot sia riusabile.
  Non cambiare simultaneamente politica di rimpiazzo e rappresentazione LRU.

### P2C Sovrapporre caricamento e calcolo indipendente

- Punto: `ds41_moe_partial` e loader CUDA. Il ramo shared side-stream
  osservato è condizionato da `shared_owner`/TP; single-host non va assunto
  equivalente. Identificare esattamente dove router ID diventano disponibili.
- Disegnare dipendenze: router → ID → lettura → upload → routed expert;
  shared expert indipendente → join prima della somma. Proteggere slot, buffer
  pinned e output fino a fine stream; mantenere tutte le dipendenze necessarie.
- Non eliminare barriere alla cieca e non riproporre la vecchia decode queue.
- Primo deliverable: schema stream/eventi e audit lifetime, PRIMA della patch.
  Poi opt-in, gestione errori/cancellazione, gate numerici e sanitizer pertinenti.
- Promuovere solo se il tempo totale migliora; spostare attese da un campo
  del profiler a un altro non è un guadagno.

## P3 Memoria e cache esperti

**P3A: screening configurazione. P3B: modifica strutturale.**

### P3A Budget 80 contro 82 o 84 GiB

- Confermare prima che non esista un nuovo report sullo stesso confronto.
- A80, B82 come primo incremento; B84 solo se82 lascia almeno8GiB durante
  il workload. Non partire da88/96. Contesto262144 invariato.
- Registrare target, riserva e cache dinamica effettiva, numero slot realmente
  allocati: l'allocatore può ridurre il budget per conservare memoria.
- Gate corto esatto, coppie decode e un prefill lungo se il candidato passa.
  Più cache GPU può togliere page cache al disco: misurare il tempo completo.
- Se slot non aumentano o il floor viene violato, STOP: nessun vantaggio
  dimostrato e nessuna riduzione automatica dei margini.
- Il vecchio64/72 non prova il beneficio80/84. Rollback: budget80.

### P3B Separare workspace prefill e stato necessario al decode

- Sorgenti: `DS41_SCRATCH`, `DS41_PREFILL_STORAGE`, `DS41_CARRY_ROWS`,
  `ds41_graph_bytes`, `ds41_graph_alloc/free`, `ds41_graph_prefill_sweep`.
- Primo deliverable: tabella buffer → dimensione a262144 → allocazione reale
  o view → usi decode/prefill/short-prefill → proprietario → ultimo uso GPU.
- Conservare cache KV, storia Engram, logits e contesto262144. La massa8,68GiB
  comprende componenti diversi: non promettere di liberarla tutta.
- Separare allocazione lazy (sposta il picco) da rilascio dopo prefill (può
  liberare memoria in decode ma costa al prossimo append). Quantificare
  riallocazioni, frammentazione, admission e picchi prima di cambiare codice.
- Tenere cache80 nel primo confronto. Solo dopo isolamento del beneficio
  memoria, considerare trasferire parte del margine alla cache esperti.
- Gate: nessun accesso a buffer liberati, errori allocazione recuperabili,
  append1/128/256/2048/4608/8192 e lungo, snapshot save/restore e logits esatti.
- Criterio progetto: recupero almeno1GiB misurato in decode e nessuna regressione
  ripetibile oltre3% sul tempo dell'intera richiesta; altrimenti ridiscutere
  complessità/beneficio. Queste soglie sono obiettivi, non evidenza esistente.

## P4 Prefill incrementale e affinità CPU

Due prove indipendenti; non abilitare entrambe nel primo confronto.

### P4A Stage sync con il profilo attuale

**Difficoltà: media, serve harness incrementale esplicito.**

- A `DS4_CUDA_V41_PREFILL_STAGE_SYNC=0`; B `=1`. Topk1 in entrambi,
  cache80, graphs0, nessuna flag di profiling nella misura.
- Primo screening: fresh prefill circa1K e append1024/2048 dopo prefisso64K.
  Se utile, estendere a4096/4608/8192 e prefisso vicino254K, verificando sempre
  `prefisso + append + generazione < 262144` (8192 non entra dopo254K).
- Un fresh prefill non misura append: usare una sessione mantenuta o snapshot
  dello STESSO1017, identico prefisso per A/B. Registrare posizione iniziale,
  token riusati, realmente ricalcolati e posizione finale.
- Il runner/replay storico hardcoded4096 va adattato in copia isolata, con
  parametri espliciti e provenienza. Non cambiare i vecchi contratti dei raw.
- Non ritestare il top-k; il medium sweep è già attivo. Non attivare insieme
  prefetch intero layer. Controllare il dispatch per4608 e le code piccole.
- Metrica primaria: tempo append e latenza primo token; secondaria: decode e
  tempo totale. Gate numerici sul confine append e sulla continuazione.
- Rollback stage-sync0. Se variabilità resta grande, archiviare inconclusivo.

### P4B Affinità ai core veloci

**Difficoltà: esecuzione semplice; nessun cambiamento globale CPU.**

- Riverificare capacity/frequenze sysfs: non assumere gli ID su un'altra macchina.
- A affinità predefinita; B `taskset -c 5-9,15-19` applicato al solo benchmark.
  Nessun taskset sul PID live; governor performance già attivo.
- Nel comando base inserire `/usr/bin/taskset -c 5-9,15-19` davanti al binario
  dopo `--`, verificando il path. Salvare hash del vero ds4-bench: il runner
  potrebbe registrare quello di taskset. Controllare Cpus_allowed_list dei thread.
- Gate4K e coppie decode; poi prefill breve/lungo per escludere regressioni
  da restrizione dei reader/worker. Non combinare con P1 nella prima misura.
- Modello invariato: logits identici richiesti. Neutro ⇒ mantenere scheduling OS.
- Eventuale deploy futuro tramite wrapper/unit dedicato e diff approvato;
  rollback ripristinando affinità originale, non un elenco CPU assunto.

## P5 DSpark V4.1

**Difficoltà: media per esecuzione, alta per valutazione qualità.**

- File supporto V4.1 circa8,04GiB, distinto da Flash0731. Verificare conversione
  e compatibilità con1017 nei sorgenti/docs attuali prima di procurarlo.
  Inventariare file già disponibili; nessun download automatico sulla Lexar.
- Fonte: [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md). Supportato single-host
  CUDA SSD; accettazione bassa può peggiorare la velocità.
- Piano memoria iniziale: drafter residente più stato; confrontare prima con
  cache72 per fare spazio, senza consumare il margine attuale. L'admission e il
  floor8GiB restano obbligatori. Non assumere che8GiB di taglio bastino.
- Tre bracci per separare i costi: A target-only cache80; B target-only cache72;
  C DSpark cache72. Confronto C/B isola speculation, C/A decide l'utilità reale.
- Config C: `--dspark --mtp-model FILE_VERIFICATO --mtp-exact-sampling`,
  più tutti i parametri baseline. Non usare il file DSpark di V4 Flash.
- Temp0 per casi greedy; temperatura e parametri reali per coding, prosa,
  ragionamento e tool. Almeno due prompt per famiglia nello screening.
  Registrare draft proposti/accettati, tempo verifica, t/s e tempo completo.
- Ripristino snapshot: il drafter deve riscaldarsi per128 token target-only;
  separare questa fase nel report senza nasconderne il costo all'utente.
- Exact sampling conserva la distribuzione prevista dall'algoritmo, ma il
  verifier batched può differire numericamente dal percorso scalare. Non usare
  il gate cmp di P1–P4 come unica decisione e non dichiarare qualità invariata
  dalla sola accettazione. Servono suite qualità, recall lungo e output reale.
- `--dspark-strict` disabilita l'accettazione speculativa: è un controllo,
  non una modalità veloce con identità garantita. Non disabilitare thinking
  o accorciare risposte per simulare un guadagno di inferenza.
- STOP se supporto incompatibile, memoria insufficiente, regressioni qualità,
  o C non batte A sui carichi scelti. Nessuna promessa di speedup prima dei dati.
- Rollback: rimuovere DSpark e ripristinare cache80 della baseline salvata.

## Registro di avanzamento

Aggiornare questa tabella dopo ogni passo. Stati ammessi: non avviato,
preparato, in corso, concluso favorevole, concluso neutro, respinto, inconclusivo.

| Punto | Stato aggiornato al 1 novembre | Prossima azione | Artifact |
| --- | --- | --- | --- |
| Preparazione | preparato | Provenienza e hook dump verificati | [P1](p1-result.md) |
| P1 | concluso neutro | Archiviato: gate esatto, AB/BA ~0% | [Risultato](p1-result.md) |
| P2A | concluso favorevole (diagnostica) | Loader25,10ms/token; scansione esclusa da P2B | [Risultato](p2a-result.md) |
| P2B | concluso neutro | Scansione0,048ms/token: non ottimizzare LRU | [Risultato](p2b-result.md) |
| P2C | preparato (audit concluso) | Prototipo opt-in scalar, poi gate4K | [Audit](p2c-audit.md) |
| P3A | non avviato | Budget e capacità effettiva80/82 | — |
| P3B | non avviato | Tabella ownership e lifetime buffer | — |
| P4A | non avviato | Preparare harness append isolato | — |
| P4B | concluso neutro | Archiviato: gate esatto, decode −0,10% | [Risultato](p4b-result.md) |
| P5 | non avviato | Inventario supporto e budget | — |

Ogni risultato deve riportare: data, punto, commit/patch runtime, hash binario,
fixture e modello, comando/env, autorizzazione della finestra, posizione reale
dei raw, controlli qualità, tempi singoli, minimo RAM/swap, decisione motivata
e prossimo passo esatto. Distinguere preparazione completata da test eseguiti.

## Fuori dal piano

Multi-session batching escluso. TP su due Spark resta un progetto distinto.
Non spostare Engram su NVMe, aumentare reader, cambiare quantizzazione,
ridurre contesto o modificare swap/readahead/governor come tentativi automatici.
Non iniziare download grandi sulla Lexar senza procedura termica autorizzata
(`~/elmisi/amygdala/sudo-scripts/lexar-cool-download.sh`). Nessuna richiesta
di privilegi o azione di servizio è necessaria per leggere e aggiornare il piano.
