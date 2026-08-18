# Letteratura sull'invarianza di topologia negli operatori spettrali
### Resoconto ragionato, in luce del fallimento misurato del pozzo di potenziale

Dieci paper scaricati e letti (113.580 parole di testo estratto). PDF e testo grezzo sono in
questa cartella; ogni titolo è stato verificato dal testo estratto e non dall'URL — un
download (`shapedna`) era un paper completamente diverso ed è stato scartato.

---

## 0. Perché questo resoconto esiste

Il 17-18 agosto abbiamo misurato che il **pozzo di potenziale infinito** (Liu, Jacobson &
Crane) **non funziona** sul nostro problema:

| braccio | crop | noisy | resample | all |
|---|---|---|---|---|
| niente pozzo | **0.7072** | 0.7561 | **0.7719** | **0.7347** |
| pozzo mal posizionato | 0.7034 | 0.7573 | 0.7533 | 0.7315 |
| pozzo α=0.55 | 0.7012 | **0.7653** | 0.7521 | 0.7215 |

La predizione registrata ("crop sale, resample invariato") è smentita su entrambi i punti.
La domanda di questo resoconto: **la letteratura lo spiega, e quali strade restano?**

Risposta breve: **sì, lo spiega — in due punti distinti — e le alternative valide sono di una
famiglia diversa da quella che abbiamo provato.**

---

## 1. Il fallimento era prevedibile dalla letteratura. Due passaggi.

### 1.1 Choukroun et al. avvertono che il potenziale AMPLIFICA il rumore di discretizzazione

*Hamiltonian Operator for Spectral Shape Analysis* (Choukroun, Shtern, Bronstein, Kimmel),
sezione "Robustness to noise", scrive testualmente:

> «the eigenfunctions of the regular Laplacian may present **smaller** distortion to noise than
> the Hamiltonian since the perturbation is amplified by **area and potential distortions**»

Con l'analisi perturbativa esplicita (loro eq. 31):

    ψ̃ᵢ = ψᵢ(1 − ψᵢᵀ δ_A ψᵢ / 2) + Σ_{k≠i} [ψᵢᵀ(δ_W − Eᵢ δ_A)ψ_k / (Eᵢ − E_k)] ψ_k

Il termine `Eᵢ δ_A` è la chiave: l'errore di area entra **moltiplicato per l'autovalore**, e gli
autovalori dell'Hamiltoniano sono molto più grandi di quelli del Laplaciano (per il loro
Teorema 1, `Eᵢ ≥ λᵢ + min V`). Un pozzo profondo alza tutti gli `Eᵢ` e quindi **amplifica ogni
errore di area**.

**Questo è esattamente il nostro `resample` −0.0198.** I gruppi `down8k`/`up60k` cambiano
proprio l'area dei triangoli. Non è un incidente della nostra implementazione: è il
comportamento che la teoria del potenziale prevede.

### 1.2 La robustezza dimostrata è su BUCHI, non su bordi spostati

Sempre Choukroun et al.: la loro dimostrazione di robustezza topologica usa «30% of the surface
area was removed due to topological noise **in the form of small holes**». Liu-Jacobson-Crane
allo stesso modo lavorano su «patches with different boundary shapes or discretizations»,
cioè **ritagli da una superficie più grande con interno condiviso ampio**.

Il nostro `crop` non è nessuna delle due cose: sposta il **bordo esterno** di una superficie che
è già essa stessa una patch. Il regime è diverso, e lo avevamo già quantificato: imporre la
condizione del paper (nessun bordo dentro la regione di interesse) lascia il **2–7%** dell'area.

---

## 2. Il malinteso di partenza: cosa promette davvero DiffusionNet

*DiffusionNet: Discretization Agnostic Learning on Surfaces* (Sharp, Attaiki, Crane,
Ovsjanikov) dice una cosa che ridimensiona la nostra premessa iniziale:

> «DiffusionNet is **not a spectral learning method** — spectral coefficients are never used to
> represent filters or latent data, and thus **no issues arise due to differing eigenbases on
> different shapes**. Spectral acceleration is merely one possible numerical scheme to compute
> diffusion.»

Cioè: la base spettrale precalcolata è **solo un acceleratore numerico** (esiste anche
l'alternativa a passo implicito, sez. 3.3.1), non una rappresentazione. La preoccupazione
"basi diverse su mesh diverse" è esplicitamente esclusa dagli autori.

E la loro promessa è **discretizzazione**, testata su remeshing, semplificazione quadrica e
mesh-contro-point-cloud — cioè tessellazioni diverse **della stessa forma**. Mai su forme
parziali.

**I nostri numeri lo confermano esattamente:**
- `resample` (retessellazione) = **0.7719**, il nostro gruppo migliore → dentro il dominio di progetto
- `crop` (dominio diverso) = **0.7072**, il peggiore → fuori dal dominio di progetto

Conclusione: **`crop` non è un problema di discretizzazione, è un problema di parzialità.**
Abbiamo applicato uno strumento da "patch di una superficie grande" a un problema di
"superficie parziale", e la chirurgia sull'operatore era la famiglia sbagliata di rimedio.

---

## 3. Il risultato teorico più utile: il crop non distrugge lo spettro, lo riparametrizza

*Partial Functional Correspondence* (Rodolà, Cosmo, Bronstein, Torsello, Cremers) è il paper
più direttamente applicabile al nostro `crop`. Due risultati:

**(a) Struttura diagonale-inclinata.** La matrice della functional map fra forma intera e forma
parziale non è arbitraria: ha una diagonale **inclinata**, con pendenza `r/k` determinata dal
**rapporto di aree**, coerentemente con la legge di Weyl (`λ_k` cresce linearmente con tasso
inversamente proporzionale all'area).

**(b) Teorema 1.** Il cambiamento al primo ordine degli autovalori della forma parziale dipende
**solo dalla variazione di energia di Dirichlet delle autofunzioni lungo il bordo tagliato**.

La lettura pratica: **tagliare una faccia non scombina lo spettro, lo riscala in modo
prevedibile.** Non serve rendere l'operatore invariante distruggendo superficie — basta
**correggere la trasformazione nota**.

E abbiamo già l'evidenza in casa: nel supplementary avevamo misurato che
`λ₁·A ∈ {11.1, 9.8, 8.2}` concorda fra BFM/FLAME/FaceScape mentre `λ₁` grezzo no. Era la stessa
legge di Weyl, osservata senza riconoscerla.

---

## 4. Schede dei paper

### 4.1 Liu, Jacobson & Crane — *A Dirac Operator for Extrinsic Shape Analysis* (CGF 36(5), 2017)
Famiglia di operatori con trade-off continuo intrinseco↔estrinseco. **Sez. 5.1** è il pozzo che
abbiamo usato: `U(p) = c / (1 + e^{−γ(d(p,q) − β)})`, con `c = 10¹⁰`, `γ = 100`, `d` geodetica
via heat method dal centro, `β` normalizzato **sull'intera collezione** perché nessun bordo
cada nella regione di interesse. Discretizzazione `U_ii = A_i·U(p_i)`.
**Regime**: patch ritagliate da superfici grandi, per confrontare interni condivisi.
**Codice**: gptoolbox contiene gli operatori di Dirac (`dirac_operator.m`, `dirac_eigs.m`) ma
**non il pozzo** — verificato leggendo il sorgente. Nessuna implementazione di riferimento.

### 4.2 Choukroun et al. — *Hamiltonian Operator for Spectral Shape Analysis* (2017)
`H = −Δ + V`. Stessa forma del pozzo, ma inquadrata come **generatore di basi**, non come
trattamento del bordo. Contributi utili a noi:
- **Teorema 1**: `max(V) + λᵢ ≥ Eᵢ ≥ min(V) + λᵢ`.
- **Vincolo principiato sulla profondità**: `μ < λᵢ / (max V − ⟨ψᵢ, V ψᵢ⟩)` per far sì che le
  prime `i` autofunzioni restino confinate. **Noi avevamo scelto `c` per sweep empirico:
  esisteva un criterio, e non lo avevamo usato.**
- **Teorema 2**: l'Hamiltoniano è ottimale per funzioni a gradiente limitato **e** valori bassi
  nelle zone ad alto potenziale.
- **Feynman-Kac**: la diffusione è modulata esponenzialmente dal potenziale → anisotropa verso
  le regioni a basso potenziale.
- **L'avvertimento sul rumore** (sez. 1.1 sopra).
Uso loro del potenziale: **arricchire** la firma (più descrittiva). Uso nostro: **sopprimere**
il bordo. Sono due cose diverse, e la loro è quella supportata dai risultati del paper.

### 4.3 Melzi et al. — *Localized Manifold Harmonics* (CGF, 2017)
Costruzione `W + μ_R·A·diag(v)` — **la stessa forma algebrica del pozzo** — ma con una
differenza decisiva: la nuova base è vincolata a stare nel **complemento ortogonale delle prime
`k₀` autofunzioni del Laplaciano** (`Ψ = (I − P_{k₀})Y`). Quindi le armoniche localizzate
**si aggiungono** alla base globale invece di sostituirla.
**Perché conta per noi**: il nostro fallimento è "il pozzo toglie informazione che la rete
usava". LMH è la versione che *non toglie niente*.
**Limite dichiarato**: serve una regione data a priori; e non è ovvio quante armoniche usare.

### 4.4 Rodolà et al. — *Partial Functional Correspondence* (CGF, 2016)
Vedi §3. Aggiunge: le parti corrispondenti sono **variabili di ottimizzazione**, regolarizzate
alla Mumford-Shah (parti grandi e regolari). Il rango `r` della matrice `C` si stima
**confrontando gli spettri delle due forme**, senza corrispondenza nota.

### 4.5 Attaiki, Pai & Ovsjanikov — *DPFM: Deep Partial Functional Maps* (3DV 2021)
Primo metodo **appreso** per corrispondenza parziale non rigida; impara i descrittori dai dati
invece di usarne di fatti a mano; gestisce anche **partial-to-partial** con regione comune
ignota. Codice pubblico. È la versione moderna e addestrabile di §4.4.

### 4.6 Wang, Ben-Chen, Polterovich & Solomon — *Steklov Spectral Geometry* (TOG)
Sostituisce il Laplace-Beltrami con l'operatore **Dirichlet-to-Neumann**: «making a shift from
intrinsic to extrinsic geometry as simple as substituting the LBO with the DtN operator».
Lo spettro di Steklov è definito **sul bordo** e descrive il volume racchiuso.
**Verdetto per noi**: sfavorevole. Il bordo è precisamente ciò che nel nostro `crop` cambia, e
le nostre facce sono superfici aperte, non bordi di volumi.

### 4.7 Ovsjanikov et al. — *Functional Maps* (SIGGRAPH 2012)
Il fondamento: rappresentare una corrispondenza come **operatore lineare fra spazi di
funzioni** invece che come mappa punto-punto, diagonalizzato nella base LBO. È il quadro in cui
§4.4 e §4.5 vivono.

### 4.8 Wiersma et al. — *DeltaConv* (TOG 2022)
Operatori anisotropi da calcolo vettoriale su **point cloud**, con stream scalare e vettoriale
accoppiati. **Nessuna connettività, nessun operatore di mesh da precalcolare.**
**Perché conta**: è l'alternativa che elimina il problema alla radice invece di correggerlo. Se
la dipendenza dalla topologia è il nostro nemico, un backbone che non vede la topologia è la
risposta più diretta.

### 4.9 Crane, Weischedel & Wardetzky — *Geodesics in Heat* (2013)
Il metodo che usiamo per `d(p,q)` nel pozzo. Risolve l'equazione del calore per un tempo breve,
normalizza il gradiente, risolve una Poisson. Rilevante come **fonte di errore**: la geodetica
dipende dalla discretizzazione, quindi anche il posizionamento del pozzo lo era.

### 4.10 Sharp et al. — *DiffusionNet* (TOG 2022)
Vedi §2. Da aggiungere: i soli ingredienti sono diffusione appresa, un MLP puntuale e feature
di gradiente spaziale; niente convoluzioni spaziali né gerarchie di pooling. Il pooling globale
del nostro encoder è **una nostra aggiunta**, non parte del design originale — ed è dove
avevamo identificato il buco del supporto non mascherato.

---

## 5. Le strade, ordinate per quanto la letteratura le sostiene

### A. Normalizzare gli autovalori per l'area (Weyl) — **la più forte, e quasi gratis**
Per Weyl `λ_k ≈ 4πk/A`: il crop cambia gli autovalori in modo **noto e calcolabile**. Dare alla
rete `λ·A` invece di `λ` grezzo cancella l'effetto al primo ordine, **senza togliere superficie**.
Sostegno: Rodolà et al. (la pendenza della diagonale È il rapporto di aree) + la nostra stessa
misura `λ₁A ∈ {11.1, 9.8, 8.2}` contro `λ₁` grezzo discorde.
Costo: cambio di poche righe nel preprocessing. Nessun nuovo dato, nessun nuovo operatore.

### B. LMH: aggiungere armoniche localizzate invece di sostituire la base
Stessa algebra del pozzo, ma ortogonale alla base globale. Attacca direttamente il nostro modo
di fallire (perdita di informazione) mantenendo il guadagno di località.
Costo: medio. Serve definire la regione; il centro geodetico che già calcoliamo basta.

### C. Backbone senza operatore precalcolato (DeltaConv / point-based)
Elimina la dipendenza invece di correggerla. È anche la risposta più diretta alla critica
"DiffusionNet senza spiegazioni e nient'altro". **Le nostre ablation in coda (`spec_mlp`,
`intrinsic_dn`) sono la versione economica di questa domanda.**

### D. Livello dati: allenare con la corruzione (esperimento Poisson, già in coda)
Se la rete compensa già i cambi di bordo usando la geometria periferica — la nostra spiegazione
del fallimento — allora darle **più varietà di bordi** è il lever giusto. Sharp et al. dicono
esattamente questo per le invarianze non-discretizzazione: «Other invariants can be encouraged
via **data augmentation**».

### E. Partial functional maps / DPFM
Teoricamente il tool corretto per la parzialità. **Ma** reintroduce una stima di corrispondenza,
cioè esattamente il passo di registrazione che il nostro paper critica. Da usare come
**baseline da battere**, non come nostro metodo.

### F. Steklov — scartata, motivata
Il bordo è la variabile che cambia; le nostre superfici sono aperte. Regime sbagliato.

---

## 6. Cosa NON rifare

- **Altra chirurgia sull'operatore che restringe il dominio.** Due misure indipendenti (il
  nostro A/B e l'analisi perturbativa di Choukroun) dicono che si paga in sensibilità alla
  discretizzazione più di quanto si guadagni in invarianza al bordo.
- **Fidarsi di un proxy spettrale.** Il nostro sweep senza training indicava α=0.55 con
  entrambe le metriche in miglioramento; il modello allenato è andato nella direzione opposta
  proprio sul gruppo bersaglio. Spettri più simili ≠ ordinamenti di identità più robusti.
- **Usare parametri "come nel paper" fuori dal loro regime.** La calibrazione prescritta da
  Liu et al. è la scelta peggiore sui nostri dati, e il motivo è strutturale, non numerico.

---

## 7. Nota sui file di questa cartella

`NeurIPS_2026_3DFace_Metrics.pdf` è **il nostro paper** (la sottomissione rifiutata, *When
Alignment Hurts*), non un riferimento scaricato: era già qui. Tutti gli altri `.pdf` sono stati
scaricati e verificati contro il titolo estratto dal testo, non contro l'URL — un download si
era rivelato un paper del tutto diverso ed è stato scartato. Ogni `.txt` è l'estrazione
`pdftotext` corrispondente, tenuta per poter citare alla lettera.

---

## 8. Ricerca di letteratura: i functional maps nell'AI (18 agosto 2026)

Ricerca fatta su richiesta, su corpus arXiv/alphaXiv. Tredici paper rilevanti, quattro letti nel
merito. La linea è: FMNet (Litany et al. 2017, primo fmap layer differenziabile) → successori
non supervisionati → DPFM per la parzialità (2021) → consistenza spaziale/spettrale (2023) →
frequency-awareness e basi apprese (2024-2026).

**Attenzione al trasferimento.** Tutta questa letteratura calcola *corrispondenze*. Noi
calcoliamo un *embedding globale* per una metrica di ranking e non abbiamo alcun fmap layer.
Quindi non tutto si trasferisce, e sotto è segnato che cosa sì e che cosa no.

### 8.1 La convenzione di preprocessing del campo è la normalizzazione d'area — e noi non la facevamo

Bracha, Dagès & Kimmel, *On Unsupervised Partial Shape Correspondence* (ACCV 2024,
arXiv:2310.14692), sezione sperimentale, alla lettera:

> «as in other methods [4, 14], we normalize the square root of the area of the shapes to one»

Cioè $\sqrt{A} = 1$: esattamente ciò che fa `v2_work/potential/areanorm_operators.py`, e lo
citano come pratica corrente altrui, non come contributo proprio. La nostra pipeline
normalizzava invece per `maxabs`, il raggio del vertice più lontano. Non è una scelta
alternativa documentata in letteratura: è fuori convenzione. **Questo è l'item più azionabile
della ricerca, ed è citabile.**

### 8.2 Esiste un teorema per cui l'errore cresce con l'AREA MANCANTE

Gli stessi autori dimostrano che l'errore introdotto dal fmap layer sotto parzialità è
*proporzionale all'area mancante*, e distinguono esplicitamente il loro risultato da quello di
Rodolà et al.:

> «While the error in [39] relates to the length of the cut, our analysis exhibits a relation to
> the missing area.»

Due conti teorici indipendenti — lunghezza del taglio (Rodolà) e area mancante (Bracha) — che
predicono entrambi il nostro asse `crop`. La nostra misura del 18 agosto ($\lambda_{128}$ che si
sposta del $16\%$ sul crop, contro il $17\%$ previsto da Weyl sull'area persa del $14{,}6\%$) è
l'istanza discreta della seconda. Trasferisce: l'oggetto è lo spettro troncato, che abbiamo anche
noi, non il fmap layer.

Nota su cosa **non** trasferisce: la loro soluzione è abbandonare il fmap layer e fare matching
diretto di feature con loss di Gromov più un regolarizzatore che preserva l'area, ottenendo SOTA
su SHREC'16 HOLES sopra anche i metodi supervisionati. Noi non abbiamo un fmap layer da
abbandonare. Codice e dati: `github.com/ABracha/DirectMatchNet`, dataset `PFAUST`.

### 8.3 Perché il pozzo era destinato a fallire, secondo il campo

Luo & Chen, *From Feature Learning to Spectral Basis Learning* (arXiv:2603.23383), nella
rassegna, elencano l'Hamiltoniano di Choukroun fra le «specialized variations of the Laplacian
eigenbasis» e concludono:

> «Despite their efficacy, these methods are fundamentally constrained by their reliance on
> axiomatic and fixed basis functions, which precludes the use of data-driven strategies for
> further basis optimization.»

Il nostro pozzo è un'istanza di una limitazione già nota, non un incidente. Da mettere accanto
all'avvertimento di Choukroun che il potenziale amplifica il rumore di discretizzazione
(sezione 1.1 sopra): due letture della letteratura che, prese prima, avrebbero risparmiato
l'esperimento.

### 8.4 L'alternativa costruttiva: APPRENDERE il potenziale invece di disegnarlo

Lo stesso paper sostituisce la base fissa con $\Psi_k = \Phi_k G$, dove $G = \mathrm{diag}\{g_i\}$
sono «inhibition functions» apprese che sopprimono selettivamente i modi. In concreto
$\Psi_k = \Phi_k e^{-T}$ con $T = \mathrm{diag}\{t_1,\dots,t_k\}$ parametri appresi, e — dettaglio
che conta —

> «The coefficients $\{t_i\}$ are initialized to zero, ensuring that the inhibition function acts
> as an identity mapping at the onset of training, thus preserving all original basis functions
> equally.»

È il pozzo, ma appreso invece che disegnato, **e inizializzato al no-op**. Il nostro partiva da
una perturbazione grande ($c = 10^{10}$, ROI che tratteneva il $22$--$50\%$ della superficie), che
è il contrario di questa disciplina. Se si riprende quella strada, si riprende così.

### 8.5 Una tensione di progetto da dichiarare, non da ignorare

La loro diffusione è deliberatamente *eigenvalue-agnostic*. Criticano il kernel di calore
standard perché

> «since the eigenvalues $\Lambda_k$ serve as fixed weights for the diffusion time,
> high-frequency components are often excessively suppressed regardless of their task-relevance»

Ma eigenvalue-agnostic significa **appaiare per indice**, e il teorema di Rodolà dice che sotto
parzialità appaiare per indice è la corrispondenza sbagliata: la diagonale è inclinata di un
fattore pari al rapporto di aree. Il loro rimedio e la nostra normalizzazione d'area toccano la
stessa riga di codice e tirano in direzioni opposte sulla parzialità. Loro riportano guadagni su
*topological noise* e casi non isometrici, che non è la nostra parzialità. **Adottarlo alla
cieca disferebbe ciò che `pot_area` sta misurando.** Va deciso con un numero, non per analogia.

Stessa linea, stesso gruppo: Luo et al., *Deep Frequency-Aware Functional Maps*
(arXiv:2402.03904), che lamenta che «different frequency information is treated [equally]» e
introduce un operatore di filtro spettrale appreso.

### 8.6 Gli altri, in breve

- **DPFM** (arXiv:2110.09994) — già in `paper/partial_fmaps.pdf` come scheda 4.5.
- **Spatially and Spectrally Consistent Deep Functional Maps** (arXiv:2308.08871) — consistenza
  ciclica come prior su collezioni di forme. Rilevante se un giorno si allena su collezioni.
- **Denoising Functional Maps** (arXiv:2503.01845, Bonn/MPI) — modelli di diffusione come prior
  sulle mappe; alta visibilità.
- **DiffuMatch** (arXiv:2507.23715) — prior spettrali category-agnostic.
- **Hyper-Network Neural Functional Maps** (arXiv:2606.30131) — dichiara la parzialità come
  scenario ancora aperto nel 2026.
- **DeepShapeMatchingKit** (arXiv:2604.10377) — analisi delle implementazioni open source e
  solver accelerato. Da leggere se servisse un confronto di runtime credibile con il campo.
- **Volumetric Functional Maps** (arXiv:2506.13212) — estensione al volume, fuori scope.
- **Integrating Optimal Transport and Functional Maps** (arXiv:2403.01781) — il più votato del
  gruppo, OT al posto del solver ai minimi quadrati.

### 8.7 Conseguenze per noi, in ordine

1. `pot_area` non è una nostra invenzione ma l'allineamento a una convenzione del campo, e questo
   la rende più facile da difendere e più imbarazzante da aver mancato. Citare Bracha et al. per
   la pratica e Rodolà et al. per la teoria.
2. Il difetto del frame `maxabs` (misurato: scala che si sposta di $\pm 0{,}034$ per identità
   sotto crop) non ha una controparte in letteratura perché nessuno usa `maxabs`. È un difetto
   nostro, e il braccio `pot_rms` serve a quantificarlo.
3. Se si torna sul pozzo, si torna nella forma appresa e inizializzata a identità (8.4), non in
   quella assiomatica che ha già fallito.
4. La diffusione eigenvalue-agnostic è un'opzione reale ma antagonista alla normalizzazione
   d'area sulla parzialità (8.5). Non provarle insieme senza separarle.
