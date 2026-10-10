# paper/: come si compila

## `main_cvpr_draft.tex` (bozza CVPR 2027, formato review a due colonne)

- Kit autori: `cvpr_kit/` (kit ufficiale di CVPR 2026 come segnaposto; fonte, sha256 e licenza in
  `cvpr_kit/README.md`). Figure: `imgs/`. Bibliografia: `references.bib` + `refs_cvpr_add.bib`.
- Sul cluster: `sbatch /home/create.aau.dk/ga41wf/containers/texbuild/scripts/compile_paper.sbatch` (partizione
  `prioritized`, TeX Live 2026 in `containers/texlive_latest.sif`). Il PDF va in `paper/build/` (non versionato);
  log, testo per pagina e anteprime PNG in `containers/texbuild/out/run_<job>/`. Lo script fa anche una passata
  diagnostica con `\hfuzz=1pt`, perché `cvpr.sty` alza `\hfuzz` a 30pt e nasconderebbe tabelle e figure che sbordano.
- Altrove: `latexmk -pdf main_cvpr_draft.tex` dentro `paper/` (servono `latexmk`, pdfLaTeX e i pacchetti
  `to-be-determined`, `nicefrac`, `placeins`, `microtype`).

## `main_short.tex` (il testo NeurIPS 2026 inviato, invariato)

Richiede due file che non sono nel repo:

- `neurips_2026.sty`: dallo zip ufficiale
  https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
  (sha256 dello zip `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9` l'11 ottobre 2026;
  sha256 del file `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`). Non è versionato: il file
  non dichiara una licenza di ridistribuzione. Sul cluster ce n'è una copia identica in
  `/home/create.aau.dk/ga41wf/containers/texbuild/inputs/neurips_2026.sty`.
- `checklist.tex`: la checklist compilata è in `v2_work/paper_board/checklist.tex` (va copiata accanto al tex).

Le figure di `main_short.tex` sono le stesse di `imgs/`.
