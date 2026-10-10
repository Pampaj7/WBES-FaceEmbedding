# Kit autori CVPR (LaTeX) usato da `paper/main_cvpr_draft.tex`

**Segnaposto: è il kit di CVPR 2026.** L'11 ottobre 2026 il repository ufficiale non ha ancora una release per
CVPR 2027 (ultima: `CVPR2026-v1(latex)`), e la pagina `https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines`
risponde 404. Quando esce il kit 2027 va sostituito qui, insieme a `\confYear` nella bozza.

## Fonte

- Repository ufficiale: https://github.com/cvpr-org/author-kit
- Release `CVPR2026-v1(latex)`, pubblicata il 2025-09-17, commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`
  ("CVPR 2026 changes by Vladimir Pavlovic").
- Scaricato l'11 ottobre 2026 da
  `https://github.com/cvpr-org/author-kit/archive/refs/tags/CVPR2026-v1(latex).zip`,
  sha256 dello zip `ef672bf4ad1c6801237b957d57b0087d4412e7dee01ad3b512abe82960c6466c`
  (gli zip generati da GitHub non sono garantiti identici nel tempo: fanno fede gli sha256 dei file sotto).
- Il ramo `main` ha commit successivi alla release (l'ultimo il 2026-05-28, "teaser w/ strip and preamble tweaks"),
  non etichettati per nessuna conferenza: non usati.

## File (identici al tag)

| file | sha256 |
|---|---|
| `cvpr.sty` | `2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153` |
| `ieeenat_fullname.bst` | `e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc` |
| `main.tex` (modello) | `251329b8a7ea25f1ba0f5a769f652235e13bc1f43b3e8b1d892e16a005b361f0` |
| `main.bib` (modello) | `1d6d78b4c02eae2607a40e15376f6ac96ef491e311241acd6c8a8a970ac3c1b2` |
| `preamble.tex` (modello) | `f666a85903f49929395951e7afb620e96d4e208334623f23135298b4d7f06228` |
| `rebuttal.tex` (modello) | `29e36ddeb5f9eebe5e62c9df74a24f9f49b587ccfb3710209a22f02239b53ebb` |
| `sec/0_abstract.tex` | `ac059e9cb8595bab8f7795d088d71e321ee46e8397d5cfb56957107d43c868ed` |
| `sec/1_intro.tex` | `55dc0cd862ccfd3c54b63367323669ff1e23d07f76291b35949c7b914c92fbec` |
| `sec/2_formatting.tex` | `bd4233b307bfe3ad791c433e021b00ed4e4017db7adde8ee8e475c267fc7191d` |
| `sec/3_finalcopy.tex` | `ae261b3daaff50cf609bb3ca7dfd030bb1e835fd85aa4fda45cdaf53cc35cc1a` |
| `sec/X_suppl.tex` | `594291f94cf3b925908dc127a93f3525c1998e96abf37a14f69d6077a7e58bd7` |
| `README_upstream.md` (il `README.md` del kit, rinominato) | `d91116b58be33154d195012d7d2ef2be310d308ed23f3803aa6baf2aeb441a5e` |

Non copiati: `.github/workflows/latex-build.yml` (CI del repository del kit) e il suo `.gitignore`.

## Licenza

Il repository del kit non dichiara una licenza (API GitHub: `license: null`, nessun file LICENSE); `cvpr.sty` non ha
intestazione di licenza; `ieeenat_fullname.bst` è di Patrick W. Daly sotto LPPL (intestazione del file). Il kit è
pubblicato da CVPR perché gli autori lo includano nei sorgenti dei propri articoli (arXiv compreso): lo teniamo qui
alla stessa condizione, invariato.

## Uso

`main_cvpr_draft.tex` carica `\usepackage[review]{cvpr_kit/cvpr}` e `\bibliographystyle{cvpr_kit/ieeenat_fullname}`
(percorsi relativi a `paper/`; LaTeX avvisa che il pacchetto si dichiara `cvpr`, avviso innocuo). Compilazione:
`/home/create.aau.dk/ga41wf/containers/texbuild/scripts/compile_paper.sbatch` (Slurm, partizione `prioritized`,
TeX Live 2026 in container); il PDF va in `paper/build/`, che non si versiona.
