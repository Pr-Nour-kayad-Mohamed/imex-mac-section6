# IMEX-MAC 2D — Section 6 Archival Package (v1)

Auteur : NOUR KAYAD, Mohamed — MENFOP, République de Djibouti
ORCID : 0009-0002-1443-7599 — E-mail : mnkr20232030@gmail.com
Date d'archivage : 2026-11-26

## Contenu
- `imex_mac_section6_solver.py` : solveur IMEX-MAC 2D complet (advection MUSCL
  limitée, diffusion thermique anisotrope implicite, projection MAC à jauge
  nulle, champ Phi à relaxation implicite, positivité T contrôlée) et suite
  numérique de la Section 6.
- `section6_results.json` : résultats canoniques de la suite (exécution de
  référence du 2026-11-26).
- `section6_sweeps.csv` : mêmes résultats au format tabulaire.
- `bilan_section6_manuscrit.pdf` : manuscrit de bilan publiable.
- `SHA256SUMS.txt` : empreintes SHA-256 des fichiers du paquet.

## Reproduction
Python >= 3.10 avec numpy et scipy.

    python3 imex_mac_section6_solver.py

Durée de référence : environ 5 s sur poste de bureau ; plusieurs minutes sous
Termux ARM64.

## Portée des affirmations (IMPORTANT)
Les résultats concernent exclusivement le cas synthétique Taylor-Green
thermique périodique. Ils établissent : la faisabilité numérique locale,
l'incompressibilité discrète à la précision machine (≈ 4×10⁻¹⁵), une
conservation d'énergie de l'ordre de 1e-5 en relatif, et le non-recours au
mécanisme de positivité thermique sur la configuration testée. Ils
n'établissent PAS : une validation industrielle ou matérielle, une
certification de sûreté, une supériorité sur d'autres solveurs, ni une
validité physique du modèle sous-jacent. Les balayages en Péclet et en CFL
rapportés sont non discriminants en l'état (voir manuscrit, section Limites).
