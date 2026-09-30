# OE1 stage 1 on SC3 (second trial, A vs B, paired per muon)

Scripts of stage 1 of the director's "Protocolo del segundo ensayo OE1 v1.0": 13 single-thread
runs of `muography_sim` in list mode (geo-train-06 affected case and geo-train-01 control, variants
A and B, 3 replicas of 100 000 primaries, plus an A/A repeat of geo-train-01 r1) executed in ONE job
on ONE node. What has and has not run on SC3: the Release builds of A2 and B2, the standalone tests, the
synthetic list-mode harness and the command-line and non-transport checks (`--write-primaries`,
`--check-primaries`, the exit codes, the checks that end a run before `BeamOn`) have run there through `srun`,
in development folders (`~/oe1-trabajo/etapa1-dev`). **No `sbatch` job has been submitted and no transport has
run on SC3 with these binaries**: every `sbatch` command below is yours to launch.

Python is 3.9 compatible (SC3 login node: 3.9.25, numpy 1.23.5). The compute nodes may carry numpy
2.0.2; the tools work with both. SciPy is optional and only cross-checks the built-in statistics.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `make_p2_hashes.sh` | local, in the repo at the A2 commit | sha256 list that ships with the P2 patch |
| `prepare_sources.sh` | SC3 login node, once | creates `~/oe1-trabajo/etapa1`, copies the pilot trees A and B, applies P2, verifies hashes and that A2 and B2 differ only in `src/DetectorConstruction.cc` |
| `audit_rng.sh` | local (WSL) or SC3, no allocation needed | static audit of the code P2 adds: no random number is drawn, no navigator is queried; optional `nm` over the objects |
| `env_stage1.sh` | sourced by every script | lmod (`gnu14/14.2.0`, `geant4/11.4.2`), records and unsets the five G4 override variables, ignores `DRYRUN` and `STAGE1_SKIP_MODULES` inside a SLURM job, `stage1_free_mb_home`, `stage1_require_job_or_rehearsal`, writes `entorno.txt` |
| `build_stage1.sh` | `srun` on `legacy` | Release build of A2 and B2, `compilacion.txt`, fails on any warning |
| `preparacion.sbatch` | `sbatch`, 2 CPU, 15 min | K2 (`--check-geometry`), K1 (six lists, A2 against B2), list validator, K-ROI |
| `make_manifest.py` | login node | `manifiesto-corridas.tsv` (13 rows) and `entradas-esperadas.sha256`; refuses lists whose N is not 100 000 and catalog or table files that are not the frozen ones |
| `pruebas_tecnicas.sbatch` | `sbatch`, 6 CPU, 20 min | T-L, T-R, T-M, T-S, T-F, T-P: 24 120 transports outside the 13 runs. **DV1 gate: refuses to start (exit 4) unless `herramientas/DV1-aprobado.txt` exists and contains the token `DV1-APROBADO`** (step 9) |
| `principal.sbatch` | `sbatch`, 13 CPU, 30 min | the 13 runs, `xargs -P 13`, one status file per run; refuses to start with less than 800 MB free |
| `run_timed.py` | used by the sbatch files | runs a command and writes `recursos.json` (wall, user, sys, max RSS) |
| `post_run.sh` | login node, after the job | `sacct`, `gzip -n -9 -k`, sha256 of raw and compressed files, one tar per attempt |

**Real work runs under SLURM only.** `preparacion.sbatch`, `pruebas_tecnicas.sbatch` and, without `DRYRUN=1`,
`principal.sbatch` refuse to start outside a SLURM job (exit 2, nothing created). Only a local rehearsal tree,
marked by `~/oe1-trabajo/etapa1/.arbol-de-prueba` (never create it on SC3), is exempt.

The Python tools live in `scripts/analysis/`: `oe1_stage1_common.py`, `oe1_stage1_roi.py`,
`oe1_stage1_lists.py`, `oe1_stage1_synthetic.py`, `oe1_stage1_tecnicas.py` and
`oe1_stage1_analysis.py`. They are copied to `herramientas/` together with the files above.

## Layout of `~/oe1-trabajo/etapa1`

`fuentes/{A2,B2}` sources, `build/{A2,B2}` binaries, `herramientas/` everything above,
`analisis/roi/` the exported mask, `listas/prep-NN/{A2,B2}` the six production lists,
`control-geometrico/prep-NN` K2 outputs, `verificacion/prep-NN` K1 and K-ROI evidence,
`pruebas/tecnicas-NN` the DV1 job, `corridas/intento-NN` the 13 runs, `slurm/` job output,
`registro/` patch, hashes and diffs. Every folder is created with a plain `mkdir` (never `-p`,
never reused): a second use of a number stops with exit code 3. A failed step is redone under a new
number (`PREP_NN`, `TEC_NN`, `INTENTO`, edited in a COPY of the script) and nothing is deleted.

## Order of use and gates

The gates are the plan's: nothing in the next block starts until the previous one is green.

**Before SC3 (local)**

1. P2 patch and its hash list, with the repo at the A2 commit (LF only):
   `git diff 9520170 <A2 commit> -- scripts/simulation/geant4_app > P2-instrumentacion.diff`, then
   `bash make_p2_hashes.sh P2-instrumentacion.diff <repo root> P2-instrumentacion.sha256`.
2. R_int export from the pilot's `roi-mascaras.npz`. It refuses anything that is not the frozen mask
   (195 pixels, rows 17-50, columns 131-138, npz sha256 `7546773b...`):
   `python3 oe1_stage1_roi.py export --npz roi-mascaras.npz --out-dir OUT --copy-npz`.

**SC3, login node**

3. `bash prepare_sources.sh P2-instrumentacion.diff P2-instrumentacion.sha256 --base-a ~/oe1-trabajo/registro/A-fuentes.sha256 --base-b ~/oe1-trabajo/registro/B-fuentes.sha256`
   (run it from a folder outside `etapa1`). Needs at least 1 GB free in `/home` (`quota -s`); it aborts
   if `etapa1` already exists, and an abort after the folder was created (failed patch, hash or diff
   check) leaves it half built: inspect `registro/`, then remove `etapa1` by hand (it holds no results
   yet) before running it again. It refuses (exit 2, before creating anything) a patch that touches a file
   the plan keeps unchanged: `DetectorConstruction.*`, `CMakeLists.txt`, `GeometryConfig.*`, `FluxTables.*`,
   `StackingAction.*`, `GeometryDiagnostics.*` or `tables/`. **Gate K0 (part):**
   `registro/preparacion-fuentes.txt` says `resultado=OK`, `registro/diff-A2-B2.txt` lists only
   `src/DetectorConstruction.cc`, and, when the pilot's `~/oe1-trabajo/registro/diff-A-B.txt` exists (it is only
   read), the line `diff_A2_B2_frente_al_del_piloto` says the hunks of diff(A2, B2) are identical to that
   file's once the header lines (paths and dates) are masked (`registro/comparacion-diff-A-B.txt` keeps the
   differences otherwise, exit 5).
4. Copy the tools to `herramientas/` and the four exported files to `analisis/roi/`; check
   `sha256sum -c R_int.mask.txt.sha256`.
5. Build: `srun -p legacy -N1 -c8 --mem=8G -t 00:30:00 bash ~/oe1-trabajo/etapa1/herramientas/build_stage1.sh`.
   **Gate:** `build/compilacion.txt` says `resultado=OK`, 0 warnings, Release flags, two binary hashes.
6. **Gate K0 (rest):** the C++ tests of the plan's step 3 and the static audit. These scripts do not run them;
   inside an `srun` allocation, with `gnu14` and `geant4/11.4.2` loaded:
   `bash tests/build_tests.sh OUT` (standalone tests, release and ASan/UBSan, 0 warnings with `-Wall -Wextra
   -Wshadow -pedantic`, test R1 over six seeds, `OUT/pruebas-cruces.json` with the seeds, the results and the
   deviations from the plan's literal numbers), `bash tests/g4/build_g4_tests.sh OUT` (synthetic list-mode
   harness: 13 scenarios, A/A, strict-warning pass over the Geant4-linked sources; the scenarios `bad-variant`,
   `bad-run-manager` and `bad-both` also assert that `main()` would exit 4 before `BeamOn`),
   `bash tests/g4/early_exit.sh BIN_A2 BIN_B2 geo-train-06.json TABLES OUT` (the same on the REAL binaries: exit 4
   with an empty `--out` and no `RUN DONE`, plus the order of the exit codes; no transport),
   `bash tests/g4/legacy_inertness.sh ORIG_APP OUT` (pre-P2 tree against P2, no transport). The Python tests:
   `cd scripts/analysis && python -B -m unittest discover -s tests -t .`.
   **Static RNG audit** (plan step 2, "Cierre"): `bash audit_rng.sh --diff registro/P2-instrumentacion.diff
   --build-dir build/A2` (and `build/B2`) audits the lines P2 ADDS: no `G4UniformRand`, `flat(`, `shoot`, `RandFlat`,
   `RandGauss`, `G4QuickRand`, `G4Navigator` or `SafetyHelper` (and their relatives) in code; the only engine calls
   allowed are `setTheSeed`, `setTheSeeds`, `getTheEngine`, `put` and `name` (plus `setSeeds` and `setTheEngine`
   in `muography_sim.cc` for the private engine of `--write-primaries`) and the single `fresh.flat()` line of its
   7-draws proof; with `--build-dir` it also runs `nm -C --undefined-only` over the P2 objects. Exit 0 = PASS.
   The audit is name-based: it does not see an implicit draw through the conversion operator of the engine
   (`double u = *engine;`, `double u = fresh;`), on a declared engine variable or through an alias declared with
   `auto`, nor draws from
   non-CLHEP generators such as `std::mt19937`; the run-time tests T-R, T-M and T-P (step 9) cover PROT:17 on the
   real binary.
   Without the patch file: `--base PILOT_geant4_app --app P2_geant4_app`.
7. `cd ~/oe1-trabajo/etapa1 && sbatch --parsable herramientas/preparacion.sbatch`.
   **Gates K2, K1, K-ROI:** `verificacion/prep-NN/estado-preparacion.txt` says `resultado=OK`.
   B2 must show `status=OK`, `overlapping=0`, two `ANOMALY-CSG ... clipped=no` lines on geo-train-06
   and no `G4Exception`; A2 on geo-train-06 is expected to reproduce the historical overlap (reported as
   a `NOTA`, not a failure). The lists become read-only (`chmod a-w`).
8. Manifest, from the lists just validated:
   `python3 herramientas/make_manifest.py --lists-dir listas/prep-NN/A2 --catalog-dir ~/oe1-trabajo/catalogo --tables-dir ~/oe1-trabajo/tablas --bin-a build/A2/muography_sim --bin-b build/B2/muography_sim --roi-mask analisis/roi/R_int.mask.txt --out herramientas/manifiesto-corridas.tsv --inputs-sha256 herramientas/entradas-esperadas.sha256`.
   It never overwrites; it writes exactly 13 rows. It exits 2 unless every list has N = 100 000 (in its
   sidecar and in its rows), the catalog files have the frozen SHA-256 (geo-train-06 `c8f6c135...`,
   geo-train-01 `ff3c69ab...`) and the three generator tables have the SHA-256 the pilot ran with
   (`row_weights_30d.txt` `93ce67ab...`, `p_cdf.txt` `7161c8ad...`, `meta.json` `7ac4b6a9...`).
9. **DV1 gate.** The technical job is 24 120 transports outside the 13 runs of the protocol (1.86 % of the
   1.3e6 of the stage; all labelled "prueba tecnica"): T-L 8 000, T-R 2 000, T-M 2 000, T-P 2 000, T-S 120,
   T-F 10 000. The script **refuses to start (exit 4, nothing created)** unless
   `herramientas/DV1-aprobado.txt` exists, is not empty and contains the token `DV1-APROBADO`. Once DV1 is
   decided, create the file with
   `printf 'DV1-APROBADO %s %s\n' "<YYYY-MM-DD>" "<reference>" > ~/oe1-trabajo/etapa1/herramientas/DV1-aprobado.txt`.
   The file and its SHA-256 are copied into `pruebas/tecnicas-NN/`. Then
   `sbatch --parsable herramientas/pruebas_tecnicas.sbatch`. It also refuses to start with less than 400 MB
   free in `/home` (exit 5). Without this job PROT:17 (logging consumes no random numbers) is verified only
   statically and T-L, T-R, T-M, T-P, T-S and T-F are not executed. **Gate:**
   `pruebas/tecnicas-NN/resultados/estado-pruebas-tecnicas.txt` says `resultado=OK`. At most one job in the
   queue at a time.
10. **Freeze commit** in both repos (analysis script, R_int specification, manifest and expected
    hashes) and record the UTC time. From this point none of those files may change; if one must, the
    attempt is void: fix, repeat steps 5-9 and start a new attempt.
11. `DRYRUN=1 bash herramientas/principal.sbatch` (on the login node, outside SLURM) must print the 13
    commands, each with `[inputs=ok]`, and the free space in `/home`; it writes nothing. Then, from
    `~/oe1-trabajo/etapa1` (the relative `--output` needs it):
    `sbatch --parsable herramientas/principal.sbatch`. `INTENTO` is edited in a copy per attempt. Without
    `DRYRUN=1`, `bash herramientas/principal.sbatch` outside a SLURM job refuses to start (exit 2, nothing created).
    Exit codes: 0 all 13 OK, 1 a run failed, 2 pre-flight failure (missing or changed input, **less than 800 MB
    free in `/home`**, the refusal above), 3 the attempt folder exists. Inside a SLURM job `DRYRUN` and
    `STAGE1_SKIP_MODULES` are IGNORED (sbatch exports the submitting shell's environment) and the job says so on
    stderr and in `entorno.txt` / `lanzamiento.txt` (`interruptores_ignorados_en_trabajo`). The values that the five G4
    override variables (`G4FORCENUMBEROFTHREADS`, `G4FORCE_EVENTS_PER_TASK`, `G4FORCE_GRAINSIZE`,
    `G4RUN_MANAGER_TYPE`, `G4FORCE_RUN_MANAGER_TYPE`) had when the script started, and after the modules were
    loaded, are recorded in `entorno.txt` before `env_stage1.sh` unsets them.
    A run whose `*** G4Exception` count differs from the manifest's is a **FAIL for B** (B must have none) but
    only **evidence for A** (`avisos.txt` in the run folder, a note in the status file, a non-blocking K6 in the
    analysis): the expected `GeomVol1002=1` of A is a historical diagnostic.
    **Disk:** the 13 runs peak at about 0.5 GB and `post_run.sh` adds the compressed copies and the tar (about
    0.95 GB in all), while `/home` had about 1.5 GB free on 2026-09-30 (`quota -s`; check it before submitting).
    Only ONE production attempt fits, and `~/oe1-trabajo/etapa1-dev` (development builds, lists and outputs that
    are not evidence) should be freed first if a second attempt or the technical job is
    needed.
12. `bash herramientas/post_run.sh JOBID NN`, then
    `rsync -av sc3-guane:oe1-trabajo/etapa1/corridas/intento-NN.tar* .` and `sha256sum -c intento-NN.tar.sha256`.
    The raw `_hist.csv` and `_muones.csv` stay on SC3 until the review; the tar carries their `.gz`.

**Analysis (local, or on the login node if the raw files are there)**

13. `python3 oe1_stage1_analysis.py --attempt intento-NN --roi-mask R_int.mask.txt --roi-frozen R_int-congelada.json --roi-npz roi-mascaras.npz --lists-dir DIR --out NEWDIR`.
    Contracts K0 and K6-K9 run first and are written to `contratos.json`. Exit 0: blocking contracts hold;
    3: a blocking contract failed and every output is stamped "BLOQUEADO -- no interpretable"
    (`--diagnostic-only` still writes the stamped tables); 2: unusable input (a missing file, including
    `--roi-npz`; an unusable manifest; a `--roi-frozen` that is not a JSON object with the keys K0 reads,
    `texto.sha256`, `hash_contenido.valor` and `origen.sha256_archivo`; a `--roi-npz` that is not a readable
    `.npz` with the five mask arrays; a `--replicas` that is not a list of integers), all checked before the output
    folder is created, so none leaves an empty folder.
    K8 and K9 (b) do not block but stop any attribution to the overlap until explained.
    - **Lists (K0, blocking).** Each of the six lists the manifest names must be found (`--lists-dir` is searched
      by file name first) and must have exactly the SHA-256 of its manifest row: "lista localizada ==
      manifiesto". A list that does not match is never used (not for the recomputation below, not for the
      straight-line check, not for the initial energies of the tables).
    - **Fingerprints (K6, K9, blocking).** In every run `prim_fnv` and `rng0` must be 16 hexadecimal digits, none
      of them zero, and N different values each (an empty or constant column is identical in A and B, and in
      an A/A repeat, so K7, K8 and K9 (a) would pass on nothing). `prim_fnv` is **recomputed from the rows of
      the located list for every id** (the recipe below, vectorised) and must equal the column: equality
      between A and B alone would not notice that both ran the wrong list. K9 (a) (`prim_fnv` and `rng0` equal
      for every id) is blocking for the geo-train-06 pairs **and** for the control pairs.
    - **Lateral displacement (K6, K9 (c)).** K6 reports the lateral displacement instead of requiring
      `max_lat_mm < 1e-6 mm` (plan correction C1: the pre-to-post chord is not the navigated path). The
      corrected binary always writes the columns `max_lat_z_mm` and `sum_lat_z_mm` and the aggregates
      `counts.max_lat_z_mm` / `counts.max_sum_lat_z_mm`: when the meta declares the aggregates the columns are
      mandatory (a log without them fails K6), every row of a run with a region has both values,
      `max_lat_z_mm <= max_lat_mm`, and the aggregates equal the column maxima. `sum_lat_z_mm` is the **scale
      of the error, not a bound**: the accumulated transverse displacement of the steps near Z. At an incidence
      angle theta the length error is about lat / sin(theta); for a chord that skims a surface it grows like
      sqrt(2 R lat) (R <= 86.3 m here), so it is reported as at most about 2 sqrt(2 R_max lat); a chord that
      penetrates Z by less than `max_lat_z_mm` may be classified differently from the real path. The table
      `cruces-union.csv` carries it as `escala_error_lz_A_mm` / `escala_error_lz_B_mm`. K9 (c) (in B,
      `lz_a0` and `lz_rock` <= 1 um) stays blocking except for a violation whose size stays within that scale,
      `lz_a0 + lz_rock <= 2 sqrt(2 R_max sum_lat_z)` with R_max = 72.9^2 / 61.6 = 86.27 m (the largest radius of
      curvature of the dense-body ellipsoid) and `sum_lat_z` in metres (chord lengths can clip Z spuriously):
      that is the non-blocking class "explicado por desplazamiento lateral", listed with its ids in
      `contratos.json`, and it holds back the attribution until read. A log of an older binary (no
      aggregates in the meta) is listed as "no verificable" and does not block, and explains nothing.
    - **Efficiency (K6, non-blocking).** `|z_eficiencia_vs_campana| > 5` (binomial sigma against the campaign's
      efficiency from `campaign-manifest.csv`, view 0: 27209253 / 36174382 = 0.752169 for geo-train-06 and 0.73795 for
      geo-train-01; 0.752094 was the pilot's 3e6-event value) is flagged for reading.
    - **Ignored keys (K7, K8).** The comparisons ignore exactly these top-level keys of `meta.json`, an explicit
      list and not a substring rule: `wall_seconds`, `seconds_per_event`, `rusage_user_cpu_seconds`,
      `rusage_system_cpu_seconds`, `rusage_max_rss_kb`. Both K7 and K8 write the ones they ignored in their
      evidence (`claves_de_tiempo_ignoradas`); K8 also ignores `oe1_etapa1.binary_sha256`, which differs between
      the A and B executables, `oe1_etapa1.variant` and the FNV of the log.
    - **Statistics.** The pooled `intervalo_no_acota_el_efecto` is always `true` (stage 1 is not sized, PROT
      section 4) and its note asserts no inequality. `cotas_superiores_clopper_pearson` gives the exact
      two-sided 95 % Clopper-Pearson limits of p_A and p_B from the pooled counts, labelled as data (SigmaY and N
      measured in this stage) and assumption (independent trials, one probability); no bound on Delta is
      derived from them.
    - **Table of runs.** `cpu_s` and `rss_kb` use the `rusage_*` keys of the meta (the process itself, from its
      start; kB) and fall back to `recursos.json` (`run_timed.py`, the child); `fuente_cpu` and `fuente_rss` say
      which one each row used (`meta.rusage`, `recursos.json` or `NA`).
    It takes about 3-4 minutes for the 13 runs (the FNV-1a-64 of the muon logs is pure Python);
    `--skip-muon-fnv` skips it and that check is then reported as not verified.
    Decompress the `.gz` files first and check them against `SHA256SUMS-crudos.txt`.

## What the binary does in list mode (after the fresh C++ review of 2026-09-30)

Command line and exit codes of `muography_sim --primaries LIST ...`:

- `--region-pair I,J|none` is **required** when the geometry has two or more anomalies: `I,J` = conduit,
  dense body or cavity (geo-train-06: `0,1`); `none` waives the region on purpose (geo-train-01: no
  anomalies, no region). Without it the binary exits 2 before any run manager exists. The sbatch scripts
  always pass it: `principal.sbatch` (`run_one`) derives it from the case, `pruebas_tecnicas.sbatch` passes
  `0,1` for its geo-train-06 lists.
- Exit codes, in the order in which `main()` can produce them (also in the header of `muography_sim.cc`):
  2 the strict parser rejects the command line; 1 the list, the mask or the geometry cannot be read or does not
  match its SHA-256, or the list disagrees with the geometry, view, case code or replica; 2 the `--region-pair`
  rule of the geometry; 1 the provenance hashes cannot be taken, or `--out` is missing, not writable or already
  holds a result; **4 right after `runManager->Initialize()`, BEFORE `BeamOn`**; the transport; 4 a run check
  failed at the end of the run; 0 everything holds.
- **Checks before `BeamOn` (plan S3).** The run manager class and `--variant` against the geometry that was
  built need no transport, so they are evaluated right after `runManager->Initialize()`
  (`oe1::CheckBeforeTransport`). If either fails the process prints an `OE1-PRERUN-CHECK ... status=FAIL` line
  on stdout and the reason on stderr and exits 4 **before `BeamOn`: no transport runs and no output file is
  written**. A passing run prints `OE1-PRERUN-CHECK ... status=OK` before `RUN DONE`. The same two checks are
  repeated at the end of the run as a second line of defence (they set the error flags below and the run
  still exits 4). The run manager must be `G4TaskRunManager` (`run_manager_not_task`; e.g.
  `G4RUN_MANAGER_TYPE=Serial` gives the sequential `G4RunManager` and is refused, which is why
  `env_stage1.sh` unsets that variable) and, with a region pair, variant A needs the conduit to be a `G4Tubs`
  and the dense body a `G4Ellipsoid` while variant B needs the conduit, the earlier anomaly, to be a
  `G4SubtractionSolid` (`variant_geometry_mismatch`).
- The exit 4 of the end of the run comes from the error flags, which are also written to
  `oe1_etapa1.errors`: ids incomplete or duplicated, histogram or detected count not reconstructed, a
  worker-thread count other than 1, a worker engine other than MixMaxRng, an aborted event, an output that could
  not be written, and the two checks above.
- `--muon-log full|min`. `full` (default) accumulates, per primary step, the chord crossings of the region
  Z, the energy and material fields and the lateral displacement. `min` still runs
  `EventAction::OnPrimaryStep` on every primary step: it keeps the O(1) status bookkeeping there (step
  count, last process, last step status, last post-step volume and kinetic energy, needed to classify the
  muons that do not reach the scoring plane) and skips only the chord / region / energy / material /
  lateral-displacement accumulation. The two levels give the same identity, state and bin columns and the
  same random-number use (test T-R).

Per-muon log `<stem>_muones.csv` (49 columns): the 47 columns of the first delivery, then `max_lat_z_mm` and
`sum_lat_z_mm`.

- The pre-step to post-step chord is **not** the path the muon followed: WentzelVI displaces the post-step
  point sideways (up to 0.99 x safety), also with `MuHadLateralDisplacement = false`. `max_lat_mm` is the
  largest displacement over all primary steps (a diagnostic; it is above 1e-6 mm in about 89 % of the
  muons). `max_lat_z_mm` and `sum_lat_z_mm` are the largest and the summed displacement over the steps
  **near Z**: the chord meets both primitives (conduit cylinder, dense-body ellipsoid) grown by 1 m
  (`kNearZMarginM`; the largest displacement measured next to Z is about 10 mm). Every chord that meets Z is
  near. `sum_lat_z_mm` is the accumulated transverse displacement of those steps: a **scale of the
  uncertainty of the chord-based lengths in Z, not a bound**. At an incidence angle theta the length error is
  about lat / sin(theta); for a chord skimming a surface it grows like sqrt(2 R lat) (R <= 86.3 m here), so
  report it as at most about 2 sqrt(2 R_max lat). A chord that penetrates Z by less than `max_lat_z_mm` may
  be classified differently from the real path. Both columns are empty when the log is `min` or the run has
  no region, and 0 when a muon has no step near Z.
- `crossed = 1` always implies `n_steps_z >= 1` and non-empty `mat_in` / `mat_out`: a step that is deep
  inside Z but not longer than 2 eps (2 x 1e-6 m; for example a 1 um step) enters the chord-length sums
  (`lz_m`, ...) but sets no categorical field.

Meta `<stem>.meta.json`:

- `oe1_etapa1.binary_sha256` (of `/proc/self/exe`; differs between the A and B binaries by construction),
  `oe1_etapa1.inputs.geometry_json.{file,sha256}` and `oe1_etapa1.inputs.tables.{row_weights_30d.txt,
  p_cdf.txt, meta.json}` (SHA-256 of the geometry JSON and of the three table files, file names only).
  `principal.sbatch` compares all five with the hashes it computed before the launch.
- `oe1_etapa1.counts`: `max_lat_mm`, `max_lat_z_mm`, `max_sum_lat_z_mm`, `near_z_steps`, `near_z_muons`
  (`null` when the log is `min` or the run has no region). `oe1_etapa1.region.near_margin_m`,
  `oe1_etapa1.region.explicit_none`, `oe1_etapa1.checks.variant_matches_geometry` (`null` when not checked).
- Top level, next to `wall_seconds`: `rusage_user_cpu_seconds`, `rusage_system_cpu_seconds` and
  `rusage_max_rss_kb` = `getrusage(RUSAGE_SELF)` of the process from its start (geometry construction
  included, unlike `wall_seconds`), because `sacct` cannot separate the 13 processes that `xargs` starts inside
  one job. `oe1_stage1_common.TIMING_META_KEYS` (an explicit list of these three keys plus `wall_seconds` and
  `seconds_per_event`) removes them from the A/A and A/B comparisons. Legacy (non-list) runs do not write them.

`prim_fnv` of the per-muon log is the FNV-1a-64 of the primary as Geant4 holds it at `BeginOfEventAction`:
the little-endian bytes of `x, y, z` (float64, mm), `dx, dy, dz` (float64), `ekin` (float64, MeV) and `pdg`
(int32), in that order. From a list row (`x_m, y_m, z_m, dx, dy, dz, ekin_gev, pdg`) it can be recomputed
offline:

```python
inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz)           # CLHEP unit(): each component times 1/norm, not divided
payload = struct.pack('<7d', x_m*1000.0, y_m*1000.0, z_m*1000.0, dx*inv, dy*inv, dz*inv, ekin_gev*1000.0)
payload += struct.pack('<i', pdg)                      # m -> mm, GeV -> MeV
h = 0xcbf29ce484222325                                 # FNV-1a-64
for b in payload:
    h = ((h ^ b) * 0x100000001b3) & (2**64 - 1)
```

(`x_m*1000.0` is the double product, as in `G4ThreeVector(x_m*m, ...)`. Multiplying by `inv` and dividing by the
norm differ in the last bits, so the multiplication form matters.)

## Failures

- Infrastructure (timeout, node, quota): rerun the affected unit in `intento-NN+1` with a `MOTIVO.md`;
  every retry adds transports and is reported.
- Code or inputs: fix, rebuild, repeat K0 to T-F, rerun all 13 with the new hashes.
- The scripts exit non-zero on any failed check; SLURM alone would report `COMPLETED 0:0`.

## What was and was not exercised

**On SC3, through `srun` only (never `sbatch`, never a transport)**, gcc 14.2.0, Geant4 11.4.2, in development
folders of `~/oe1-trabajo/etapa1-dev` (new folders for every round, nothing reused or deleted):

- first delivery and the fresh C++ review (plan corrections C1, W1-W3, S2, S4, S6-S9; folders `*-fix1`, jobs
  30543-30548): Release builds of A2 and B2 with 0 warnings; the standalone tests (release and ASan/UBSan, 0
  warnings, test R1 over six seeds); the synthetic harness (12 scenarios); the legacy-inertness probe; the
  command-line exit codes; `--write-primaries` N = 50 by A2 and B2 byte-identical and `--check-primaries`
  output identical;
- final fix round (plan S3 and the reviewers' suggestions; folders `*-fix2`, jobs 30549-30554): Release builds of
  A2 (job 30550, sha256 `a07d83c4...0728f5`) and B2 (job 30551, `ae761866...996852`) with 0 warnings; the
  command-line script (job 30552: 0 mismatches, among them the legacy command lines and `--check-geometry`
  byte-identical to the pilot builds, `--write-primaries` N = 50 by A2 and B2 byte-identical,
  `--check-primaries` output identical, and `tests/g4/early_exit.sh`: 32 checks on the real binaries, exit 4
  before `BeamOn` with an empty `--out`, no transport); `audit_rng.sh` over the P2 sources and the `nm` output of
  both builds (findings: 0); the synthetic harness with its 13 scenarios, the legacy-inertness probe and the
  standalone tests in release and sanitizer builds with 0 warnings, R1 skipped (job 30553; the crossing detector
  did not change in this round and R1 over six seeds passed locally); the Python unit tests (213, 9 skipped) on a
  compute node (job 30554: Python 3.9.25, numpy 2.0.2) and on the login node (Python 3.9.25, numpy 1.23.5,
  SciPy absent); and `prepare_sources.sh` on the login node in a fake HOME with COPIES of the real pilot bundles
  A and B (result `OK`, diff(A2, B2) identical to the pilot's `registro/diff-A-B.txt`; the A-to-B patch, which
  touches `DetectorConstruction.cc`, is refused with exit 2 before anything is created).

**Locally (WSL Ubuntu, Geant4 11.4.2 built there, gcc 13.3; development data, not evidence).** The same tests, plus
eight mutations of the P2 sources that the harness detects (each one makes at least one scenario fail), a
mutation that disables the exit before `BeamOn` (`tests/g4/early_exit.sh` fails on it: the run starts and writes
its files) and an 18-case mutation suite of the RNG audit (15 mutations that it refuses; the unchanged control, a
comment-only change and a mention inside a string literal, which it accepts). Small transports of the technical
lists only (2000 muons, source seed 2026092990, both cases, A2 and B2) and of the 60-muon synthetic list: the
production `run_one` on the five T-F runs, the analysis (`CONTRATOS: OK` with
`prim_fnv` recomputed from the list rows for every id, equal to what the C++ binary logged), the T-F check, and
the warning rule of `run_one` (an A mismatch is evidence, a B mismatch is a FAIL). With a fake HOME and a
stand-in `muography_sim` that writes the outputs of the corrected binary: `bash -n` of the 13 shell scripts,
`make_manifest.py` (its refusals), `DRYRUN`, the disk guard, the SLURM-switch guard and the override record (32
checks); the DV1 gate and the disk guard of `pruebas_tecnicas.sbatch`, the technical job end to end, 13
production runs, the warning rule and the analysis (21 checks); and `prepare_sources.sh` with copies of the pilot
bundles (25 new checks and the 15 earlier ones). Python 3.12/numpy 2.0.1/SciPy 1.14, 3.13/numpy 2.4/SciPy 1.16
(Windows) and 3.12/numpy 2.0.2 (WSL, SciPy absent, 7 tests skipped): 213 tests each at the time; after the
review fixes (S2, S6, the `--roi-npz` and `--replicas` checks), 223 tests pass on Windows (Python 3.13.5) and in
WSL (Python 3.12.3, 7 skipped). The quota helpers (`stage1_free_mb_home`, `free_mb_home`, `post_run.sh`) were also
run against a stub that mirrors the real `quota -s`: an exceeded quota prints `*` and exits 1 (they read -520 MB),
and a quota that never answers is cut at 30 s (they fail closed; `post_run.sh` reports `?`).

**Not exercised:** a transport on SC3 with these binaries; any `sbatch` job (lmod inside a batch job, the queue,
`MaxSubmitPU`); the technical job end to end with the real binaries (T-L needs the pilot binaries, which exist only
on SC3; with the stand-in simulator it ran to `resultado=OK`); the 13-run production job with lists of 100 000
muons after the last changes (it ran locally with the earlier binaries, 13 of 13); T-P after them; `post_run.sh`
and `build_stage1.sh` against the real `sacct` and `srun`.

**Disk.** The 13 runs peak at about 0.5 GB and `post_run.sh` adds the compressed copies and the tar: about
0.95 GB in all. On 2026-09-30 `/home` had about 1.46 GB free (`quota -s`: check it before submitting;
`principal.sbatch` refuses below 800 MB and `pruebas_tecnicas.sbatch` below 400 MB). Only ONE production attempt
fits. `~/oe1-trabajo/etapa1-dev` (0.37 GB of development builds, lists and outputs, none of it evidence) should be
freed first if a second attempt or the technical job is needed.
