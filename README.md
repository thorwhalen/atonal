# atonal

Tools for mathematical analysis of 12‑tone musical structures.


## 1) Context and goals

This package focuses on **pitch‑class set theory** (Forte-style set classes and
related relationships) as a practical, data-oriented toolkit.

The core abstraction is a **pitch class**: pitches modulo 12, typically labeled
`0..11` (0=C, 1=C♯, …, 11=B). A **pitch‑class set** (pc‑set) is an **unordered**
collection of pitch classes, ignoring octave and enharmonic spelling.

This is the same abstraction pipeline described in 
[A mathematics of musical harmony](https://github.com/thorwhalen/atonal/blob/main/misc/docs/A%20mathematics%20of%20musical%20harmony.md):

- move from sound → pitch,
- discretize to 12‑TET,
- quotient by octave/enharmonic equivalence,
- study **subsets** of the 12 pitch classes,
- and then factor by musically meaningful symmetries (especially
	**transposition** $T_n$ and **inversion** $I$).

Under $T/I$ equivalence, many different pc‑sets collapse into a single set class
represented by a canonical **prime form**. That enables combinatorial analysis
of harmonic “shapes” independent of key.

`atonal` aims to make these objects and relations easy to compute, and (crucially)
easy to *analyze as data*.


## 2) Data in `misc/tables`

The folder `misc/tables/` contains precomputed tables (Parquet format) that
represent:

- **Nodes**: every subset of the 12 pitch classes ($2^{12} = 4096$ rows)
- **Links**: several useful relations between those subsets (edge tables)

These tables are generated in `misc/making_pitch_class_sets_data.ipynb` using functions from
`atonal/base.py`.

### 2.1 Nodes table: `pitch_class_sets/twelve_tone_sets.parquet`

This file contains 4096 rows, one for each pitch‑class subset.

You can load it with pandas:

```python
import pandas as pd

nodes = pd.read_parquet('misc/tables/pitch_class_sets/twelve_tone_sets.parquet')
```

Field guide:

- `id_` (int, 0..4095)
	- A 12‑bit bitmap encoding the set: bit `i` is 1 iff pitch class `i` is present.
	- This is a compact single‑identifier for joins across all link tables.

- `pcset` (sequence of ints)
	- The sorted pitch classes present in the set.
	- Example: `(0, 4, 7)` is the C major triad pc‑set.

- `cardinality` (int)
	- The number of pitch classes in the set (size of `pcset`).

- `contains_zero` (bool)
	- Convenience feature: whether pitch class 0 is present.
	- Useful because many canonical representatives are transposed to include 0.

- `complement_id` (int)
	- Bitmap id of the complement set with respect to the 12‑tone universe.
	- Computed as `4095 ^ id_`.

- `prime_form` (sequence of ints)
	- Canonical representative of the set class under $T/I$ equivalence.
	- Computed by comparing best normal order of the set and its inversion,
		transposed to start at 0, then choosing the lexicographically smallest.
	- Example: major/minor triads share prime form `[0, 3, 7]`.

- `forte_name` (str or null)
	- Forte set‑class label for cardinalities 3..9 when defined (e.g. `"3-11"`,
		`"6-Z29"`).
	- For 7..9 the name is derived via complement labeling (same suffix).
	- Null outside 3..9.

- `is_forte_set` (bool)
	- True when the row’s `pcset` is itself a canonical prime‑form representative
		for a Forte-labeled class (i.e. `pcset == prime_form` and that prime form
		is in the module’s Forte lookup table).
	- Practically: “one chosen representative per set class” (where available).

- `interval_vector` (length‑6 sequence of ints)
	- The interval‑class vector $(ic_1, ic_2, ic_3, ic_4, ic_5, ic_6)$.
	- Each entry counts unordered pitch‑class pairs at that interval class,
		where interval class is the shortest distance mod 12.
	- This is a key invariant for characterizing harmonic color.

- `is_t_symmetric` (bool)
	- True if the set is invariant under some non‑zero transposition $T_n$.
	- Symmetric sets have fewer than 12 distinct transpositions.

- `z_correspondent_prime_form` (sequence of ints or null)
	- Prime form of the Z‑correspondent, if any.
	- Z‑related sets share the same `interval_vector` but are not $T/I$ equivalent.
	- Reported only when a correspondent exists in the module’s Forte catalog.

- `z_correspondent_forte_name` (str or null)
	- Forte label for the Z‑correspondent (when available).

- `n_T` (int)
	- Number of distinct transpositions in the set’s $T$ orbit.
	- For symmetric sets this is less than 12.

- `n_I` (int)
	- Number of distinct inversions in the set’s $I T$ orbit.

- `kh_size` (int)
	- Size of the set’s Forte **$K_h$ subcomplex** under the operational
		definition used in `atonal.base.kh_complex_size()`.
	- Intuition: counts how many sets are linked to the nexus set by reciprocal
		inclusion relations involving both the set and its complement.

- `hexachord_combinatorial` (str or null)
	- Combinatoriality label for hexachords (cardinality 6):
		- `"a"`: all‑combinatorial (both T and I combinatorial)
		- `"p"`: prime combinatorial (T combinatorial only)
		- `"i"`: inversion combinatorial (I combinatorial only)
	- Null for non‑hexachords.

- `max_T_invariance` (int) and `max_T_invariance_n` (int)
	- Over all non‑trivial transpositions $T_n$ (n=1..11),
		`max_T_invariance` is the maximum number of pitch classes left invariant
		under the best such $T_n$; `_n` stores the argmax.

- `max_I_invariance` (int) and `max_I_invariance_n` (int)
	- Analogous invariance count for inversion‑transpositions $I T_n$ (n=0..11).

- `best_normal_order` (sequence of ints)
	- Forte “packed” normal order: a most‑compact cyclic ordering of the set.
	- This is the canonical pre‑prime template used before transposing to 0.

- `label` (str)
	- A human-friendly label for plotting/inspection.
	- Typically includes `forte_name` when present and the concrete `pcset`.


### 2.2 Links tables: `misc/tables/pitch_class_sets/links/*.parquet`

These Parquet files are edge tables that relate nodes by `id_`.

Load example:

```python
import pandas as pd

nodes = pd.read_parquet('misc/tables/pitch_class_sets/twelve_tone_sets.parquet')
links = pd.read_parquet('misc/tables/pitch_class_sets/links/immediate_subset_links.parquet')

edges = links.merge(nodes.add_prefix('src_'), left_on='source', right_on='src_id_') \
						.merge(nodes.add_prefix('tgt_'), left_on='target', right_on='tgt_id_')
```

Link tables and meanings:

#### `complement_links.parquet`

- Columns: `source`, `target`
- Meaning: pairs each set with its 12‑tone complement.
- Construction: `target = 4095 ^ source`.
- Notes:
	- Stored once per undirected pair (2048 edges).

#### `immediate_subset_links.parquet`

- Columns: `source`, `target`
- Meaning: Hasse diagram edges of the subset lattice.
	- `target` is obtained by adding exactly one pitch class to `source`.
	- `cardinality(target) = cardinality(source) + 1`.
- Musical intuition: moving by one pitch class is the smallest possible change
	in “pitch material” at the pc‑set level.

#### `ti_equiv_links.parquet`

- Columns: `source`, `target`
- Meaning: connects pc‑sets that belong to the same $T/I$ set class
	(i.e. share `prime_form`).
- Notes:
	- This is an undirected, all‑pairs‑within‑class expansion.
	- Use it to move between different transpositions/inversions of the “same”
		harmonic shape.

#### `z_relation_links.parquet`

- Columns: `source`, `target`, `interval_vector`, `prime_form_a`, `prime_form_b`
- Meaning: connects sets that share the same `interval_vector` but have different
	`prime_form` (i.e. are not $T/I$ equivalent).
- Notes:
	- If you want the classical Forte Z‑relations, you will typically filter this
		table to cardinalities 3..9 and/or to rows where `forte_name` is non‑null.

#### `rp_triads.parquet` and `rp_tetrads.parquet`

- Columns: `source`, `target`, `max_common`
- Meaning: Forte **$R_p$ similarity** relation, computed within a fixed
	cardinality.
	- Two sets of size $n$ are $R_p$‑similar when, under some $T_n$ or $I T_n$,
		they share exactly $n-1$ pitch classes.
	- For `rp_triads`, `max_common` is 2.
	- For `rp_tetrads`, `max_common` is 3.
- Musical intuition: “closest neighbors” within a fixed chord size, allowing
	transposition/inversion.


## 3) Main functions used to build the data

The tables above are built in `misc/making_pitch_class_sets_data.ipynb` using these functions
from `atonal/base.py`:

- `build_pcset_nodes_df()`
	- Generates the 4096‑row nodes table, computing prime forms, interval vectors,
		invariances, $K_h$ sizes, and labels.

- `build_immediate_subset_links_df()`
	- Generates the subset lattice edges (add one pitch class).

- `build_complement_links_df()`
	- Generates complement pairs.

- `build_ti_equivalence_links_df(nodes_df)`
	- Connects sets with the same `prime_form` (same $T/I$ class).

- `build_z_relation_links_df(nodes_df)`
	- Connects sets with the same `interval_vector` but different `prime_form`.

- `build_rp_similarity_links_df(nodes_df, cardinality=3|4)`
	- Generates $R_p$ similarity graphs for triads/tetrads.

Core primitives used in these builders (also useful directly):

- `int_to_pcset()` / `pcset_to_int()` for conversion between bitmap ids and
	explicit pc‑sets.
- `prime_form()`, `best_normal_order()`, `interval_vector()`.
- `forte_name()` and `z_correspondent_prime_form()`.


---

### Development note

The precomputed tables in `misc/tables/` are meant to be usable as-is for
analysis and graph workflows; the builders are provided so you can regenerate
or extend the data with your own fields/relations.
