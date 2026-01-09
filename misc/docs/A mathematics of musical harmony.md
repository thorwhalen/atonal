# The Mathematics of Musical Harmony: From Sound to Set Theory

## Introduction: A Combinatorial Approach to Harmony

What we explore here is a mathematical—or more precisely, a **combinatorial**—approach to musical harmony. Combinatorics is a branch of discrete mathematics that studies counting, arrangements, and the structure of collections. Anyone with a Western music education knows about scales and chords, and likely understands that musicality is relative: to most ears (except those with perfect pitch), transposing a piece to a higher or lower key, or playing it faster or slower, doesn't fundamentally change its identity. This perceptual reality guides which aspects of music we'll abstract away.

We're focusing specifically on **scales** and **chords**—both of which can be understood as collections of selected pitches, or **notes**. Scales are typically played sequentially (melodically), while chords are played simultaneously (harmonically). However, we'll abstract away this temporal dimension as part of "how we render" a particular selection of notes. Our interest lies in the selection itself, and more precisely, in the relationships between different selections.

To analyze music mathematically, we must first define our fundamental objects of study.

## Step 1: From Sound to Pitch

We begin by reducing all possible sounds to **pitches**—pure frequencies—abstracting away other acoustic properties like timbre (tone color), attack (how a note begins), dynamics (loudness), and duration. While these elements profoundly affect musical expression, they're not our focus in studying harmonic structure.

## Step 2: Discretization via Equal Temperament

Next, we discretize the continuum of possible pitches, restricting ourselves to notes on a **12-tone equal-tempered scale**. In this system, the octave (a 2:1 frequency ratio) is divided into 12 equal steps on a logarithmic scale. 

The frequency of any note can be calculated as:

$$f_n = 440 \times 2^{(n-69)/12}$$

where *n* is the MIDI note number (with 69 corresponding to A4 at 440 Hz), and each integer step represents one semitone. This formula ensures that each semitone represents the same multiplicative interval: the twelfth root of 2 (approximately 1.05946).

Now we're working in a discrete, 12-tone chromatic space.

## Step 3: Octave and Enharmonic Equivalence

We reduce this space further through two equivalence relations:

**Enharmonic equivalence** is straightforward: in equal temperament, C♯ and D♭ produce identical frequencies, so we treat them as the same pitch class. The notation is merely a convention with no acoustic consequence.

**Octave equivalence** is more consequential musically. Octaves do matter enormously for sound: playing C3 with E6 sounds vastly different from (C3, E3)—wide spacing versus close spacing affects both the clarity of harmony and the acoustic fusion of the notes. Similarly, (C1, E1) sounds different from (C3, E3) due to register: lower frequencies create more acoustic "friction" and different degrees of roughness or consonance. 

A complete study could certainly account for these factors. But here we're deliberately simplifying, focusing on what traditional music theory calls **pitch classes**—the essence of "C-ness" or "E-ness" independent of which octave the note appears in. This abstraction captures something fundamental about how we perceive musical structure: a C major chord is recognizably "the same" whether voiced as (C3, E3, G3) or (C2, E4, G5).

## Step 4: Unordered Collections

At this point, we have 12 pitch classes, conventionally labeled C, C♯, D, D♯, E, F, F♯, G, G♯, A, A♯, B (or numbered 0 through 11).

We're interested in studying combinations of these pitch classes—scales and chords—from a harmonic perspective. Thus we make one more crucial abstraction: we disregard **order**.

Order certainly matters when "playing out" a selection of notes. Consider melody: most melodies we encounter use notes from a 7-note major scale (abstracting away the 12 possible transpositions—we could sing any melody higher or lower). For melody, order is essential. In fact, the ordered sequence of notes plus their rhythmic durations essentially *is* what a melody is, though there's still room for expressive variation.

But our focus is on the harmonic quality of note collections themselves—what intervals and relationships exist within them—not the sequential patterns they might form. A C major scale contains the same pitch classes whether we play C-D-E-F-G-A-B ascending or B-G-E-C-A-F-D in some other order. Both contain the same harmonic "material."

## The Power Set: Our Universe of Possibilities

Having made these abstractions, we arrive at what mathematicians call **the power set of a 12-element set**: the collection of all possible subsets of our 12 pitch classes. This includes:

- The empty set (no notes)
- All 12 individual pitch classes (singleton sets)
- All possible dyads (2-note combinations)
- All possible trichords (3-note combinations)
- ...continuing through...
- All possible 11-note collections
- The complete chromatic collection (all 12 notes)

How many such subsets exist? Exactly **4,096** (that is, 2¹²). For each of the 12 pitch classes, we make a binary choice: include it or exclude it. Twelve independent binary choices yield 2¹² possibilities.

This gives us a well-defined universe of objects to study. But 4,096 is still a dauntingly large number. This is where Allen Forte's contribution becomes essential.

## Forte's Contribution: Equivalence Classes and Set Theory

In his seminal 1973 book *The Structure of Atonal Music*, Allen Forte developed a systematic approach to classifying these 4,096 possibilities into musically meaningful categories. While Forte's work was motivated by analyzing atonal music (music without a traditional tonal center), his theoretical framework has proven broadly applicable. Today, his terminology and concepts are used to analyze harmony across many musical systems, from late Romantic chromaticism to jazz to contemporary composition.

The key insight is that we can abstract even further by grouping together subsets that are musically equivalent in how they sound.

### Transpositional Equivalence: Recognizing Patterns Across Keys

The first and most musically obvious equivalence is **transpositional equivalence**. Consider the chord {C, E, G}—a C major triad. Now consider {D, F♯, A}—a D major triad. These are different subsets of our 12 pitch classes, but to any ear, trained or untrained, they sound essentially "the same"—both are major triads. The specific pitches differ, but the **intervallic relationships**—the pattern of distances between the notes—are identical: a major third (4 semitones) followed by a minor third (3 semitones).

This is why you can instantly recognize "Happy Birthday" whether it's sung starting on C, D, or F♯. The melody's essence lies not in the absolute pitches but in the intervals between them. Similarly for harmony, what matters is the collection of intervals present, not which specific pitch classes realize those intervals.

Mathematically, we say two pitch-class sets are **transpositionally equivalent** (or **T-equivalent**) if one can be obtained from the other by transposition—that is, by adding the same number modulo 12 to all elements. The C major triad {0, 4, 7} (using numbers where C=0, C♯=1, etc.) becomes the D major triad {2, 6, 9} by adding 2 to each element (mod 12). We denote this operation as **T₂**, meaning "transpose by 2 semitones."

By grouping all transpositionally related sets together into **equivalence classes**, we dramatically reduce our count from 4,096 sets to a much more manageable number. Each equivalence class represents a distinct harmonic "shape" or "quality"—major triad, minor triad, diminished seventh chord, whole-tone collection, and so on—independent of its absolute pitch location.

### Inversional Equivalence: The Mirror Image

Forte took the abstraction one step further by also considering **inversional equivalence**. Inversion is a mirror operation: imagine the 12 pitch classes arranged around a circle (like a clock face). Inversion reflects them across some diameter of that circle.

Algebraically, inversion around pitch class *x* is defined as: **Iₓ(p) = x - p (mod 12)**. For example, inversion around C (I₀) maps C→C, E→G♯, G→E, etc.

Consider again {C, E, G}, a C major triad. If we invert it around C (apply I₀), we get {0, 8, 5} = {C, Ab, F}—an F minor triad in first inversion, which reordered is {F, Ab, C}, a minor triad. The major and minor triads sound notably different—bright versus dark, happy versus sad in conventional Western terms—yet they're clearly related. They share the same **interval content** but in opposite "directions": where the major triad has a major third on bottom and minor third on top, its inversion has a minor third on bottom and major third on top.

Musically, inversional equivalence is more controversial than transpositional equivalence. Not everyone agrees that major and minor triads should be considered "equivalent"—their affective qualities are quite different. However, Forte argued that for understanding harmonic structure, especially in post-tonal music where traditional major/minor distinctions may not apply, inversional equivalence is useful because inverted sets share the same **interval-class content**: they contain the same multiset of interval types, just realized in different configurations.

An **interval class** (ic) represents the shortest distance between two pitch classes on the circle, ranging from 0 (unison) to 6 (tritone). For example, C to E and E to C both represent interval class 4 (a major third), even though as directed intervals they're different.

When we consider both transpositional and inversional equivalence (denoted **TnI**—transposition by *n* semitones followed by inversion, or equivalently **TnI**, the group of all such operations), we further reduce the number of distinct **set classes**. For sets of cardinality 3 to 9, this dual equivalence system yields Forte's famous catalog of **208 prime forms**—the canonical representatives of each distinct harmonic type, typically written in most compact form starting from 0.

For example:
- **Set class 3-11**: prime form [0,3,7] represents all major and minor triads
- **Set class 4-27**: prime form [0,2,5,8] represents all half-diminished and dominant seventh chords
- **Set class 6-35**: prime form [0,2,4,6,8,10] represents the whole-tone collection

### Why Some Cardinalities Matter Less

While theoretically we could analyze all 4,096 subsets, in practice certain **cardinalities** (set sizes) are less interesting for harmonic analysis:

**Cardinality 0 (the empty set):** Nothing is sounding—there's no harmony to analyze. Musically trivial.

**Cardinality 1 (single pitch classes):** With only one pitch class, there's no harmonic relationship to study. A single note is just a pitch, not a chord. While individual pitches certainly matter in music, they're not the focus of *harmonic* analysis, which by definition concerns relationships between multiple notes.

**Cardinality 12 (the complete chromatic):** The set of all 12 pitch classes. There's only one such set, and it contains every possible interval. It represents total chromaticism—again, musically meaningful in certain contexts, but not a fruitful object for comparative harmonic analysis since there's nothing to distinguish it from.

**Cardinality 11 (eleven notes):** Using 11 of the 12 chromatic notes produces an extremely dense, nearly chromatic sonority. To most ears, such collections sound like "almost everything"—they're hard to distinguish from one another perceptually because they're so saturated. The single *missing* note becomes the salient feature, which is why analysis often focuses instead on the complement (the cardinality-1 set that's absent).

**Cardinalities 2 and 10:** Dyads (2-note sets) are sometimes excluded because they represent single intervals rather than chords—the basic building blocks of harmony rather than interesting harmonic objects in themselves. A single interval doesn't give us much structural richness to analyze. That said, some analyses do consider dyads, especially when they function as important motivic cells in a composition.

Their complements, 10-note sets, are similarly very dense and perceptually undifferentiated—they're "almost chromatic" and hard to distinguish from one another by ear.

The most musically rich and analytically interesting collections tend to be those with **cardinalities 3 through 9**: trichords, tetrachords, pentachords, hexachords, and their complements. These have enough notes to create distinctive harmonic colors and intervallic structures, but not so many that they blur into undifferentiated chromaticism. They correspond to what we actually hear as "chords" and "scales" in music—collections with recognizable character and identity.

## The Result: A Manageable Catalog of Harmonic Archetypes

Through these successive abstractions—octave equivalence, enharmonic equivalence, unordered sets, transpositional equivalence, and inversional equivalence—Forte reduced the seemingly overwhelming space of 4,096 possible pitch-class collections to a finite, musically meaningful catalog. 

Focusing on cardinalities 3-9 and applying T/I equivalence yields **208 distinct set classes**—the fundamental vocabulary of chromatic harmony. Each set class represents a unique harmonic archetype: a particular constellation of intervals that gives rise to a recognizable sonic quality.

This system provides powerful tools for analyzing how composers use harmony, especially in post-tonal music where traditional functional harmony may not apply. We can track which set classes appear, how they're transformed (by transposition or inversion), how they relate to each other through common subsets or complements—all while focusing on the perceptually salient features: the intervallic structures that give each sonority its distinctive sound and character.

The beauty of this approach is that it honors the perceptual reality of how we hear music (recognizing transpositions as "the same," hearing intervallic patterns as fundamental units) while providing rigorous mathematical tools to study the combinatorial structures underlying harmonic choices. We've moved from the infinite continuum of all possible sounds to a finite catalog of about 200 essential harmonic shapes—the fundamental vocabulary that underlies the vast diversity of chromatic musical expression.