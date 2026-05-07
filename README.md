# `emg2text`

If you use this dataset, code, or audio samples, please cite:

```bibtex
@inproceedings{nonInvasiveElectromyographic,
  title     = {Non-invasive electromyographic speech neuroprosthesis: a geometric
perspective},
  author    = {Gowda, Harshavardhan T. and others},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

We present a neuromuscular speech interface
that translates silently voiced articulations
directly into text. We record surface elec-
tromyographic (EMG) signals from multiple
articulatory sites on the face and neck as
participants *silently* articulate speech, enabling
direct EMG-to-text translation. Such an
interface has the potential to restore commu-
nication for individuals who have lost the
ability to produce intelligible speech due to
laryngectomy, neuromuscular disease, stroke,
or trauma-induced damage (e.g., radiotherapy
toxicity) to the speech articulators. Prior work
has largely focused on mapping EMG collected
during *audible* articulation to time-aligned
audio targets or transferring these targets to
*silent* EMG recordings, which inherently
requires audio and limits applicability to
patients who can no longer speak. In contrast,
we propose an efficient representation of high-
dimensional EMG signals and demonstrate
direct sequence-to-sequence EMG-to-text
conversion at the phonemic level without
relying on time-aligned audio.

```text

Download the data from the below links.

To obtain the data:

    1. Download the data from: https://osf.io/bgh7t/ (under Files/Box).

    (https://osf.io/bgh7t/files/box - the OSF site shows them as 0B since they are hosted externally. You can still click on individual files and download them. 0B is misleading.)

    2. Unzip it into the root project directory so it looks like this:
        emg2speech/
        ├── DATA/
        │   ├── ckptsLargeVocab/ # Trained check points for data large-vocab (Also,   all LMs + WFST graphs).
        │   ├── ckptsNatoWords/ # Trained check points for data NATO-words.
        │   ├── ckptsSmallVocab/ # Trained ckeck points for data small-vocab
        └── ├── emg2qwerty/ # Trained check points for emg2qwerty.
        ``` ├── dataLargeVocab.pkl # large-vocab EMG data.
            ├── labelsLargeVocab.pkl # Labels for large-vocab.
            ├── dataSmallVocab.npy # small-vocab EMG data.
            ├── labelsSmallVocab.npy # Labels for small-vocab.
            ├── Audio # Synthesized personalized audio file samples.
            ├── Subject 1 # NATO words, subject 1
            ├── Subject 2 # NATO words, subject 2
            ├── Subject 3 # NATO words, subject 3
            ├── Subject 4 # NATO words, subject 4
    