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
