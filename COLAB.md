# Running Group 7 Experiments in Google Colab

1. Open https://colab.research.google.com/
2. **Runtime → Change runtime type → T4 GPU** if available.
3. In a code cell, run:

```python
!git clone https://github.com/vinayreddy508/group7-phishing-cnn.git
%cd /content/group7-phishing-cnn
!pip install -q -r requirements.txt
```

4. Obtain the official PhiUSIIL dataset (UCI #967):

```python
!python scripts/download_data.py
```

5. Verify the pipeline quickly:

```python
!python scripts/run_experiments.py --quick --output runs/smoke
```

6. Run the main experiments (full dataset, 3 seeds):

```python
!python scripts/run_experiments.py --output runs/domain_disjoint
```

7. Inspect `runs/domain_disjoint/` for actual result CSVs, figures, split diagnostics and robustness results. Local outputs are ignored by Git and should not be pushed.

8. Export important final summary tables and charts to your own computer/Drive for the report, without uploading the raw dataset or row-level URL predictions to GitHub.

**Important:** Quick-mode scores are for code debugging only and are not acceptable final report metrics. The full run trains multiple CNNs and may require significant runtime. You can rerun shorter experimental subsets by changing your own local config copy, but document all deviations in the report.

**Troubleshooting:** If the notebook environment already includes an incompatible TensorFlow/Python combination, restart runtime after installing; use Python 3.10–3.12.
