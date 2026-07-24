# Workflow-CI - Telco Customer Churn

Submission kriteria 3 kelas Membangun Sistem Machine Learning (Dicoding).

## Struktur

- `MLProject/` - MLflow Project dengan entry point `main` untuk retraining otomatis
  - `MLProject` - definisi project dan parameter
  - `conda.yaml` - environment (Python 3.12.7, mlflow 2.19.0)
  - `modelling.py` - pelatihan Random Forest dengan manual logging
  - `telco_preprocessing/` - dataset siap latih (`train.csv`, `test.csv`)
  - `DockerHub.txt` - tautan Docker image publik
- `.github/workflows/ci.yml` - workflow CI: melatih model, menyimpan artefak ke
  repositori, membangun Docker image dengan `mlflow build-docker`, lalu mendorongnya
  ke Docker Hub

## Menjalankan secara lokal

```bash
mlflow run MLProject --env-manager=local
```

## Catatan tracking

Berbeda dengan kriteria 2 yang mencatat ke DagsHub, skrip ini sengaja mencatat ke
`mlruns/` lokal milik runner. `mlflow build-docker` membutuhkan model URI lokal,
dan menggantungkan artefak CI pada token eksternal membuat build gagal ketika
jaringan bermasalah.

## Dataset

Hasil preprocessing dari repositori kriteria 1:
https://github.com/shabir67/Eksperimen_SML_muhammad-shobir-abdussyakur
