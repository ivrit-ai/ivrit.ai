# ivrit.ai

Codebase of ivrit.ai, a project aiming to make available Hebrew datasets to enable high-quality Hebrew-supporting AI models.

Huggingface: https://huggingface.co/ivrit-ai
Paper: https://arxiv.org/abs/2307.08720

# Usage Guidance

See [examples](./examples) for how to use the models.

## Downloading From Sources

### Crowd Recital

- psycopg (the modern version of psycopg2) requires the PostgreSQL client libraries (libpq) to be installed.
- On Ubuntu, this can be done by installing the `libpq-dev` package.
- On MacOS, this can be done by installing the `libpq` library.

Use the following command line to download the recital session data:
```
# Within the active venv
python download.py crowd_recital \
      # Filter by recording language, multiple languages are supported (space separated). Default: he
      --languages he \
      
      # A "postgresql://" connection string to the PG db of the recital system to download from
      --pg-connection-string postgresql://postgres:postgres@server/db-name \
      --output-dir downloads/recital \
      
      # AWS S3 settings of the bucket where the recital session data is stored
      --s3-bucket bucket.name \
      --aws-access-key access-key \
      --aws-secret-key secret-key \
      
      # If the normalization step is to be skipped (text cleanup, align to audio)
      --skip-normalize \
      
      # If to skip the CSV manifest generation step
      --skip-manifest \

      # Limit max sessions to download/process in this run
      --max-sessions 10 \

      # Force downloading already downloaded sessions
      --force-download \
      
      # If specific session ids are to be processed, can specify multiple space-separated session ids
      --session-ids sess1 sess2 \

      # Normalization Configuration

      # CT2 normalization model to use for alignment
      --align-model ivrit-ai/whisper-large-v3-turbo-ct2 \
      
      # Specify "auto" to use cuda if available, otherwise cpu. Can specify multiple
      # devices separated by spaces to increase parallelism. Default: auto
      --align-devices cuda:0 cuda:1 \

      # How many workers per device. This allows to better utilize strong GPUs or CPU multi-cores. Default: 1
      --align-device-density \

      # Failure threshold for alignment - portion of segments with 0 length tolerated before failing the alignment. Default: 0.2
      --failure-threshold 0.2 \

      # Force normalization of already processed entries
      --force-normalize-reprocess \

      # Force recalculation of quality score for aligned transcripts
      --force-rescore
```

### Podcasts / YoutTube (RSS based) sources

- Requires more documentation.

### Knesset Sources

- Requires more documentation.

# Citations

If you use our datasets, the following quote is preferable:

```
@misc{marmor2023ivritai,
      title={ivrit.ai: A Comprehensive Dataset of Hebrew Speech for AI Research and Development},
      author={Yanir Marmor and Kinneret Misgav and Yair Lifshitz},
      year={2023},
      eprint={2307.08720},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
