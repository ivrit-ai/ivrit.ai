# ivrit.ai

Codebase of ivrit.ai, a project aiming to make available Hebrew datasets to enable high-quality Hebrew-supporting AI models.

Huggingface: https://huggingface.co/ivrit-ai
Paper: https://arxiv.org/abs/2307.08720

# Usage Guidance

## Downloading From Sources

### Crowd Recital

- psycopg (the modern version of psycopg2) requires the PostgreSQL client libraries (libpq) to be installed.
- On Ubuntu, this can be done by installing the `libpq-dev` package.
- On MacOS, this can be done by installing the `libpq` library.

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
