arcana-novai@Arcana-NovAi:~/Documents/GitHub/Xoe-NovAi$ tree -L 3 -a
.
├── app
│   └── XNAi_rag_app
│       ├── chainlit_app.py
│       ├── config_loader.py
│       ├── crawl.py
│       ├── dependencies.py
│       ├── healthcheck.py
│       ├── logging_config.py
│       ├── main.py
│       ├── metrics.py
│       └── verify_imports.py
├── config.toml
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.chainlit
├── Dockerfile.crawl
├── .dockerignore
├── .env
├── .gitignore
├── Makefile
├── README.md
├── requirements-api.txt
├── requirements-chainlit.txt
├── requirements-crawl.txt
├── scripts
│   ├── create_structure.sh
│   ├── ingest_library.py
│   ├── query_test.py
│   ├── setup-new-stack.sh
│   └── validate_config.py
└── tests
    ├── conftest.py
    ├── test_crawl.py
    ├── test_healthcheck.py
    ├── test_integration.py
    └── test_truncation.py
