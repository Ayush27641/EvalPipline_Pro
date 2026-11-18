-- Executed automatically by the pgvector/pgvector:pg16 image on the FIRST start of an
-- empty data volume (docker-entrypoint-initdb.d). Safe to re-run manually.
--
-- The application also issues `CREATE EXTENSION IF NOT EXISTS vector` on startup (Step 2)
-- so non-compose databases (CI service container, managed Postgres) work as well.

CREATE EXTENSION IF NOT EXISTS vector;
