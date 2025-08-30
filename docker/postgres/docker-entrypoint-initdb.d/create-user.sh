#!/bin/bash

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Coздать выделенного пользователя с привилегиями только на чтение.
    --
    -- Внимание! Данный код не закрывает все возможные вектора атаки.
    -- Например, пользователь все еще может выполнить pg_sleep() и спровоцировать DoS.
    -- Для прода нужна более тщательная настройка привилегий и аудит безопасности.
    --
    CREATE ROLE xxplain WITH
        LOGIN
        NOINHERIT
        PASSWORD '$POSTGRES_XXPLAIN_PASSWORD';

    REVOKE ALL ON SCHEMA public FROM public;
    REVOKE ALL ON DATABASE postgres FROM public;

    GRANT CONNECT ON DATABASE postgres TO xxplain;
    GRANT USAGE ON SCHEMA public TO xxplain;

    GRANT SELECT ON ALL TABLES IN SCHEMA public TO xxplain;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT SELECT ON TABLES TO xxplain;
EOSQL
