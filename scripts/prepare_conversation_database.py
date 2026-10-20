"""Explicit additive bootstrap using credentials only inside the existing container."""

import argparse
import re
import subprocess

DATABASE_PATTERN = re.compile(r"resolv_conversations(?:_smoke_[A-Za-z0-9_]+)?\Z")


def _execute(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    args = parser.parse_args(argv)
    if not DATABASE_PATTERN.fullmatch(args.database) or len(args.database) > 64:
        parser.error("database must be resolv_conversations or resolv_conversations_smoke_SUFFIX")
    sql = (
        f"CREATE DATABASE IF NOT EXISTS `{args.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;\n"
        f"GRANT ALL PRIVILEGES ON `{args.database}`.* TO '__APP_USER__'@'%';\n"
    )
    # Validate the container's account before substituting into a SQL literal.
    shell = (
        'case "$MYSQL_USER" in ""|*[!A-Za-z0-9_]*) exit 2;; esac\n'
        'sed "s/__APP_USER__/$MYSQL_USER/g" | '
        'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -uroot --batch\n'
    )
    try:
        _execute(
            ["docker", "compose", "-f", "compose.mysql.yaml", "exec", "-T", "mysql", "sh", "-c", shell],
            input=sql,
        )
    except Exception:  # noqa: BLE001 — subprocess stderr can contain credentials
        raise SystemExit("Conversation database preparation failed; check the existing MySQL container.") from None
    print(f"Conversation database prepared (additive): {args.database}")


if __name__ == "__main__":
    main()
