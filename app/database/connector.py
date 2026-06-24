"""
Database Connector Module

Provides a structured SQLite database connector with schema inspection
and query execution capabilities.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, text


class DatabaseConnector:
    """
    SQLite connector handling database connections.
    Provides utility hooks to feed text-to-SQL pipelines explicit column schema configurations.
    """

    def __init__(self) -> None:
        """Initialize database connections to the local SQLite database."""
        # Locate job_fallback.db in the project root folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(project_root, "job_fallback.db")
        self.db_path = db_path
        self.connection_string = f"sqlite:///{db_path}"
        self.engine = create_engine(self.connection_string)
        logging.info(f"Connecting to SQLite database at: {db_path}")

    def execute_query(
        self,
        query: str,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        Executes raw SQL query and returns rows as a list of dicts.
        """
        try:
            with self.engine.connect() as conn:
                if timeout:
                    conn = conn.execution_options(timeout=timeout)

                result_proxy = conn.execute(text(query), params or {})

                if result_proxy.returns_rows:
                    columns = list(result_proxy.keys())
                    rows = [
                        dict(zip(columns, row))
                        for row in result_proxy.fetchall()
                    ]

                    return {
                        "success": True,
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows),
                        "query": query,
                    }
                else:
                    return {
                        "success": True,
                        "row_count": result_proxy.rowcount,
                        "query": query,
                    }
        except Exception as e:
            logging.error(f"SQL execution failure: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
            }

    def get_text2sql_context(self) -> dict:
        """
        Formats database schema, tables, and relationships metadata for text-to-SQL prompts.
        """
        tables_info = []
        relationships = []

        try:
            with self.engine.connect() as conn:
                # 1. Get all user tables
                tables_res = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                )
                tables = [row[0] for row in tables_res.fetchall()]

                for table in tables:
                    table_desc = [f"Table: {table}", "Columns:"]

                    # 2. Get columns
                    cols_res = conn.execute(text(f"PRAGMA table_info({table});"))
                    # Columns fields: cid, name, type, notnull, dflt_value, pk
                    for col in cols_res.fetchall():
                        col_name = col[1]
                        col_type = col[2]
                        is_nullable = "NULL" if col[3] == 0 else "NOT NULL"
                        pk_marker = "PK" if col[5] > 0 else ""
                        col_line = f"  - {col_name} ({col_type}) {pk_marker} {is_nullable}".strip().replace("  ", " ")
                        table_desc.append("  " + col_line)

                    # 3. Get sample data
                    sample_res = conn.execute(text(f"SELECT * FROM {table} LIMIT 3;"))
                    columns = list(sample_res.keys())
                    sample_rows = [
                        dict(zip(columns, r))
                        for r in sample_res.fetchall()
                    ]
                    if sample_rows:
                        table_desc.append(f"Sample data ({len(sample_rows)} rows):")
                        for row in sample_rows:
                            table_desc.append(f"  {row}")

                    # 4. Get row count
                    count_res = conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                    row_count = count_res.scalar() or 0
                    table_desc.append(f"Total rows: {row_count}")

                    tables_info.append("\n".join(table_desc))

                    # 5. Get foreign key relationships
                    fk_res = conn.execute(text(f"PRAGMA foreign_key_list({table});"))
                    # FK fields: id, seq, table, from, to, on_update, on_delete, match
                    for fk in fk_res.fetchall():
                        source_col = fk[3]
                        ref_table = fk[2]
                        ref_col = fk[4]
                        relationships.append(f"{table}.{source_col} -> {ref_table}.{ref_col}")

        except Exception as e:
            logging.error(f"Error fetching database schema info: {e}", exc_info=True)

        return {
            "database_name": "job_fallback.db",
            "schema_name": "main",
            "tables": tables_info,
            "relationships": relationships,
            "sql_dialect": "SQLite",
        }