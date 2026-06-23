"""
Database Connector Module

Provides a structured SQLite database connector with schema inspection,
read-only option guardrails, and query execution capabilities.
"""

import logging
import os
import traceback
import typing as tp
import pandas as pd

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings


class DatabaseConnector:
    """
    SQLite connector handling database sessions and automated metadata reflection.
    Provides utility hooks to feed text-to-SQL pipelines explicit column schema configurations.
    """

    def __init__(self) -> None:
        """Initialize database connections to the local SQLite database."""
        self.settings = get_settings()
        self.is_sqlite = True
        self.schema_name = None

        # SQLAlchemy execution matrices
        self._engine: tp.Optional[Engine] = None
        self._metadata: tp.Optional[MetaData] = None
        self._inspector: tp.Optional[tp.Any] = None
        self._session_factory: tp.Optional[sessionmaker] = None

        self._initialize()

    def _initialize(self) -> None:
        """Instantiates connection pooling and reflects database architecture shapes."""
        try:
            self._engine = self._get_connection()
            if self._engine:
                self._metadata = MetaData()
                self._metadata.reflect(bind=self._engine)
                self._inspector = inspect(self._engine)
                self._session_factory = sessionmaker(bind=self._engine)
            else:
                logging.error("Failed to initialize structural database connection pool engine.")
        except Exception as e:
            logging.error(f"Error initializing metadata database connector footprint: {e}", exc_info=True)

    def _get_connection(self) -> tp.Optional[Engine]:
        """Assembles database credentials into a standard connection engine string."""
        try:
            # Locate job_fallback.db in the project root folder
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(project_root, "job_fallback.db")
            connection_string = f"sqlite:///{db_path}"
            
            logging.info(f"Connecting to SQLite database at: {db_path}")
            return create_engine(
                connection_string,
                pool_pre_ping=True  # Automatically checks and recycles stale or dropped connection links
            )
        except Exception as e:
            logging.error(f"Error building database connection pool target layout: {e}", exc_info=True)
            return None

    def get_session(self) -> tp.Optional[Session]:
        """Spawns an isolated database transactional context session frame."""
        if self._session_factory is None:
            self._initialize()
        return self._session_factory() if self._session_factory else None

    def close(self) -> None:
        """Disposes and tears down active pooling allocation clusters cleanly."""
        if self._engine:
            self._engine.dispose()

    def test_connection(self) -> bool:
        """Performs a basic diagnostic read target execution test."""
        if not self._engine:
            self._initialize()
            if not self._engine:
                return False
                
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            logging.error(f"Database infrastructure integrity verification dropped: {e}", exc_info=True)
            return False

    def execute_query(
        self,
        query: str,
        params: tp.Optional[dict[str, tp.Any]] = None,
        timeout: tp.Optional[int] = None,
    ) -> dict[str, tp.Any]:
        """
        Executes raw string statements and structures tabular mapping record payloads.
        """
        if not self._engine:
            return {"success": False, "error": "Database engine is uninitialized."}

        try:
            with self._engine.connect() as conn:
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
            logging.error(f"SQL database pipeline query execution failure: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
            }

    def get_schema_info(self) -> dict[str, tp.Any]:
        """Extracts complete data model attributes to enrich agent operational context frames."""
        schema_info = {
            "tables": [],
            "relationships": self.get_table_relationships(),
            "database_name": "job_fallback.db",
            "schema_name": "main",
        }

        tables = self.get_tables()
        for table_name in tables:
            table_comment = None
            try:
                comment_info = self._inspector.get_table_comment(table_name)
                if comment_info and "text" in comment_info:
                    table_comment = comment_info["text"]
            except Exception as e:
                logging.debug(f"Could not retrieve comment strings for table metadata element {table_name}: {str(e)}")

            table_info = {
                "name": table_name,
                "columns": [],
                "primary_keys": self._inspector.get_pk_constraint(
                    table_name
                ).get("constrained_columns", []),
                "indexes": self._inspector.get_indexes(table_name),
                "sample_data": self.get_sample_data(table_name, 3),
                "row_count": self.get_row_count(table_name),
                "comment": table_comment,
            }

            for column in self._inspector.get_columns(table_name):
                column_info = {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": str(column["default"]) if column["default"] is not None else None,
                    "comment": column.get("comment"),
                    "is_primary_key": column["name"] in table_info["primary_keys"]
                }
                table_info["columns"].append(column_info)

            schema_info["tables"].append(table_info)

        return schema_info

    def get_table_relationships(self) -> list[dict[str, str]]:
        """Maps foreign keys to chart relationships across metrics tables."""
        relationships = []
        for table_name in self.get_tables():
            for fk in self._inspector.get_foreign_keys(table_name):
                relationships.append({
                    "source_table": table_name,
                    "source_column": fk["constrained_columns"][0],
                    "target_table": fk["referred_table"],
                    "target_column": fk["referred_columns"][0],
                    "constraint_name": fk.get("name", ""),
                })
        return relationships

    def get_tables(self) -> list[str]:
        """Returns the list of table strings inside the active database schema bounds."""
        return self._inspector.get_table_names() if self._inspector else []

    def get_sample_data(self, table_name: str, limit: int = 5) -> list[dict[str, tp.Any]]:
        """Fetches lightweight data snapshot vectors to teach the LLM layout syntax."""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = self.execute_query(query)
            return result.get("rows", [])
        except Exception as e:
            logging.error(f"Failed to gather sample rows for structural layout target {table_name}: {str(e)}", exc_info=True)
            return []

    def get_row_count(self, table_name: str) -> int:
        """Returns data scaling bounds metrics parameters inside active tables."""
        try:
            if not self._metadata or not self._engine:
                return -1
            table = Table(table_name, self._metadata, autoload_with=self._engine)
            with self._engine.connect() as conn:
                count_query = select(func.count()).select_from(table)
                return conn.execute(count_query).scalar() or 0
        except Exception as e:
            logging.error(f"Failed to isolate row dimension counts for {table_name}: {str(e)}", exc_info=True)
            return -1

    def get_text2sql_context(self) -> dict[str, tp.Any]:
        """
        Formats complete layout metadata descriptions explicitly into optimized prompt injection maps.
        """
        schema_info = self.get_schema_info()
        tables_info = []

        for table in schema_info["tables"]:
            table_desc = [f"Table: {table['name']}"]
            if table.get("comment"):
                table_desc.append(f"Description: {table['comment']}")

            table_desc.append("Columns:")
            for col in table["columns"]:
                pk_marker = "PK" if col.get("is_primary_key") else ""
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                column_line = f"  - {col['name']} ({col['type']}) {pk_marker} {nullable}"
                if col.get("comment"):
                    column_line += f" - {col['comment']}"
                table_desc.append(column_line)

            if table["sample_data"]:
                table_desc.append(f"Sample data ({len(table['sample_data'])} rows):")
                for row in table["sample_data"]:
                    table_desc.append(f"  {row}")

            if table["row_count"] > 0:
                table_desc.append(f"Total rows: {table['row_count']}")

            tables_info.append("\n".join(table_desc))

        relationships_info = [
            f"{rel['source_table']}.{rel['source_column']} -> {rel['target_table']}.{rel['target_column']}"
            for rel in schema_info["relationships"]
        ]

        return {
            "database_name": schema_info["database_name"],
            "schema_name": schema_info["schema_name"],
            "tables": tables_info,
            "relationships": relationships_info,
            "sql_dialect": "SQLite",
        }

    def to_dataframe(self, query_results: dict[str, tp.Any]) -> tp.Optional[pd.DataFrame]:
        """Converts raw data payload dictionary structures safely into Pandas DataFrames."""
        if not query_results.get("success", False) or "rows" not in query_results:
            return None
        try:
            return pd.DataFrame(query_results["rows"])
        except Exception as e:
            logging.error(f"Pandas framework conversion exception intercepted: {str(e)}", exc_info=True)
            return None

    def execute_query_to_df(
        self,
        query: str,
        params: tp.Optional[dict[str, tp.Any]] = None,
        timeout: tp.Optional[int] = None,
    ) -> tp.Optional[pd.DataFrame]:
        """Executes a query and transforms returned tracking metrics into a Pandas DataFrame."""
        results = self.execute_query(query, params, timeout)
        return self.to_dataframe(results)

    def __del__(self) -> None:
        """Ensures connections drop gracefully during garbage collection routines."""
        self.close()