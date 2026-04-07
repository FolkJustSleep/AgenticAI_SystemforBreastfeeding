import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
# TEST
# def get_connection() -> psycopg2.extensions.connection | bool:
#     try:
#         return psycopg2.connect(
#             dbname=os.getenv("DATABASE"),
#             user=os.getenv("DB_USER"),
#             password=os.getenv("DB_PASSWORD"),
#             host=os.getenv("DB_HOST"),
#             port=os.getenv("DB_PORT")
#         )
#     except Exception as e:
#         print(f"Error connecting to PostgreSQL: {e}")
#         return False
    
# PRODUCTION
def get_connection() -> psycopg2.extensions.connection | bool:
    try:
        return psycopg2.connect(
            f"{os.getenv("PSQL_INTERNAL_URL")}"
        )
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return False