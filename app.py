from flask import Flask, request, render_template
import psycopg2
import os
from dotenv import load_dotenv

if os.path.exists("link_to_database.env"):
    load_dotenv("link_to_database.env")

app = Flask(__name__)

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]

        try:
            connection = psycopg2.connect(DB_CONNECTION_STRING)
            cursor = connection.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                age INT,
                gender VARCHAR(50)
            );
            """
            cursor.execute(create_table_query)

            insert_query = "INSERT INTO people (name, age, gender) VALUES (%s, %s, %s);"
            
            cursor.execute(insert_query, (name, age, gender))
            
            connection.commit()
            print("Success! One person have been securely saved to the remote server.")

            cursor.close()
            connection.close()

        except Exception as e:
            print(f"Something went wrong: {e}")

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
            print("Database connection closed cleanly.")

        return "Saved successfully! Nice"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)