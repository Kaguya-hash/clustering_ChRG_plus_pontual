from flask import Flask, request, render_template
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv("link_to_database.env")

app = Flask(__name__)

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]

        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                age INT,
                gender VARCHAR(50)
            );
        """)

        cur.execute(
            "INSERT INTO people (name, age, gender) VALUES (%s, %s, %s)",
            (name, age, gender)
        )

        conn.commit()

        cur.close()
        conn.close()

        return "Saved successfully!"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)