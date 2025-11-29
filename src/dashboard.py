import os
from datetime import datetime

import psycopg2
from flask import Flask, render_template
from dotenv import load_dotenv

load_dotenv() 

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

app = Flask(__name__)


def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


@app.route("/")
@app.route("/alerts")
def alerts_view():
    conn = get_conn()
    cur = conn.cursor()

    # grab latest 100 alerts
    cur.execute(
        """
        SELECT id, ts, src_ip, dst_ip, sport, dport, proto, score
        FROM alerts
        ORDER BY ts DESC
        LIMIT 100
        """
    )
    rows = cur.fetchall()

    cur.close()
    conn.close()

    alerts = []
    for r in rows:
        alerts.append({
            "id": r[0],
            "ts": r[1].strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": r[2],
            "dst_ip": r[3],
            "sport": r[4],
            "dport": r[5],
            "proto": r[6] or "",
            "score": float(r[7]),
        })

    return render_template("alerts.html", alerts=alerts)


if __name__ == "__main__":
    # debug=True is fine for dev, not prod
    app.run(host="127.0.0.1", port=5000, debug=True)
