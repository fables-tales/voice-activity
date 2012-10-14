import sqlite3
import sys

if __name__ == "__main__":
    conn = sqlite3.connect("db.sqlite")
    cur = conn.cursor()
    start = float(sys.argv[1])
    end   = float(sys.argv[2])
    filename = sys.argv[3]
    filename = filename.replace("source_audio/", "")
    voice = int(sys.argv[4])
    keyboard = int(sys.argv[5])

    cur.execute("INSERT INTO samples VALUES(?,?,?,?,?)", (filename, start, end, voice, keyboard))
    conn.commit()
