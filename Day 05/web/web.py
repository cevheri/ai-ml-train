import sqlite3 as sql
from flask import Flask, request, jsonify, redirect, url_for, render_template

import mysql.connector

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="ai_app_1"
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


# @app.route('/add', methods=['POST'])
# def add():
#     data = request.get_json()
#     conn = sql.connect('database.db')
#     cur = conn.cursor()
#     cur.execute('INSERT INTO user_table(name, email, password) VALUES(?, ?, ?)',
#                 (data['name'], data['email'], data['password']))
#     conn.commit()
#     return redirect(url_for('index'))


@app.route('/add', methods=['POST'])
def add():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    cursor = db.cursor()
    cursor.execute('INSERT INTO user_table(name, email, password) VALUES(%s, %s, %s)',
                   (name, email, password))
    db.commit()
    db.close()
    return redirect(url_for('index'))


@app.route('/userlist')
def userlist():
    return render_template('userlist.html')


@app.route('/users', methods=['GET'])
def users():
    conn = sql.connect('database.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM user_table')
    rows = cur.fetchall()
    conn.close()
    return jsonify(rows)


@app.route('/delete', methods=['DELETE'])
def delete():
    data = request.get_json()
    conn = sql.connect('database.db')
    cur = conn.cursor()
    cur.execute('DELETE FROM user_table WHERE id = ?', (data['id'],))
    conn.commit()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
