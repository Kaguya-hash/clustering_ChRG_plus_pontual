from flask import Flask, render_template, request, jsonify
import json
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from sklearn import cluster, datasets
from sklearn.metrics import adjusted_rand_score
import addicionalDatasets
import addicionalClusters
import io, base64
import matplotlib.pyplot as plt
import numpy as np

def dataset_to_png_base64(X):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def clustering_to_png_base64(X, y_pred):
    labels = np.unique(y_pred)
    n = len(labels)

    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    colors = {lab: cmap(i / max(n - 1, 1)) for i, lab in enumerate(labels)}
    colors[-1] = (0.5, 0.5, 0.5, 0.4)

    fig, ax = plt.subplots(figsize=(4, 4))

    for lab in labels:
        mask = y_pred == lab
        name = "Noise" if lab == -1 else f"Cluster {lab + 1}"
        ax.scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.6, color=colors[lab], label=name)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    legend = ax.legend(
    loc="upper left",
    frameon=True, fontsize=6, ncol=max(1, n // 10),
    markerscale=1.2, handlelength=0.8, columnspacing=0.6, handletextpad=0.3,
)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.5)
    legend.get_frame().set_edgecolor("none")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

load_dotenv("link_to_database.env")

app = Flask(__name__)

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

@app.route("/", methods=["GET"])
def get_datasets():

    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT id, dataset_name, python_function_name, parameters_schema, hidden_parameters_schema FROM datasets")
    datasets = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("datasets.html", datasets=datasets)

@app.route("/preview/<dataset_name>", methods=["POST"])
def preview_dataset(dataset_name):
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM datasets WHERE dataset_name = %s", (dataset_name,))
    dataset = cur.fetchone()
    cur.close()
    conn.close()

    user_parameters = request.get_json()

    kwargs = {}
    if user_parameters:
        for param_name, param_data in user_parameters.items():
            kwargs[param_name] = param_data
    if dataset['hidden_parameters_schema']:
        for param_name, param_data in dataset['hidden_parameters_schema'].items():
            kwargs[param_name] = param_data["default"]

    python_function_name = dataset['python_function_name']
    parts = python_function_name.split(".")
    current_obj = globals()[parts[0]]
    for part in parts[1:]:
        current_obj = getattr(current_obj, part)

    print(kwargs)
    print(current_obj)

    X, _ = current_obj(**kwargs)
    plot = dataset_to_png_base64(X)

    return jsonify({"plot": plot})

@app.route("/<dataset_name>/<params_json>", methods=["GET"])
def get_methods(dataset_name, params_json):

    user_parameters = json.loads(params_json)

    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT * FROM datasets WHERE dataset_name = %s", (dataset_name,))
    dataset = cur.fetchone()

    cur.execute("SELECT id, method_name, parameters_schema FROM clustering_methods")
    methods = cur.fetchall()

    cur.close()
    conn.close()

    hidden_parameters_schema = dataset['hidden_parameters_schema']
    user_parameters = json.loads(params_json)

    kwargs = {}
    if user_parameters:
        for param_name, param_data in user_parameters.items():
            kwargs[param_name] = param_data
    if hidden_parameters_schema:
        for param_name, param_data in hidden_parameters_schema.items():
            kwargs[param_name] = param_data["default"]

    python_function_name = dataset['python_function_name']

    parts = python_function_name.split(".")

    current_obj = globals()[parts[0]]

    for part in parts[1:]:
        current_obj = getattr(current_obj, part)

    X, _ = current_obj(**kwargs)
    plot_b64 = dataset_to_png_base64(X)

    return render_template("methods.html", methods=methods, dataset_name=dataset_name, user_parameters=user_parameters, dataset_plot=plot_b64)

@app.route("/<dataset_name>/<params_json>/<method_name>/<params_json2>", methods=["GET"])
def get_result(dataset_name, params_json, method_name, params_json2):
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT * FROM datasets WHERE dataset_name = %s", (dataset_name,))
    dataset = cur.fetchone()

    cur.execute("SELECT * FROM clustering_methods WHERE method_name = %s", (method_name,))
    method = cur.fetchone()

    cur.close()
    conn.close()

    hidden_parameters_schema = dataset['hidden_parameters_schema']
    user_parameters = json.loads(params_json)

    kwargs = {}
    if user_parameters:
        for param_name, param_data in user_parameters.items():
            kwargs[param_name] = param_data
    if hidden_parameters_schema:
        for param_name, param_data in hidden_parameters_schema.items():
            kwargs[param_name] = param_data["default"]

    python_function_name = dataset['python_function_name']

    parts = python_function_name.split(".")

    current_obj = globals()[parts[0]]

    for part in parts[1:]:
        current_obj = getattr(current_obj, part)

    X, _ = current_obj(**kwargs)

    hidden_parameters_schema = method['hidden_parameters_schema']
    user_parameters = json.loads(params_json2)

    kwargs = {}
    if user_parameters:
        for param_name, param_data in user_parameters.items():
            kwargs[param_name] = param_data
    for param_name, param_data in user_parameters.items():
        kwargs[param_name] = param_data
    if hidden_parameters_schema:
        for param_name, param_data in hidden_parameters_schema.items():
            kwargs[param_name] = param_data["default"]

    python_function_name = method['python_function_name']

    parts = python_function_name.split(".")

    current_obj = globals()[parts[0]]

    for part in parts[1:]:
        current_obj = getattr(current_obj, part)

    python_method = current_obj(**kwargs)

    python_method.fit(X)

    if hasattr(python_method, "labels_"):
        y_pred = python_method.labels_.astype(int)
    else:
        y_pred = python_method.predict(X)

    dataset_plot = dataset_to_png_base64(X)
    clustering_plot = clustering_to_png_base64(X, y_pred)

    return render_template(
        "result.html",
        dataset_name=dataset_name,
        method_name=method_name,
        dataset_plot=dataset_plot,
        clustering_plot=clustering_plot,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

