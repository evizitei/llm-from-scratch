"""Dev server hosting the component visualizers.

Run with:

    uv run python viz/app.py

then open http://127.0.0.1:5001/ and pick a component.
"""

from __future__ import annotations

from flask import Flask, render_template

from viz.components import COMPONENTS


def create_app() -> Flask:
    app = Flask(__name__)

    for component in COMPONENTS:
        app.register_blueprint(component["blueprint"])

    @app.get("/")
    def index():
        return render_template("index.html", components=COMPONENTS)

    return app


app = create_app()


def main() -> None:
    app.run(debug=True, port=5001)


if __name__ == "__main__":
    main()
