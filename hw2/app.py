from flask import Flask, request, jsonify

from stud.implementation import build_model

app = Flask(__name__)
model = build_model("cpu")


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def annotate(path):

    try:

        json_body = request.json
        sentences_s = json_body["sentences_s"]
        predictions_s = model.predict(sentences_s)

    except Exception as e:

        app.logger.error(e, exc_info=True)
        return {
            "error": "Bad request",
            "message": "There was an error processing the request. Please check logs/server.stderr",
        }, 400

    return jsonify(sentences_s=sentences_s, predictions_s=predictions_s)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345)
