from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from utils.model import load_model, latest_modified_weight

app = Flask(__name__)
api = Api(app)
model = load_model(latest_modified_weight())


class WineTasty(Resource):

    def post(self):
        posted_data = request.get_json()

        assert 'fixed acidity' in posted_data
        assert 'volatile acidity' in posted_data
        assert 'citric acid' in posted_data
        assert 'residual sugar' in posted_data
        assert 'chlorides' in posted_data
        assert 'free sulfur dioxide' in posted_data
        assert 'total sulfur dioxide' in posted_data
        assert 'density' in posted_data
        assert 'pH' in posted_data
        assert 'sulphates' in posted_data
        assert 'alcohol' in posted_data

        pred = model.predict([
            list(posted_data.values())
        ])[0]

        return jsonify({'prediction': {'class': str(pred)}})


api.add_resource(WineTasty, '/tasty')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
